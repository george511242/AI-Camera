import configparser
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from threading import Lock
import time
import fcntl
import os

class ConfigRepository:
    _lock = Lock()
    
    def __init__(self, path: str = '/src/config.ini'):
        self.config_path = Path(path)
        self.lock_path = Path(path + '.lock')
        self.config = configparser.ConfigParser()
        self.config.read(self.config_path)
        self._last_save_time = self._get_file_mtime()
        self._save_interval = 0.1
        self.isSaved = True
        self._file_mtime = self._get_file_mtime()
        self._pending_changes = {}  # 追蹤當前session的更改

    def _get_file_mtime(self):
        try:
            return self.config_path.stat().st_mtime if self.config_path.exists() else 0
        except:
            return 0

    def _acquire_file_lock(self, lock_file):
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (IOError, OSError):
            return False

    def _release_file_lock(self, lock_file):
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except:
            pass

    def _merge_external_changes(self):
        try:
            logging.info("Reloading config from disk to merge external changes")
            self.config = configparser.ConfigParser()
            self.config.read(self.config_path)

            if self._pending_changes:
                logging.info("Re-applying pending changes from current session")
                for section_name, section_data in self._pending_changes.items():
                    if not self.config.has_section(section_name):
                        self.config.add_section(section_name)

                    for key, value in section_data.items():
                        self.config.set(section_name, key, value)
                        logging.info(f"Re-applied session change: [{section_name}] {key} = {value}")

            self._file_mtime = self._get_file_mtime()
            logging.info("Successfully merged external changes")

        except Exception as e:
            logging.error(f"Error merging external changes: {e}")

    def reload_config(self):
        """重新載入配置文件，同時保留尚未寫入的 pending changes"""
        self.config = configparser.ConfigParser()
        self.config.read(self.config_path)
        self._file_mtime = self._get_file_mtime()
        # Re-apply any unsaved in-memory changes so they aren't lost
        if self._pending_changes:
            for section, kvs in self._pending_changes.items():
                if not self.config.has_section(section):
                    self.config.add_section(section)
                for k, v in kvs.items():
                    self.config.set(section, k, v)

    def get_all_configs(self):
        return self.config._sections
    
    def get_config(self, category, key, default = None):
        self._file_mtime = self._get_file_mtime()
        if self._file_mtime > self._last_save_time:
            self.reload_config()
        try:
            return self.config.get(category, key)
        except Exception as e:
            return default

    def _ensure_config_file(self):
        if not self.config_path.exists():
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config_path.touch()

    def set_config(self, category, key: str, value: str):
        self.isSaved = False
        try:
            self.config.set(category, key, value)
        except configparser.NoSectionError:
            self.config.add_section(category)
            self.config.set(category, key, value)

        if category not in self._pending_changes:
            self._pending_changes[category] = {}
        self._pending_changes[category][key] = value

    def save(self):
        if not self.isSaved:
            # 檢查是否需要限制保存頻率
            current_time = time.time()
            if current_time - self._last_save_time < self._save_interval:
                logging.warning("Save operation too frequent, skipping...")
                return False

            with ConfigRepository._lock:  # 使用類鎖來確保線程安全
                lock_file = None
                try:
                    self.lock_path.parent.mkdir(parents=True, exist_ok=True)
                    lock_file = open(self.lock_path, 'w')

                    max_wait = 5
                    wait_time = 0.1
                    total_waited = 0

                    while total_waited < max_wait:
                        if self._acquire_file_lock(lock_file):
                            break
                        time.sleep(wait_time)
                        total_waited += wait_time
                    else:
                        logging.warning("Could not acquire file lock, proceeding anyway")

                    current_mtime = self._get_file_mtime()
                    if current_mtime > self._file_mtime:
                        logging.info("Config file was modified by another process, merging changes")
                        self._merge_external_changes()

                    # 在創建備份之前先驗證當前配置文件是否為空
                    if self.config_path.exists() and self.config_path.stat().st_size == 0:
                        logging.error("Current config file is empty, aborting save operation")
                        return False

                    # 創建臨時文件並寫入新的配置
                    with NamedTemporaryFile(mode='w', delete=False) as temp_file:
                        self.config.write(temp_file)
                    
                    temp_file_path = Path(temp_file.name)
                    
                    # 驗證臨時文件是否正確寫入
                    if temp_file_path.stat().st_size == 0:
                        logging.error("Temporary file is empty, aborting save operation")
                        temp_file_path.unlink()
                        return False

                    # 創建備份文件名，包含時間戳
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    backup_path = self.config_path.with_name(
                        f"{self.config_path.stem}_{timestamp}.bak"
                    )

                    # 如果原始文件存在且不為空，則創建備份
                    if self.config_path.exists() and self.config_path.stat().st_size > 0:
                        shutil.copy2(self.config_path, backup_path)
                        
                        # 驗證備份是否成功
                        if not backup_path.exists() or backup_path.stat().st_size == 0:
                            logging.error("Backup creation failed")
                            temp_file_path.unlink()
                            return False

                    # 將臨時文件移動到目標位置
                    shutil.move(temp_file_path, self.config_path)
                    
                    # 再次驗證最終文件
                    if not self.config_path.exists() or self.config_path.stat().st_size == 0:
                        logging.error("Final config file is empty or missing")
                        if backup_path.exists():
                            shutil.copy2(backup_path, self.config_path)
                        return False

                    # 清理舊的備份文件（保留最近的 5 個備份）
                    self._cleanup_old_backups()

                    self.isSaved = True
                    self._last_save_time = current_time
                    self._file_mtime = self._get_file_mtime()  # 更新文件修改時間
                    self._pending_changes = {}  # 清理已保存的更改記錄
                    return True

                except Exception as e:
                    logging.error(f"Error saving config: {e}")
                    if 'backup_path' in locals() and backup_path.exists():
                        try:
                            shutil.copy2(backup_path, self.config_path)
                            logging.info("Successfully restored from backup")
                        except Exception as restore_error:
                            logging.error(f"Failed to restore from backup: {restore_error}")
                    return False
                finally:
                    if lock_file:
                        try:
                            self._release_file_lock(lock_file)
                            lock_file.close()
                        except Exception as e:
                            logging.warning(f"Failed to release lock: {e}")

                        try:
                            if self.lock_path.exists():
                                self.lock_path.unlink()
                        except Exception as e:
                            logging.warning(f"Failed to remove lock file: {e}")

                    if 'temp_file_path' in locals() and temp_file_path.exists():
                        try:
                            temp_file_path.unlink()
                        except Exception as e:
                            logging.warning(f"Failed to delete temporary file: {e}")

        return True

    def _cleanup_old_backups(self, keep_count=5):
        """清理舊的備份文件，只保留最近的幾個"""
        try:
            backup_pattern = f"{self.config_path.stem}_*.bak"
            backup_files = sorted(
                self.config_path.parent.glob(backup_pattern),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            for backup_file in backup_files[keep_count:]:
                try:
                    backup_file.unlink()
                except Exception as e:
                    logging.warning(f"Failed to delete old backup {backup_file}: {e}")
                    
        except Exception as e:
            logging.error(f"Error during backup cleanup: {e}")