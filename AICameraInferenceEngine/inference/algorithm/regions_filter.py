import cv2
from ..model import Model, Result


class RegionsFilter(Model):
    def __init__(self, regions: list[list[float]]) -> None:
        '''
        region: [
            [x1, y1, x2, y2],
            [x1, y1, x2, y2],
            ...
        ]\n
        '''
        self.regions = regions

    def __call__(self, img: cv2.typing.MatLike, results: 'list[Result]'):
        if not len(results):
            return results

        outputs: list[Result] = []
        for result in results:
            cx, cy = result.center
            if any(
                (region[0] <= cx <= region[2]) and (region[1] <= cy <= region[3])
                for region in self.regions
            ):
                outputs.append(result)
        return outputs
