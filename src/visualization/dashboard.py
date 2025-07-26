"""
Dashboard Layout Generator
Aggregates multiple plots into a single PNG summary dashboard
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

class DashboardBuilder:
    """
    Combines individual plot images into a grid dashboard.
    """
    def __init__(self, plot_dir: str = "./plots", output_path: str = "./plots/dashboard.png"):
        self.plot_dir = Path(plot_dir)
        self.output_path = Path(output_path)

    def build(self, filenames: list, cols: int = 2, thumb_size: tuple = (400,300), padding: int = 10):
        images = [Image.open(self.plot_dir/f) for f in filenames]
        rows = (len(images)+cols-1)//cols
        w, h = thumb_size
        dashboard = Image.new('RGB', (cols*(w+padding)+padding, rows*(h+padding)+padding), 'white')
        for idx, img in enumerate(images):
            img_thumb = img.resize(thumb_size)
            row, col = divmod(idx, cols)
            x = padding + col*(w+padding)
            y = padding + row*(h+padding)
            dashboard.paste(img_thumb, (x,y))
        dashboard.save(self.output_path)
        print(f"Dashboard saved to {self.output_path}")
