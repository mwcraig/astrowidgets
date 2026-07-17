from importlib.util import find_spec

import astropy.visualization as apviz
from astropy.table import Table
from matplotlib import colormaps as mpl_colormaps
import numpy as np
import pytest

from astro_image_display_api.api_test import ImageAPITest
from astro_image_display_api import ImageViewerInterface

plotly = pytest.importorskip("plotly",
                             reason="Package required for test is not "
                                    "available.")
import plotly.graph_objects as go  # noqa: E402

from astrowidgets.plotly import ImageWidget  # noqa: E402

HAS_KALEIDO = find_spec("kaleido") is not None


def expected_rgb(arr, cuts, stretch, cmap_name):
    """
    The RGB array the widget should send to the figure for the given
    data, cuts, stretch and colormap name.
    """
    normalized = cuts(np.asarray(arr))
    if stretch is not None:
        normalized = stretch(normalized)
    cmap = mpl_colormaps[cmap_name]
    return (cmap(np.clip(normalized, 0, 1))[..., :3] * 255).astype(np.uint8)


def test_instance():
    image = ImageWidget()
    assert isinstance(image, ImageViewerInterface)


def test_reload_keeps_single_image_trace():
    # Loading images must replace the image trace, never append a second
    # one next to it.
    image = ImageWidget()
    image.load_image(np.zeros((10, 10)), image_label='first')
    image.load_image(np.ones((20, 30)), image_label='second')
    image.load_image(np.ones((20, 30)), image_label='second')

    image_traces = [t for t in image.viewer.data if isinstance(t, go.Image)]
    assert len(image_traces) == 1
    assert np.asarray(image_traces[0].z).shape == (20, 30, 3)


@pytest.mark.parametrize(
    "shape,symbol",
    [("circle", "circle-open"), ("square", "square-open"),
     ("crosshair", "cross-thin-open"), ("plus", "cross-thin-open"),
     ("diamond", "diamond-open")]
)
def test_catalog_markers_use_scatter_shape_size_and_arrays(shape, symbol):
    image = ImageWidget()

    image.load_catalog(
        Table({"x": [1.0], "y": [2.0]}),
        catalog_label="test",
        catalog_style={"color": "red", "shape": shape, "size": 6},
    )
    marker = image._catalog_trace("test")
    assert type(marker) is go.Scatter
    assert marker.marker.size == 6
    assert marker.marker.symbol == symbol
    np.testing.assert_array_equal(marker.x, [1.0])
    np.testing.assert_array_equal(marker.y, [2.0])

    image.set_catalog_style(
        catalog_label="test", size=7, shape=shape
    )

    marker = image._catalog_trace("test")
    assert type(marker) is go.Scatter
    assert marker.marker.size == 7
    assert marker.marker.symbol == symbol

    # Restyling must update the trace in place, never add a second trace
    # for the same catalog.
    scatters = [t for t in image.viewer.data if isinstance(t, go.Scatter)]
    assert len(scatters) == 1


def test_catalog_marks_use_resolved_label():
    # The analog of the bqplot regression test for #213: set_catalog_style
    # and remove_catalog with no label must target the trace named with
    # the resolved label, not create an orphan trace named "None" or fail
    # to find one.
    image = ImageWidget()
    image.load_catalog(Table({'x': [1.0], 'y': [2.0]}), catalog_label='test')
    scatters = [t.name for t in image.viewer.data
                if isinstance(t, go.Scatter)]
    assert scatters == ['test']

    image.set_catalog_style(size=7)
    scatters = [t.name for t in image.viewer.data
                if isinstance(t, go.Scatter)]
    assert scatters == ['test']
    assert image._catalog_trace('test').marker.size == 7

    image.remove_catalog()
    assert [t for t in image.viewer.data if isinstance(t, go.Scatter)] == []


def test_catalog_generated_label_targets_marks():
    # A catalog loaded without a label gets a generated label; styling or
    # removing it by that generated label must target its trace.
    image = ImageWidget()
    image.load_catalog(Table({'x': [1.0], 'y': [2.0]}))

    label = image.catalog_labels[0]
    scatters = [t.name for t in image.viewer.data
                if isinstance(t, go.Scatter)]
    assert scatters == [label]

    image.set_catalog_style(catalog_label=label, size=7)
    scatters = [t.name for t in image.viewer.data
                if isinstance(t, go.Scatter)]
    assert scatters == [label]
    assert image._catalog_trace(label).marker.size == 7

    image.remove_catalog(catalog_label=label)
    assert [t for t in image.viewer.data if isinstance(t, go.Scatter)] == []


class TestPlotlyWidget(ImageAPITest):
    image_widget_class = ImageWidget

    @pytest.mark.skipif(not HAS_KALEIDO,
                        reason="Saving to raster formats requires the "
                               "optional kaleido package.")
    def test_save(self, tmp_path):
        super().test_save(tmp_path)

    @pytest.mark.skipif(not HAS_KALEIDO,
                        reason="Saving to raster formats requires the "
                               "optional kaleido package.")
    def test_save_overwrite(self, tmp_path):
        super().test_save_overwrite(tmp_path)

    def test_displayed_rgb_tracks_cuts_stretch_and_colormap(self, data):
        # The displayed RGB array must be recomputed from the stored cuts,
        # stretch and colormap whenever any of them changes.
        self.image.load_image(data)

        cuts = apviz.AsymmetricPercentileInterval(5, 90)
        stretch = apviz.LogStretch()
        self.image.set_cuts(cuts)
        self.image.set_stretch(stretch)
        self.image.set_colormap('viridis')

        displayed = np.asarray(self.image._image_trace.z)
        np.testing.assert_array_equal(
            displayed, expected_rgb(data, cuts, stretch, 'viridis'))

    def test_default_colormap_is_greys_r(self, data):
        # With no colormap explicitly set, the display should use Greys_r
        # (low = black, high = white), the usual astronomical convention.
        # Settings are stored per image, so get_colormap can only report
        # it once an image is loaded.
        self.image.load_image(data)
        assert self.image.get_colormap() == 'Greys_r'

        displayed = np.asarray(self.image._image_trace.z)
        np.testing.assert_array_equal(
            displayed,
            expected_rgb(data, self.image.get_cuts(),
                         self.image.get_stretch(), 'Greys_r'))

    def test_load_image_keeps_current_display_settings(self):
        # Loading a new image should display it with the cuts, stretch and
        # colormap that are currently in effect, carried forward from the
        # previously displayed image, and store them for the new image so
        # that the get_* methods agree with the display.
        rng = np.random.default_rng(seed=42)
        arr = rng.integers(1100, 1300, size=(50, 60)).astype(np.uint16)
        arr[25, 30] = 65535

        self.image.load_image(arr, image_label='first')
        cuts = apviz.AsymmetricPercentileInterval(5, 90)
        stretch = apviz.LogStretch()
        self.image.set_cuts(cuts, image_label='first')
        self.image.set_stretch(stretch, image_label='first')
        self.image.set_colormap('viridis', image_label='first')

        self.image.load_image(arr, image_label='second')

        assert self.image.get_cuts(image_label='second') is cuts
        assert self.image.get_stretch(image_label='second') is stretch
        assert self.image.get_colormap(image_label='second') == 'viridis'

        displayed = np.asarray(self.image._image_trace.z)
        np.testing.assert_array_equal(
            displayed, expected_rgb(arr, cuts, stretch, 'viridis'))

    def test_apply_viewport_sets_axis_ranges(self, data):
        # The stored viewport must be pushed into the figure as explicit
        # axis ranges: the fov spans the x axis, the y range follows the
        # figure aspect ratio, and both ranges ascend (origin lower).
        self.image.load_image(data)
        self.image.set_viewport(center=(75, 50), fov=100)

        xrange = self.image.viewer.layout.xaxis.range
        yrange = self.image.viewer.layout.yaxis.range
        assert xrange == (25, 125)
        aspect = (self.image._display_height / self.image._display_width)
        assert yrange == (50 - 50 * aspect, 50 + 50 * aspect)
