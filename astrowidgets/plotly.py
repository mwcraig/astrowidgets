"""
Plotly backend for astrowidgets.

The public image-viewer API is inherited from
`~astro_image_display_api.ImageViewerLogic`, whose methods are templates
that own all state handling and label resolution and then call private
rendering hooks. This module implements only those hooks, pushing the
already-validated state into a `plotly.graph_objects.FigureWidget`.
"""
import warnings
from pathlib import Path

import numpy as np
from astropy.nddata import NDData

import ipywidgets as ipw

from matplotlib import colormaps as mpl_colormaps

import plotly.graph_objects as go

from astro_image_display_api import ImageViewerLogic, docs_from_super_if_missing

__all__ = ['ImageWidget']

# Name of the single trace that displays the image, so that it can be
# found and replaced. Catalogs are drawn as scatter traces named by their
# catalog label; the image trace is a go.Image, so the two namespaces
# cannot collide.
IMAGE_TRACE_NAME = 'astrowidgets-image'

# Map the marker shape names used by the astro-image-display-api catalog
# styles to plotly scatter symbols. The open variants are used so that
# markers outline sources instead of covering them.
SHAPE_TO_SYMBOL = {
    'circle': 'circle-open',
    'square': 'square-open',
    'diamond': 'diamond-open',
    'plus': 'cross-thin-open',
    'crosshair': 'cross-thin-open',
}


# The inheritance order below matters -- VBox needs to come first
@docs_from_super_if_missing
class ImageWidget(ipw.VBox, ImageViewerLogic):
    def __init__(self, *args, display_width=500, display_aspect_ratio=1):
        super().__init__(*args)
        # ImageViewerLogic is a dataclass; we do not run its __init__, so run
        # the post-init hook it would otherwise provide to set up its state.
        ImageViewerLogic.__post_init__(self)

        self._display_width = display_width
        self._display_height = display_aspect_ratio * display_width

        self._figure = go.FigureWidget()
        self._figure.update_layout(
            width=self._display_width,
            height=self._display_height,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
        )
        # The viewport is controlled through the API (set_viewport), which
        # sets explicit axis ranges; hide the axes themselves.
        self._figure.update_xaxes(visible=False, showgrid=False)
        self._figure.update_yaxes(visible=False, showgrid=False)

        # Colormap used for the first displayed image when none has been
        # set: low = black, high = white, the usual astronomical convention.
        self._default_colormap = 'Greys_r'

        self._data = None
        self._wcs = None

        # Provide an Output widget to which prints can be directed for
        # debugging.
        self._print_out = ipw.Output()

        self.children = (self._figure, )

    @property
    def viewer(self):
        """
        The `plotly.graph_objects.FigureWidget` that displays the image.
        """
        return self._figure

    @property
    def print_out(self):
        """
        Return an output widget for display in the notebook which
        captures any printed output produced by the viewer widget.

        Intended primarily for debugging.
        """
        return self._print_out

    # ------------------------------------------------------------------
    # Helpers for finding traces in the figure
    # ------------------------------------------------------------------
    @property
    def _image_trace(self):
        """
        The `plotly.graph_objects.Image` trace displaying the image, or
        None if no image has been displayed yet.
        """
        for trace in self._figure.data:
            if isinstance(trace, go.Image):
                return trace
        return None

    def _catalog_trace(self, catalog_label):
        """
        The `plotly.graph_objects.Scatter` trace drawn for a catalog, or
        None if the catalog has no trace.

        Parameters
        ----------
        catalog_label : str
            Resolved label of the catalog.
        """
        for trace in self._figure.data:
            if isinstance(trace, go.Scatter) and trace.name == catalog_label:
                return trace
        return None

    def _recolorize(self, image_label):
        """
        Recompute the RGB rendering of the displayed image and push it
        into the figure's image trace.

        The stored cuts normalize the data to [0, 1], the stored stretch
        reshapes that interval, and the stored colormap (a matplotlib
        colormap name) maps it to RGB. The single image trace is updated
        in place, or created on the first call; a new trace is never
        appended next to an existing one. The viewport (axis ranges) is
        left untouched; it is owned by the ``_apply_viewport`` hook.

        Parameters
        ----------
        image_label : str
            Resolved label of the image to render.
        """
        if self._data is None:
            return

        data = np.asarray(self._data)

        cuts = self.get_cuts(image_label=image_label)
        normalized = cuts(data)

        stretch = self.get_stretch(image_label=image_label)
        if stretch is not None:
            normalized = stretch(normalized)

        cmap_name = (self.get_colormap(image_label=image_label)
                     or self._default_colormap)
        cmap = mpl_colormaps[cmap_name]
        rgb = (cmap(np.clip(normalized, 0, 1))[..., :3] * 255).astype(np.uint8)

        trace = self._image_trace
        if trace is None:
            # With an explicit ascending y-axis range (set by
            # _apply_viewport) row j of z is drawn at y = j, so row 0 is
            # at the bottom: origin lower, matching the pixel coordinates
            # used everywhere in the API.
            self._figure.add_trace(
                go.Image(z=rgb, name=IMAGE_TRACE_NAME, hoverinfo='none')
            )
        else:
            trace.z = rgb

    # ------------------------------------------------------------------
    # Rendering hooks
    #
    # The public API methods inherited from ImageViewerLogic are templates
    # that own all state handling and label resolution, then call these
    # hooks with already-resolved labels to push the stored state into the
    # plotly figure. The _apply_* hooks are only called for the displayed
    # image.
    # ------------------------------------------------------------------
    def _batch_update(self):
        """
        Batch the front-end updates of a group of state changes.

        Overrides the no-op hook from
        `~astro_image_display_api.ImageViewerLogic`; ``load_image`` wraps
        its state changes and rendering-hook calls in this context.

        Returns
        -------
        context manager
            The figure's own ``batch_update`` context, which collects all
            trace and layout assignments made inside the block and sends
            them to the front end as a single update, so the browser
            redraws once instead of after every assignment.
        """
        return self._figure.batch_update()

    def _render_image(self, image_label):
        """
        Make the image stored under a label the one the widget displays.

        Overrides the no-op rendering hook from
        `~astro_image_display_api.ImageViewerLogic`. ``load_image`` calls it
        after storing the image data and settings; the ``_apply_*`` hooks
        are called afterwards to push the stored settings into the display.

        Parameters
        ----------
        image_label : str
            Resolved label of the image to display.
        """
        info = self._images[image_label]

        if info.colormap is None:
            # load_image carries the displayed image's colormap forward to
            # the new image, and any image that has been displayed has a
            # colormap (this very block guarantees it), so a missing
            # colormap means nothing was carried forward: this image is
            # the first to be displayed. Store the widget's default for it
            # so that get_colormap reports what is displayed.
            info.colormap = self._default_colormap

        data = info.data
        self._data = data.data if isinstance(data, NDData) else data
        self._wcs = info.wcs

        self._recolorize(image_label)

    def _apply_cuts(self, image_label):
        """
        Re-display the image using the cut levels stored for a label.

        Overrides the no-op rendering hook from
        `~astro_image_display_api.ImageViewerLogic`; only called when the
        label is displayed. Changing the cuts only affects the color
        mapping, so the current viewport (zoom/pan) is left untouched.

        Parameters
        ----------
        image_label : str
            Resolved label whose stored cuts to apply.
        """
        self._recolorize(image_label)

    def _apply_stretch(self, image_label):
        """
        Re-display the image using the stretch stored for a label.

        Overrides the no-op rendering hook from
        `~astro_image_display_api.ImageViewerLogic`; only called when the
        label is displayed. Changing the stretch only affects the color
        mapping, so the current viewport (zoom/pan) is left untouched.

        Parameters
        ----------
        image_label : str
            Resolved label whose stored stretch to apply.
        """
        self._recolorize(image_label)

    def _apply_colormap(self, image_label):
        """
        Re-display the image using the colormap stored for a label.

        Overrides the no-op rendering hook from
        `~astro_image_display_api.ImageViewerLogic`; only called when the
        label is displayed. The colormap is applied while building the RGB
        array sent to the figure, so a colormap change means recomputing
        that array.

        Parameters
        ----------
        image_label : str
            Resolved label whose stored colormap to apply.
        """
        self._recolorize(image_label)

    def _apply_viewport(self, image_label):
        """
        Push the viewport stored for a label into the figure's axis ranges.

        Overrides the no-op rendering hook from
        `~astro_image_display_api.ImageViewerLogic`; only called when the
        label is displayed.

        The stored field of view spans the horizontal axis; the vertical
        range is derived from the figure's aspect ratio so that pixels
        stay square. The y range ascends, so that row 0 of the image is
        drawn at the bottom (origin lower).

        Parameters
        ----------
        image_label : str
            Resolved label whose stored viewport (center and field of view)
            to apply.
        """
        # Get the viewport in pixel coordinates; the API layer handles all
        # of the WCS stuff in the event the stored fov is in sky units.
        viewport = self.get_viewport(image_label=image_label,
                                     sky_or_pixel='pixel')
        center = viewport['center']
        fov = viewport['fov']

        half_width = fov / 2
        half_height = fov * (self._display_height / self._display_width) / 2

        with self._figure.batch_update():
            self._figure.update_xaxes(
                range=[center[0] - half_width, center[0] + half_width])
            self._figure.update_yaxes(
                range=[center[1] - half_height, center[1] + half_height])

    def _draw_catalog(self, catalog_label):
        """
        Draw (or redraw) a catalog's markers as a named scatter trace.

        Overrides the no-op rendering hook from
        `~astro_image_display_api.ImageViewerLogic`; called by
        ``load_catalog`` and ``set_catalog_style``.

        Parameters
        ----------
        catalog_label : str
            Resolved label of the catalog to draw. Its markers are drawn
            as a `plotly.graph_objects.Scatter` trace whose name is the
            label; if the catalog already has a trace, that trace is
            updated in place rather than a duplicate being appended.
        """
        catalog = self.get_catalog(catalog_label=catalog_label)
        style = self.get_catalog_style(catalog_label=catalog_label)

        x = np.asarray(catalog['x'])
        y = np.asarray(catalog['y'])

        shape = style.get('shape', 'circle')
        marker = dict(
            color=style.get('color', 'red'),
            size=style.get('size', 5),
            # Pass unknown shapes through so that plotly's own symbol
            # names can be used directly in a catalog style.
            symbol=SHAPE_TO_SYMBOL.get(shape, shape),
        )

        trace = self._catalog_trace(catalog_label)
        if trace is None:
            self._figure.add_trace(
                go.Scatter(x=x, y=y, mode='markers', name=catalog_label,
                           marker=marker, showlegend=False)
            )
        else:
            with self._figure.batch_update():
                trace.x = x
                trace.y = y
                trace.marker = marker

    def _remove_catalog_marks(self, catalog_label):
        """
        Remove a catalog's scatter trace from the figure.

        Overrides the no-op rendering hook from
        `~astro_image_display_api.ImageViewerLogic`; ``remove_catalog``
        calls it once per removed catalog (after expanding ``"*"``).

        Parameters
        ----------
        catalog_label : str
            Resolved label of the catalog whose markers to remove.
        """
        self._figure.data = tuple(
            trace for trace in self._figure.data
            if not (isinstance(trace, go.Scatter)
                    and trace.name == catalog_label)
        )

    # Saving contents of the view
    def save(self, filename, overwrite=False, **kwargs):
        """
        Save the current view to a file.

        Parameters
        ----------
        filename : str or `os.PathLike`
            Name of the file to save to. Saving to ``.html`` produces a
            self-contained interactive page and needs no extra packages;
            any image format supported by plotly (e.g. ``.png``, ``.svg``,
            ``.pdf``) requires the optional ``kaleido`` package.
        overwrite : bool, optional
            Whether to overwrite an existing file.
        """
        p = Path(filename)
        if p.exists() and not overwrite:
            raise FileExistsError(
                f'File {filename} already exists. Use '
                f'overwrite=True to overwrite it.')

        if p.suffix == '.html':
            self._figure.write_html(str(filename))
        else:
            # Requires the optional kaleido package; plotly raises an
            # informative error if it is not installed.
            with warnings.catch_warnings():
                # plotly 6.x internally passes its deprecated engine
                # argument down to its own to_image, tripping its own
                # deprecation warning; there is nothing a caller can do
                # about it, so silence that specific warning.
                warnings.filterwarnings(
                    'ignore',
                    # (?s) so that the pattern can match past the newline
                    # the message starts with.
                    message=r"(?s).*'engine' argument is deprecated",
                    category=DeprecationWarning,
                )
                self._figure.write_image(str(filename))
