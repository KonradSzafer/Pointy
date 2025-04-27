import torch
import plotly.io as pio
import plotly.graph_objs as go


def normalize_color_range(colors: torch.Tensor) -> torch.Tensor:
    """ Normalize color values to be between 0 and 1 """
    return (colors - colors.min()) / (255 - colors.min())


def rgb_to_hex(colors: list[tuple[float]]) -> list[str]:
    """ Convert RGB colors to hex format """
    return ["#%02x%02x%02x" % (r, g, b) for r, g, b in colors]


def get_colors_hex(colors: torch.Tensor) -> list[str]:
    """ Get colors in hex format """
    if torch.all((colors >= 0) & (colors <= 1)):
        # Need to be transposed in order to iterate over the colors
        colors = colors.T
    else:
        # Normalize colors to be between 0 and 1
        colors_normalized = torch.zeros(colors.T.shape)
        colors_normalized[:, 0] = normalize_color_range(colors[0])
        colors_normalized[:, 1] = normalize_color_range(colors[1])
        colors_normalized[:, 2] = normalize_color_range(colors[2])
        colors = colors_normalized
    
    # Convert to hex format
    return rgb_to_hex([(r, g, b) for r, g, b in colors])


def create_axis_config(
    backgroundcolor: str = "white",
    visible: bool = False,
    showbackground: bool = True
) -> dict:
    """ Create axis configuration for plotly """
    return dict(
        gridcolor=backgroundcolor,
        zerolinecolor=backgroundcolor,
        backgroundcolor=backgroundcolor,
        visible=visible,
        showbackground=showbackground
    )


def plot3d(
    pointcloud: torch.Tensor,
    colors: list[str] = None,
    textured: bool = False,
    background_color: str = "white",
    scene_visible: bool = False,
    point_size: int = 3,
    save_path: str = None
) -> None:
 
    if textured:
        if colors is None:
            colors = get_colors_hex(pointcloud[3:].cpu())
        else:
            colors = rgb_to_hex(colors)

        marker = dict(
            color=colors,
            size=point_size,
            opacity=1.0
            if textured
            else None   
        )
    else:
        marker = dict(size=point_size)

    trace = go.Scatter3d(
        x=pointcloud[0].cpu().numpy(),
        y=pointcloud[1].cpu().numpy(),
        z=pointcloud[2].cpu().numpy(),
        mode="markers",
        marker=marker,
    )
    
    # Compute axis ranges to maintain aspect ratio
    x_range = (pointcloud[0].max() - pointcloud[0].min()).item()
    y_range = (pointcloud[1].max() - pointcloud[1].min()).item()
    z_range = (pointcloud[2].max() - pointcloud[2].min()).item()

    scene = dict(
        xaxis=create_axis_config(background_color, scene_visible),
        yaxis=create_axis_config(background_color, scene_visible),
        zaxis=create_axis_config(background_color, scene_visible),
        aspectmode="manual",
        aspectratio=dict(x=x_range, y=y_range, z=z_range)
    )
    layout = go.Layout(
        title="",
        scene=scene,
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        margin=dict(t=0, l=0, b=0, r=0),
        width=800,
        height=800,
    )
    fig = go.Figure(data=[trace], layout=layout)

    if save_path:
        pio.write_image(fig, save_path)
    pio.show(fig)
