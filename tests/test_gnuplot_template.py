from pathlib import Path


def test_vote_model_template_restores_autoscale_before_each_terminal_pass():
    """The documentation panel must not clip the following PDF to [0, 1]."""
    template = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "vote-model-clusters.gnuplot"
    ).read_text(encoding="utf-8")

    render_loop = template.index("do for [render_index=1:2]")
    autoscale_x = template.index("set autoscale x", render_loop)
    autoscale_y = template.index("set autoscale y", render_loop)
    multiplot = template.index("set multiplot", render_loop)

    assert render_loop < autoscale_x < multiplot
    assert render_loop < autoscale_y < multiplot
