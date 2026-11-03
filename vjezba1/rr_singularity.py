import numpy as np

def forward_kinematics(q1, q2, l1=1.0, l2=1.0):
    """Return base, elbow, and endpoint coordinates; angles are in radians."""
    elbow = np.array([l1 * np.cos(q1), l1 * np.sin(q1)])
    endpoint = elbow + l2 * np.array([np.cos(q1 + q2), np.sin(q1 + q2)])
    return np.vstack((np.zeros(2), elbow, endpoint))


def jacobian(q1, q2, l1=1.0, l2=1.0):
    """Position Jacobian: [x_dot, y_dot]^T = J @ [q1_dot, q2_dot]^T."""
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    return np.array([
        [-l1 * s1 - l2 * s12, -l2 * s12],
        [l1 * c1 + l2 * c12, l2 * c12],
    ])


def create_demo(initial_q1=0.0, initial_q2=45.0):
    """Build the figure and return it together with its live widgets."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    l1 = l2 = 1.0

    # The boundary of the unit joint-rate disk. Applying J gives the ellipse.
    theta = np.linspace(0.0, 2.0 * np.pi, 501)
    joint_rate_circle = np.vstack((np.cos(theta), np.sin(theta)))

    fig, (ax_arm, ax_vel) = plt.subplots(1, 2, figsize=(11, 6.6))
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.39, top=0.85, wspace=0.28)
    fig.suptitle("Two-link arm: loss of an instantaneous motion direction", y=0.96)

    ax_arm.set_title("Arm configuration")
    ax_arm.set(xlim=(-2.3, 2.3), ylim=(-2.3, 2.3), xlabel="x [m]", ylabel="y [m]")
    ax_arm.set_aspect("equal", adjustable="box")
    ax_arm.grid(alpha=0.22)
    ax_arm.axhline(0, color="0.65", linewidth=0.8)
    ax_arm.axvline(0, color="0.65", linewidth=0.8)
    arm_line, = ax_arm.plot([], [], "o-", color="0.25", linewidth=5, markersize=8)
    tip, = ax_arm.plot([], [], "o", color="tab:blue", markersize=9)
    ax_arm.text(0.03, 0.04, "l1 = l2 = 1 m", transform=ax_arm.transAxes)

    # sqrt(5) m/s is the maximum singular value for the example; fixed limits
    # make the shrinking width directly comparable across slider settings.
    ax_vel.set_title("Achievable end-effector velocities")
    ax_vel.set(xlim=(-2.6, 2.6), ylim=(-2.6, 2.6),
               xlabel=r"$v_x$ [m/s]", ylabel=r"$v_y$ [m/s]")
    ax_vel.set_aspect("equal", adjustable="box")
    ax_vel.grid(alpha=0.22)
    ax_vel.axhline(0, color="0.65", linewidth=0.8)
    ax_vel.axvline(0, color="0.65", linewidth=0.8)
    velocity_line, = ax_vel.plot([], [], color="tab:blue", linewidth=2)
    velocity_fill = ax_vel.fill([0], [0], color="tab:blue", alpha=0.14)[0]
    principal_axes = [
        ax_vel.plot([], [], "--", color="0.45", linewidth=1)[0]
        for _ in range(2)
    ]
    ax_vel.text(0.03, 0.04, r"$\|\dot q\|_2 \leq 1$ rad/s", transform=ax_vel.transAxes)

    matrix_text = fig.text(0.08, 0.30, "", va="top", family="monospace", fontsize=14)
    metrics_text = fig.text(0.53, 0.30, "", va="top", family="monospace", fontsize=14)
    status_text = fig.text(0.08, 0.18, "", va="center", fontsize=14)

    q1_slider = Slider(fig.add_axes((0.23, 0.095, 0.43, 0.03)),
                       "Shoulder q1 [deg]", -180, 180,
                       valinit=initial_q1, valfmt="%0.1f")
    q2_slider = Slider(fig.add_axes((0.23, 0.045, 0.43, 0.03)),
                       "Elbow q2 [deg]", -90, 90,
                       valinit=initial_q2, valfmt="%0.1f")
    button_ax = fig.add_axes((0.79, 0.045, 0.17, 0.08))
    # Hover redraws refresh the shared background, keeping the button color in
    # sync with subsequent slider updates.
    straight_button = Button(button_ax, "Straight (0 deg)", useblit=False)

    canvas = fig.canvas
    background = None
    dynamic_artists = [
        arm_line, tip, velocity_fill, velocity_line, *principal_axes,
        matrix_text, metrics_text,
    ]
    for slider in (q1_slider, q2_slider):
        # Redraw the moving fill, handle, and value, keeping the track and label
        # in the background. ax.lines includes the handle and zero marker.
        dynamic_artists.extend([slider.poly, *slider.ax.lines, slider.valtext])
    if canvas.supports_blit:
        for artist in dynamic_artists:
            artist.set_animated(True)

    # The update callback draws both widgets and plots together, avoiding an
    # extra full-figure redraw from each slider's default set_val behavior.
    q1_slider.drawon = q2_slider.drawon = False

    def draw_dynamic_artists():
        for artist in dynamic_artists:
            fig.draw_artist(artist)

    def on_draw(event):
        nonlocal background
        if canvas.supports_blit:
            # A full draw (including after a resize) refreshes the static axes,
            # ticks, and labels. Cache them for subsequent slider updates.
            background = canvas.copy_from_bbox(fig.bbox)
            draw_dynamic_artists()

    canvas.mpl_connect("draw_event", on_draw)

    def update(_=None):
        nonlocal background
        q1, q2 = np.deg2rad([q1_slider.val, q2_slider.val])
        points = forward_kinematics(q1, q2, l1, l2)
        J = jacobian(q1, q2, l1, l2)
        U, singular_values, _ = np.linalg.svd(J)
        tolerance = max(J.shape) * np.finfo(float).eps * singular_values[0]
        rank = np.count_nonzero(singular_values > tolerance)
        determinant = l1 * l2 * np.sin(q2)

        arm_line.set_data(points[:, 0], points[:, 1])
        tip.set_data([points[-1, 0]], [points[-1, 1]])

        velocities = J @ joint_rate_circle
        velocity_line.set_data(velocities[0], velocities[1])
        velocity_fill.set_xy(velocities.T)
        for i, line in enumerate(principal_axes):
            endpoint = singular_values[i] * U[:, i]
            line.set_data([-endpoint[0], endpoint[0]], [-endpoint[1], endpoint[1]])

        matrix_text.set_text(
            "Jacobian J [m/rad]\n"
            f"[{J[0, 0]:+7.3f}  {J[0, 1]:+7.3f}]\n"
            f"[{J[1, 0]:+7.3f}  {J[1, 1]:+7.3f}]"
        )
        condition = (
            f"{singular_values[0] / singular_values[-1]:.2f}"
            if rank == 2 else "infinity"
        )
        metrics_text.set_text(
            f"rank(J) = {rank}    det(J) = {determinant:+.4f}\n"
            f"sigma_min = {singular_values[-1]:.4f}    sigma_max = {singular_values[0]:.4f}\n"
            f"condition number = {condition}"
        )
        if rank < 2:
            status = "Singular: motion along the arm is unavailable; the velocity ellipse is a line."
        else:
            status = "Rank 2: both velocity directions are available; the ellipse narrows near q2 = 0 deg."
        if status_text.get_text() != status:
            status_text.set_text(status)
            background = None
        if canvas.supports_blit and background is not None:
            canvas.restore_region(background)
            draw_dynamic_artists()
            canvas.blit(fig.bbox)
        else:
            canvas.draw_idle()

    q1_slider.on_changed(update)
    q2_slider.on_changed(update)
    straight_button.on_clicked(lambda event: q2_slider.set_val(0))
    update()
    return fig, q1_slider, q2_slider, straight_button


def main():
    import matplotlib.pyplot as plt

    # Keep widget references alive for the duration of the GUI event loop.
    fig, q1_slider, q2_slider, straight_button = create_demo()
    plt.show()


if __name__ == "__main__":
    main()