function panel() {
    var r1 = 8;
    var r2 = 8;
    return hull(
        CAG.circle({center: [r1, r1], radius: r1}),
        CAG.circle({center: [150-r1, r1], radius: r1}),
        CAG.circle({center: [r2, 135-r2], radius: r2}),
        CAG.circle({center: [130, 240-r2], radius: r2}),
        CAG.circle({center: [150-r2, 240-r2], radius: r2}))
        .subtract(CAG.circle({center: [30, 8], radius: 2}))
        .subtract(CAG.circle({center: [30, 144], radius: 2}))
        .subtract(CAG.circle({center: [120, 8], radius: 2}))
        .subtract(CAG.circle({center: [120, 222], radius: 2}))
    ;
}

function main() {
    return union(
        panel().scale([-1, 1, 1]).translate([0, -135]),
        panel().rotateZ([90]).translate([135, 0])
    );
}
