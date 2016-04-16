intersection() {
    union() {
        cylinder(r1=100, r2=0, h=20);
        for (i = [0:200]) {
            rotate([0, 0, 5*i])
            translate([100+i/6, 0, i/2-20])
            rotate([2, 0, -2])
            cube([120-i/3, 15, 50], center=true);
        }
    }
    intersection() {
        cylinder(r=115, h=300, $fn=60, center=true);
        cylinder(r1=70, r2=150, h=120, $fn=60);
    }
}