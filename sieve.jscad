function main() {
    var length = 300;
    var width = 230;
    var hole = 20;
    var pitch = 22;

    var sieve = CAG.roundedRectangle({
        radius: [length/2, width/2],
        roundradius: 5,
    });
        
    for (var x = -6; x <= 6; x++) {
        for (var y = -4; y <= 4-abs(x)%2; y++) {
            sieve = sieve.subtract(
                CAG.circle({
                    radius: hole/2,
                    resolution: 6,
                })
                .translate([x*pitch,
                            (y+(abs(x)%2)/2)*pitch*1.1]));
        }
    }

    return sieve;
}
