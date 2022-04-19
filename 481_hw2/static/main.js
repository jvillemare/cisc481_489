/**
 * main.js
 * 481_hw2
 */

function submit4() {
    var urlPuzzle = "";
    for( var row = 1; row < 5; row++ ) {
        for( var col = 1; col < 5; col++ ) {
            var v = document.getElementById(row + '' + col).value;
            if( v === "" )
                v = "x";
            urlPuzzle += v;
        }
        urlPuzzle += ",";
    }
    // trim off the extra comma (,) at the end
    urlPuzzle = urlPuzzle.substring(0, urlPuzzle.length - 1);
    console.log(urlPuzzle);
    // redirect to solve
    location.href = '/solve/' + urlPuzzle;
}


