
int: THRESHOLD;

set of int: NFILES;
set of int: NLABELS;

array [NFILES, NLABELS] of int: data;

var set of NFILES: files;

solve minimize sum (j in NLABELS) ((sum (i in files) (data[i,j]) - THRESHOLD)^2);

output ["files = \(files)\n"];
