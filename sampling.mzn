
int: THRESHOLD;

set of int: NFILES;
set of int: NLABELS;

array [NFILES, NLABELS] of 0..1: DATA;
array [NLABELS] of int: LIMITS;

var set of NFILES: files;

constraint forall (j in NLABELS) (
    let { var int: colsum = sum (i in files) (DATA[i,j]) }
    in colsum >= LIMITS[j] - 20 /\ colsum <= THRESHOLD + 500);

solve satisfy;

% solve maximize sum (j in NLABELS, i in files) (DATA[i,j]);

% constraint forall (j in NLABELS) (sum (i in files) (DATA[i,j]) <= THRESHOLD);
% solve maximize card(files);

% solve minimize sum (j in NLABELS) ((sum (i in files) (DATA[i,j]) - THRESHOLD)^2);

output ["files = \(files)\n"];
