// Gmsh project created on Thu May 08 14:36:37 2025
SetFactory("OpenCASCADE");
//+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ 

Box(1) = {0, 0, 0, 10, 20, 4};
//+
Box(2) = {10, 0, 0, 100, 20, 4};
//+
Box(3) = {110, 0, 0, 100, 20, 4};
//+
Box(4) = {210, 0, 0, 10, 20, 4};
//+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+  

Box(5) = {0, 0, 26, 10, 20, 4};
//+
Box(6) = {10, 0, 26, 100, 20, 4};
//+
Box(7) = {110, 0, 26, 100, 20, 4};
//+
Box(8) = {210, 0, 26, 10, 20, 4};
//+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ 

Box(9) = {0, 8.5, 4, 10, 3, 22};
//+
Box(10) = {10, 8.5, 4, 100, 3, 22};
//+
Box(11) = {110, 8.5, 4, 100, 3, 22};
//+
Box(12) = {210, 8.5, 4, 10, 3, 22};
//+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ //+ 

Cylinder(13) = {45, 8.5, 15, 0, 3, 0, 9, 2*Pi};
//+
Cylinder(14) = {90, 8.5, 15, 0, 3, 0, 9, 2*Pi};
//+
Cylinder(15) = {130, 8.5, 15, 0, 3, 0, 9, 2*Pi};
//+
Cylinder(16) = {175, 8.5, 15, 0, 3, 0, 9, 2*Pi};


Cylinder(17) = {-10, -10, -10, 0, 40, 0, 10, 1*Pi};
Rotate {{0, 1, 0}, {0, 0, 0}, -Pi/2} {
  Volume{17};
}

Cylinder(18) = {-10, -10, -210, 0, 40, 0, 10, 1*Pi};
Rotate {{0, 1, 0}, {0, 0, 0}, -Pi/2} {
  Volume{18};
}

Cylinder(19) = {-40, -10, 110, 0, 40, 0, 10, 1*Pi};
Rotate {{0, 1, 0}, {0, 0, 0}, Pi/2} {
  Volume{19};
}

//+
BooleanDifference{ Volume{11}; Delete; }{ Volume{16}; Volume{15}; Delete; }
//+
BooleanDifference{ Volume{10}; Delete; }{ Volume{14}; Volume{13}; Delete; }
//+
BooleanUnion{ Volume{12}; Delete; }{ Volume{11}; Volume{10}; Volume{9}; Delete; }
//+
BooleanUnion{ Volume{8}; Delete; }{ Volume{7}; Volume{6}; Volume{5}; Delete; }
//+
BooleanUnion{ Volume{4}; Delete; }{ Volume{3}; Volume{2}; Volume{1}; Delete; }
//+
Coherence;
//+
//+Physical Curve("restrain", 272) = {222, 212};
//+
Physical Volume("PLA", 274) = {21, 20, 22};
//+
Physical Volume("restrain", 275) = {18, 17};

//+
Physical Volume("load", 276) = {19};
//+
Physical Curve("support_out_of_plane", 277) = {234, 235, 236, 233};
