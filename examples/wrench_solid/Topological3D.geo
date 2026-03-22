// Gmsh project created on Thu May 08 14:36:37 2025
SetFactory("OpenCASCADE");
//+
Rectangle(1) = {-110, 11, 1.5, 10, -22, 0};
//+
Rectangle(2) = {-100, 11, 1.5, 100, -22, 0};
//+
Rectangle(3) = {0, 11, 1.5, 100, -22, 0};
//+
Rectangle(4) = {100, 11, 1.5, 10, -22, 0};
//+


Rectangle(5) = {-110, 15, 10, 10, -4, 0};
//+
Rectangle(6) = {-100, 15, 10, 100, -4, 0};
//+
Rectangle(7) = {0, 15, 10, 100, -4, 0};
//+
Rectangle(8) = {100, 15, 10, 10, -4, 0};
//+


Rectangle(9) = {-110, -15, 10, 10, 4, 0};
//+
Rectangle(10) = {-100, -15, 10, 100, 4, 0};
//+
Rectangle(11) = {0, -15, 10,100, 4, 0};
//+
Rectangle(12) = {100, -15, 10, 10, 4, 0};
//+



Circle(100) = {-65, 0, 1.5, 9, 0, 2*Pi};
//+
Circle(101) = {-20, 0, 1.5, 9, 0, 2*Pi};
//+
Circle(102) = {20, 0, 1.5, 9, 0, 2*Pi};
//+
Circle(103) = {65, 0, 1.5, 9, 0, 2*Pi};
//+
Curve Loop(13) = {100};
//+
Plane Surface(13) = {13};
//+
Curve Loop(14) = {101};
//+
Plane Surface(14) = {14};
//+
Curve Loop(15) = {102};
//+
Plane Surface(15) = {15};
//+
Curve Loop(16) = {103};
//+
Plane Surface(16) = {16};
//+
BooleanDifference{ Surface{2}; Delete; }{ Surface{13}; Surface{14}; Delete; }
//+
BooleanDifference{ Surface{3}; Delete; }{ Surface{15}; Surface{16}; Delete; }
//+
Extrude {0, 0, -20} {
  Surface{9}; Surface{10}; Surface{11}; Surface{12}; Surface{8}; Surface{5}; Surface{6}; Surface{7}; 
}

//+
Extrude {0, 0, -3} {
  Surface{1}; Surface{2}; Surface{3}; Surface{4}; 
}
//+
Coherence;

//+
Physical Curve("load", 131) = {19};
//+
Physical Curve("support", 132) = {74, 55};
//+
Physical Volume("PLA", 133) = {1, 9, 6, 2, 10, 7, 3, 11, 8, 4, 12, 5};
