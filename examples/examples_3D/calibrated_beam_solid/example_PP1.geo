//+
SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, 5000, 300, 0};
//+
Rectangle(2) = {0, 0, 0, 5000, 300, 0};
//+
Rectangle(3) = {0, 0, 0, 5000, 300, 0};
//+
Extrude {0, 0, 500} {
  Surface{1}; 
}
//+
Physical Curve("support_xy", 21) = {4};
//+
Physical Curve("support_y", 22) = {2};
//+
Physical Surface("load", 23) = {8};
//+
Physical Volume("solid", 24) = {1};
