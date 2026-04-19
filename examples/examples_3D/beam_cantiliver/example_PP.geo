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
Physical Surface("support", 21) = {7};
//+
Physical Curve("load", 22) = {17};
//+
Physical Volume("solid", 23) = {1};
