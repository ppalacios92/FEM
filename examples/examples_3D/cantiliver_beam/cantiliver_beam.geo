//+
SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, 2000, 300, 0};
//+
Rectangle(2) = {0, 0, 0, 2000, 300, 0};
//+
Rectangle(3) = {0, 0, 0, 2000, 300, 0};
//+
Extrude {0, 0, 300} {
  Surface{1}; 
}

//+
Physical Volume("solid", 24) = {1};
//+
Physical Surface("support", 25) = {7};
//+
Physical Surface("load", 31) = {5};
