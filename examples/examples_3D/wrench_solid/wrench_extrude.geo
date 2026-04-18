// Gmsh project created on Fri Apr 18 15:45:51 2025
SetFactory("OpenCASCADE");

//+Volumen Total== 6863.698695796743 (1mm tamano maximo y minimo)
Circle(1) = {55, 0, 0, 14.5, 0, 2*Pi};
//+
Circle(2) = {61, 0, 0, 9, 0, 2*Pi};
//+
Circle(3) = {27.5, 0, 0, 3.25, 0, 2*Pi};
//+
Circle(4) = {-27.5, 0, 0, 3.25, 0, 2*Pi};
//+
Circle(5) = {-55, 0, 0, 16, 0, 2*Pi};
//+
Circle(6) = {-61, 0, 0, 13, 0, 2*Pi};

R1=23;
R2=29/2;
Lt=R1+R2;
y0=-11/2-R1;
x0=110/2 - Sqrt((R1+R2)^2-(y0)^2);
	

Circle(7) = {x0, y0, 0, 23, 0, 2*Pi};
Circle(8) = {x0, -y0, 0, 23, 0, 2*Pi};


R1=47;
R2=32/2;
Lt=R1+R2;
y1=-11/2-R1;
x1=(110/2 - Sqrt((R1+R2)^2-(y1)^2))*-1;

Circle(9) = {x1, y1, 0, 47, 0, 2*Pi};
Circle(10) = {x1, -y1, 0, 47, 0, 2*Pi};//+

//+ +++++++++++++

Rectangle(1) = {-27.5, -6.5/2, 0, 27.5*2, 6.5, 0};
Rectangle(2) = {-43, -11/2, 0, 90, 11, 0};
Rectangle(3) = {-93, -17/2, 0, 47.5, 17, 0};
Rectangle(4) = {97, -13/2, 0, -47.5, 13, 0};
Rectangle(5) = {-48.5, 11/2, 0, 40, 15, 0};
Rectangle(6) = {-48.5, -11/2-15, 0, 40, 15, 0};
Rectangle(7) = {10.5, 11/2, 0, 40, 10, 0};
Rectangle(8) = {10.5, -11/2-10, 0, 40, 10, 0};


//+ Crear supercies +++++++++++++

//+
Curve Loop(9) = {10};
//+
Plane Surface(9) = {9};
//+
Curve Loop(10) = {8};
//+
Plane Surface(10) = {10};
//+
Curve Loop(11) = {7};
//+
Plane Surface(11) = {11};
//+
Curve Loop(12) = {9};
//+
Plane Surface(12) = {12};
//+
Curve Loop(13) = {6};
//+
Plane Surface(13) = {13};
//+
Curve Loop(14) = {5};
//+
Plane Surface(14) = {14};
//+
Curve Loop(15) = {1};
//+
Plane Surface(15) = {15};
//+
Curve Loop(16) = {2};
//+
Plane Surface(16) = {16};
//+
Curve Loop(17) = {3};
//+
Plane Surface(17) = {17};
//+
Curve Loop(18) = {4};
//+
Plane Surface(18) = {18};

//+ Fragments Mango Derecha Pequeno-Grande +++++++++++++
//+
BooleanFragments{ Surface{15}; Delete; }{ Surface{4}; Delete; }
//+
BooleanFragments{ Surface{16}; Delete; }{ Surface{21}; Delete; }
//+
BooleanFragments{ Surface{22}; Delete; }{ Surface{19}; Delete; }


//+ Fragments Mango Derecha Izquierda-Grande +++++++++++++

//+
BooleanFragments{ Surface{13}; Delete; }{ Surface{3}; Delete; }
//+
BooleanFragments{ Surface{14}; Delete; }{ Surface{31}; Surface{32}; Delete; }


//+ Uniones Mango Derecha +++++++++++++//+

//+
BooleanUnion{ Surface{35}; Delete; }{ Surface{34}; Surface{36}; Surface{37}; Delete; }
//+
//+
BooleanFragments{ Surface{35}; Delete; }{ Surface{30}; Delete; }
//+
Recursive Delete {
  Surface{33}; Surface{42}; Surface{38}; Surface{39}; Surface{41}; 
}


//+ Uniones Mango Izquierda +++++++++++++//+

//+
BooleanUnion{ Surface{20}; Delete; }{ Surface{25}; Surface{28}; Surface{26}; Delete; }
//+
Recursive Delete {
  Surface{24}; Surface{29}; Surface{21}; Surface{23}; Surface{27}; 
}

//+ Uniones Mango +++++++++++++
//+
BooleanUnion{ Surface{1}; Delete; }{ Surface{18}; Surface{17}; Delete; }


//+ Fragments de Circulos Derecha +++++++++++++


//+
BooleanFragments{ Surface{20}; Delete; }{ Surface{7}; Surface{2}; Surface{8}; Delete; }
//+
BooleanFragments{ Surface{11}; Delete; }{ Surface{47}; Delete; }
//+
BooleanFragments{ Surface{10}; Delete; }{ Surface{45}; Delete; }
//+
BooleanUnion{ Surface{41}; Delete; }{ Surface{44}; Surface{43}; Surface{42}; Delete; }
//+
BooleanUnion{ Surface{46}; Delete; }{ Surface{55}; Surface{50}; Delete; }
//+
Recursive Delete {
  Surface{54}; Surface{53}; Surface{52}; Surface{49}; Surface{48}; Surface{47}; 
}
//+
Recursive Delete {
  Surface{51}; Surface{56}; 
}




//+ Fragments de Circulos Izquierda +++++++++++++
//+
BooleanFragments{ Surface{40}; Delete; }{ Surface{46}; Surface{5}; Surface{6}; Delete; }
//+
BooleanFragments{ Surface{9}; Delete; }{ Surface{47}; Delete; }
//+
BooleanFragments{ Surface{12}; Delete; }{ Surface{48}; Delete; }
//+
BooleanUnion{ Surface{42}; Delete; }{ Surface{45}; Surface{44}; Surface{43}; Delete; }

//+
BooleanUnion{ Surface{46}; Delete; }{ Surface{52}; Surface{57}; Delete; }
//+
Recursive Delete {
  Surface{50}; Surface{49}; Surface{54}; Surface{55}; 
}
//+
Recursive Delete {
  Surface{56}; Surface{51}; 
}
//+
Recursive Delete {
  Surface{53}; Surface{58}; 
}
//+

//+
Rotate {{0, 0, 1}, {-55, 0, 0}, 17*Pi/360} {
  Surface{42}; 
}
//+
Rotate {{0, 0, 1}, {55, 0, 0}, 17*Pi/360} {
  Surface{41}; 
}
//+
Coherence;
//+
Physical Surface("Cabeza_5mm_der", 252) = {41};
//+
Physical Surface("Cabeza_5mm_izq", 253) = {42};
//+
Physical Surface("Mango_3mm", 254) = {43};
//+
Physical Surface("Mango_1_6mm", 255) = {1};
//+
Physical Curve("Support", 256) = {250, 247};
//+
Physical Curve("LoadBoundary", 257) = {245};
//+

Extrude {0, 0, -1.5} {
  Surface{43};
}
//+
Extrude {0, 0, 1.5} {
  Surface{43};
}
//+
Extrude {0, 0, 0.8} {
  Surface{1};
}
//+
Extrude {0, 0, -0.8} {
  Surface{1};
}
//+
Extrude {0, 0, 2.5} {
  Surface{42}; Surface{41};
}
//+
Extrude {0, 0, -2.5} {
  Surface{42}; Surface{41};
}

//+
Mesh.ElementOrder = 1;
Physical Volume("solido", 364) = {5, 7, 2, 3, 4, 1, 6, 8};
