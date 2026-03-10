SetFactory("OpenCASCADE");

// ===========================
// 1. Elipse principal
//Ellipse(1) = {0, 0, 0, 7, 3, 0, 2*Pi};
//Curve Loop(1) = {1};
//Plane Surface(1) = {1};


//+
Point(2) = {-1, 1, 0, 1.0};
//+
Point(3) = {-0.75, 3, 0, 1.0};
//+
Point(4) = {-0.5, 2.25, 0, 1.0};
//+
Point(5) = {0, 2.25, 0, 1.0};
//+
Point(6) = {0.5, 2.25, 0, 1.0};
//+
Point(7) = {0.75, 3, 0, 1.0};
//+
Point(8) = {1, 1, 0, 1.0};


//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {5, 6};
//+
Line(6) = {6, 7};
//+
Line(7) = {7, 8};


// --- Parámetros ---
x = 3;
y_val = Sqrt(360)/7;

// --- Puntos sobre la elipse para x = ±3 ---
Point(201) = { -x,  y_val, 0, 0.1 };
Point(203) = {  x,  y_val, 0, 0.1 };
u = 1 + 14/Sqrt(409);
xmin = -u;
ymin = 6*Sqrt(10)/7 + (-0.5*u + 1.5) - (3*Sqrt(10)/7)*Sqrt(4 - (u - 1)^2);
Point(900) = { xmin, ymin, 0, 0.1 };
Point(901) = { -xmin, ymin, 0, 0.1 };
//+
Spline(8) = {201, 900, 2};
//+
Spline(9) = {8, 901, 203};
Point(10) = { -7, 0, 0, 0.1 };
Point(11) = { 7, 0, 0, 0.1 };
a = (3*Sqrt(33) - 7)/112;
Point(101) = {
    -3,
    Abs(-3/2) - a * (-3)^2 - 3 + Sqrt(1 - (Abs(Abs(-3) - 2) - 1)^2),
    0, 1.0
};
Point(102) = {
    -2,
    Abs(-2/2) - a * (-2)^2 - 3 + Sqrt(1 - (Abs(Abs(-2) - 2) - 1)^2),
    0, 1.0
};
Point(103) = {
    -1,
    Abs(-1/2) - a * (-1)^2 - 3 + Sqrt(1 - (Abs(Abs(-1) - 2) - 1)^2),
    0, 1.0
};
Point(104) = {
    0,
    Abs(0/2) - a * (0)^2 - 3 + Sqrt(1 - (Abs(Abs(0) - 2) - 1)^2),
    0, 1.0
};
Point(105) = {
    1,
    Abs(1/2) - a * (1)^2 - 3 + Sqrt(1 - (Abs(Abs(1) - 2) - 1)^2),
    0, 1.0
};
Point(106) = {
    2,
    Abs(2/2) - a * (2)^2 - 3 + Sqrt(1 - (Abs(Abs(2) - 2) - 1)^2),
    0, 1.0
};
Point(107) = {
    3,
    Abs(3/2) - a * (3)^2 - 3 + Sqrt(1 - (Abs(Abs(3) - 2) - 1)^2),
    0, 1.0
};
Point(108) = {
    -4,
    Abs(-4/2) - a * (-4)^2 - 3 + Sqrt(1 - (Abs(Abs(-4) - 2) - 1)^2),
    0, 1.0
};
Point(109) = {
    4,
    Abs(4/2) - a * (4)^2 - 3 + Sqrt(1 - (Abs(Abs(4) - 2) - 1)^2),
    0, 1.0
};


//+
Spline(10) = {201, 10, 108};
//+
Spline(11) = {108, 101, 102};
//+
Spline(12) = {102, 103, 104};
//+
Spline(13) = {104, 105, 106};
//+
Spline(14) = {106, 107, 109};
//+
Spline(15) = {109, 11, 203};


//+
Curve Loop(1) = {8, 2, 3, 4, 5, 6, 7, 9, -15, -14, -13, -12, -11, -10};
//+
Curve Loop(2) = {10, 11, 12, 13, 14, 15, -9, -7, -6, -5, -4, -3, -2, -8};


//+
Plane Surface(1) = {2};


//+
Transfinite Curve {10, 15} = 52 Using Progression 1;
//+
Transfinite Curve {8, 9} = 26 Using Progression 1;
//+
Transfinite Curve {2, 7} = 20 Using Progression 1;
//+
Transfinite Curve {3, 6} = 8 Using Progression 1;
//+
Transfinite Curve {4, 5} = 6 Using Progression 1;
//+
Transfinite Curve {11, 14} = 20 Using Progression 1;
//+
Transfinite Curve {12, 13} = 21 Using Progression 1;



//+
Physical Surface("steel", 17) = {1};

//+
Physical Point("support", 18) = {5, 104};
//+
Physical Curve("load_p_x", 19) = {15};

//+
Physical Curve("load_m_x", 20) = {10};
