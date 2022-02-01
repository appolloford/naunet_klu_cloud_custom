#include <math.h>
/* */
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_sparse.h>  // access to sparse SUNMatrix
/* */
/*  */
#include "naunet_ode.h"
/*  */
#include "naunet_constants.h"
#include "naunet_macros.h"
#include "naunet_physics.h"

#define IJth(A, i, j)            SM_ELEMENT_D(A, i, j)
#define NVEC_CUDA_CONTENT(x)     ((N_VectorContent_Cuda)(x->content))
#define NVEC_CUDA_STREAM(x)      (NVEC_CUDA_CONTENT(x)->stream_exec_policy->stream())
#define NVEC_CUDA_BLOCKSIZE(x)   (NVEC_CUDA_CONTENT(x)->stream_exec_policy->blockSize())
#define NVEC_CUDA_GRIDSIZE(x, n) (NVEC_CUDA_CONTENT(x)->stream_exec_policy->gridSize(n))

/* */

int Jac(realtype t, N_Vector u, N_Vector fu, SUNMatrix jmatrix, void *user_data,
        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    /* */
    realtype *y            = N_VGetArrayPointer(u);
    sunindextype *rowptrs  = SUNSparseMatrix_IndexPointers(jmatrix);
    sunindextype *colvals  = SUNSparseMatrix_IndexValues(jmatrix);
    realtype *data         = SUNSparseMatrix_Data(jmatrix);
    NaunetData *u_data     = (NaunetData *)user_data;

    // clang-format off
        
    // clang-format on

    realtype k[NREACTIONS] = {0.0};
    EvalRates(k, y, u_data);

#if NHEATPROCS
    realtype kh[NHEATPROCS] = {0.0};
    EvalHeatingRates(kh, y, u_data);
#endif 

#if NCOOLPROCS
    realtype kc[NCOOLPROCS] = {0.0};
    EvalCoolingRates(kc, y, u_data);
#endif

    // clang-format off
    // number of non-zero elements in each row
    rowptrs[0] = 0;
    rowptrs[1] = 4;
    rowptrs[2] = 7;
    rowptrs[3] = 10;
    rowptrs[4] = 12;
    rowptrs[5] = 14;
    rowptrs[6] = 16;
    rowptrs[7] = 19;
    rowptrs[8] = 21;
    rowptrs[9] = 23;
    rowptrs[10] = 26;
    rowptrs[11] = 28;
    rowptrs[12] = 31;
    rowptrs[13] = 33;
    rowptrs[14] = 39;
    rowptrs[15] = 44;
    rowptrs[16] = 56;
    rowptrs[17] = 59;
    rowptrs[18] = 62;
    rowptrs[19] = 63;
    rowptrs[20] = 65;
    rowptrs[21] = 72;
    rowptrs[22] = 80;
    rowptrs[23] = 88;
    rowptrs[24] = 96;
    rowptrs[25] = 103;
    rowptrs[26] = 108;
    rowptrs[27] = 111;
    rowptrs[28] = 117;
    rowptrs[29] = 123;
    rowptrs[30] = 126;
    rowptrs[31] = 128;
    rowptrs[32] = 131;
    rowptrs[33] = 135;
    rowptrs[34] = 138;
    rowptrs[35] = 142;
    rowptrs[36] = 145;
    rowptrs[37] = 155;
    rowptrs[38] = 158;
    rowptrs[39] = 160;
    rowptrs[40] = 164;
    rowptrs[41] = 167;
    rowptrs[42] = 170;
    rowptrs[43] = 174;
    rowptrs[44] = 177;
    rowptrs[45] = 180;
    rowptrs[46] = 183;
    rowptrs[47] = 195;
    rowptrs[48] = 196;
    rowptrs[49] = 200;
    rowptrs[50] = 204;
    rowptrs[51] = 208;
    rowptrs[52] = 301;
    rowptrs[53] = 362;
    rowptrs[54] = 427;
    rowptrs[55] = 462;
    rowptrs[56] = 512;
    rowptrs[57] = 565;
    rowptrs[58] = 602;
    rowptrs[59] = 654;
    rowptrs[60] = 673;
    rowptrs[61] = 696;
    rowptrs[62] = 709;
    rowptrs[63] = 720;
    rowptrs[64] = 728;
    rowptrs[65] = 743;
    rowptrs[66] = 766;
    rowptrs[67] = 774;
    rowptrs[68] = 786;
    rowptrs[69] = 800;
    rowptrs[70] = 807;
    rowptrs[71] = 817;
    rowptrs[72] = 826;
    rowptrs[73] = 834;
    rowptrs[74] = 839;
    rowptrs[75] = 846;
    rowptrs[76] = 921;
    rowptrs[77] = 982;
    rowptrs[78] = 1044;
    rowptrs[79] = 1099;
    rowptrs[80] = 1108;
    rowptrs[81] = 1166;
    rowptrs[82] = 1222;
    rowptrs[83] = 1234;
    rowptrs[84] = 1250;
    rowptrs[85] = 1252;
    rowptrs[86] = 1265;
    rowptrs[87] = 1287;
    rowptrs[88] = 1302;
    rowptrs[89] = 1367;
    rowptrs[90] = 1394;
    rowptrs[91] = 1429;
    rowptrs[92] = 1440;
    rowptrs[93] = 1449;
    rowptrs[94] = 1513;
    rowptrs[95] = 1546;
    rowptrs[96] = 1647;
    rowptrs[97] = 1690;
    rowptrs[98] = 1738;
    rowptrs[99] = 1773;
    rowptrs[100] = 1800;
    rowptrs[101] = 1926;
    rowptrs[102] = 2071;
    rowptrs[103] = 2143;
    rowptrs[104] = 2256;
    rowptrs[105] = 2290;
    rowptrs[106] = 2299;
    rowptrs[107] = 2307;
    rowptrs[108] = 2375;
    rowptrs[109] = 2425;
    rowptrs[110] = 2437;
    rowptrs[111] = 2452;
    rowptrs[112] = 2458;
    rowptrs[113] = 2548;
    rowptrs[114] = 2594;
    rowptrs[115] = 2646;
    rowptrs[116] = 2683;
    rowptrs[117] = 2687;
    rowptrs[118] = 2699;
    rowptrs[119] = 2705;
    rowptrs[120] = 2764;
    rowptrs[121] = 2808;
    rowptrs[122] = 2822;
    rowptrs[123] = 2880;
    rowptrs[124] = 2907;
    rowptrs[125] = 2915;
    rowptrs[126] = 2929;
    rowptrs[127] = 2942;
    rowptrs[128] = 2950;
    rowptrs[129] = 3023;
    rowptrs[130] = 3065;
    rowptrs[131] = 3103;
    rowptrs[132] = 3169;
    rowptrs[133] = 3255;
    rowptrs[134] = 3279;
    rowptrs[135] = 3287;
    rowptrs[136] = 3301;
    rowptrs[137] = 3329;
    rowptrs[138] = 3398;
    rowptrs[139] = 3464;
    rowptrs[140] = 3473;
    rowptrs[141] = 3513;
    rowptrs[142] = 3519;
    rowptrs[143] = 3542;
    rowptrs[144] = 3577;
    rowptrs[145] = 3584;
    rowptrs[146] = 3592;
    rowptrs[147] = 3602;
    rowptrs[148] = 3650;
    rowptrs[149] = 3693;
    rowptrs[150] = 3701;
    rowptrs[151] = 3714;
    rowptrs[152] = 3725;
    rowptrs[153] = 3732;
    rowptrs[154] = 3741;
    rowptrs[155] = 3767;
    rowptrs[156] = 3792;
    rowptrs[157] = 3874;
    rowptrs[158] = 3905;
    rowptrs[159] = 3950;
    rowptrs[160] = 3982;
    rowptrs[161] = 4017;
    rowptrs[162] = 4032;
    rowptrs[163] = 4098;
    rowptrs[164] = 4133;
    rowptrs[165] = 4186;
    rowptrs[166] = 4228;
    rowptrs[167] = 4292;
    rowptrs[168] = 4344;
    rowptrs[169] = 4389;
    rowptrs[170] = 4459;
    rowptrs[171] = 4508;
    rowptrs[172] = 4526;
    rowptrs[173] = 4546;
    rowptrs[174] = 4560;
    rowptrs[175] = 4664;
    rowptrs[176] = 4709;
    rowptrs[177] = 4781;
    rowptrs[178] = 4819;
    rowptrs[179] = 4834;
    rowptrs[180] = 4863;
    rowptrs[181] = 4880;
    rowptrs[182] = 4914;
    rowptrs[183] = 4930;
    rowptrs[184] = 5016;
    rowptrs[185] = 5062;
    rowptrs[186] = 5139;
    rowptrs[187] = 5196;
    rowptrs[188] = 5214;
    rowptrs[189] = 5226;
    rowptrs[190] = 5272;
    rowptrs[191] = 5316;
    rowptrs[192] = 5331;
    rowptrs[193] = 5348;
    rowptrs[194] = 5359;
    rowptrs[195] = 5370;
    rowptrs[196] = 5376;
    rowptrs[197] = 5381;
    rowptrs[198] = 5398;
    rowptrs[199] = 5421;
    rowptrs[200] = 5434;
    rowptrs[201] = 5453;
    rowptrs[202] = 5466;
    rowptrs[203] = 5482;
    rowptrs[204] = 5496;
    rowptrs[205] = 5505;
    rowptrs[206] = 5514;
    rowptrs[207] = 5541;
    rowptrs[208] = 5563;
    rowptrs[209] = 5585;
    rowptrs[210] = 5598;
    rowptrs[211] = 5609;
    rowptrs[212] = 5642;
    rowptrs[213] = 5671;
    rowptrs[214] = 5691;
    rowptrs[215] = 5701;
    
    // the column index of non-zero elements
    colvals[0] = 0;
    colvals[1] = 53;
    colvals[2] = 54;
    colvals[3] = 68;
    colvals[4] = 1;
    colvals[5] = 55;
    colvals[6] = 56;
    colvals[7] = 2;
    colvals[8] = 57;
    colvals[9] = 58;
    colvals[10] = 3;
    colvals[11] = 59;
    colvals[12] = 4;
    colvals[13] = 60;
    colvals[14] = 5;
    colvals[15] = 61;
    colvals[16] = 6;
    colvals[17] = 62;
    colvals[18] = 63;
    colvals[19] = 7;
    colvals[20] = 69;
    colvals[21] = 8;
    colvals[22] = 72;
    colvals[23] = 9;
    colvals[24] = 73;
    colvals[25] = 74;
    colvals[26] = 10;
    colvals[27] = 79;
    colvals[28] = 11;
    colvals[29] = 70;
    colvals[30] = 82;
    colvals[31] = 12;
    colvals[32] = 83;
    colvals[33] = 13;
    colvals[34] = 64;
    colvals[35] = 65;
    colvals[36] = 67;
    colvals[37] = 84;
    colvals[38] = 85;
    colvals[39] = 14;
    colvals[40] = 86;
    colvals[41] = 87;
    colvals[42] = 95;
    colvals[43] = 120;
    colvals[44] = 15;
    colvals[45] = 51;
    colvals[46] = 52;
    colvals[47] = 75;
    colvals[48] = 76;
    colvals[49] = 77;
    colvals[50] = 78;
    colvals[51] = 80;
    colvals[52] = 81;
    colvals[53] = 88;
    colvals[54] = 89;
    colvals[55] = 90;
    colvals[56] = 16;
    colvals[57] = 95;
    colvals[58] = 96;
    colvals[59] = 17;
    colvals[60] = 97;
    colvals[61] = 133;
    colvals[62] = 18;
    colvals[63] = 19;
    colvals[64] = 106;
    colvals[65] = 20;
    colvals[66] = 95;
    colvals[67] = 107;
    colvals[68] = 108;
    colvals[69] = 131;
    colvals[70] = 132;
    colvals[71] = 145;
    colvals[72] = 21;
    colvals[73] = 98;
    colvals[74] = 99;
    colvals[75] = 109;
    colvals[76] = 110;
    colvals[77] = 121;
    colvals[78] = 135;
    colvals[79] = 136;
    colvals[80] = 22;
    colvals[81] = 112;
    colvals[82] = 113;
    colvals[83] = 122;
    colvals[84] = 174;
    colvals[85] = 175;
    colvals[86] = 183;
    colvals[87] = 184;
    colvals[88] = 23;
    colvals[89] = 114;
    colvals[90] = 115;
    colvals[91] = 123;
    colvals[92] = 147;
    colvals[93] = 148;
    colvals[94] = 185;
    colvals[95] = 186;
    colvals[96] = 24;
    colvals[97] = 116;
    colvals[98] = 117;
    colvals[99] = 149;
    colvals[100] = 150;
    colvals[101] = 187;
    colvals[102] = 188;
    colvals[103] = 25;
    colvals[104] = 118;
    colvals[105] = 206;
    colvals[106] = 207;
    colvals[107] = 208;
    colvals[108] = 26;
    colvals[109] = 71;
    colvals[110] = 125;
    colvals[111] = 27;
    colvals[112] = 91;
    colvals[113] = 92;
    colvals[114] = 105;
    colvals[115] = 126;
    colvals[116] = 127;
    colvals[117] = 28;
    colvals[118] = 93;
    colvals[119] = 94;
    colvals[120] = 128;
    colvals[121] = 129;
    colvals[122] = 130;
    colvals[123] = 29;
    colvals[124] = 124;
    colvals[125] = 134;
    colvals[126] = 30;
    colvals[127] = 140;
    colvals[128] = 31;
    colvals[129] = 141;
    colvals[130] = 180;
    colvals[131] = 32;
    colvals[132] = 111;
    colvals[133] = 142;
    colvals[134] = 143;
    colvals[135] = 33;
    colvals[136] = 154;
    colvals[137] = 155;
    colvals[138] = 34;
    colvals[139] = 158;
    colvals[140] = 159;
    colvals[141] = 160;
    colvals[142] = 35;
    colvals[143] = 66;
    colvals[144] = 161;
    colvals[145] = 36;
    colvals[146] = 156;
    colvals[147] = 157;
    colvals[148] = 162;
    colvals[149] = 163;
    colvals[150] = 164;
    colvals[151] = 165;
    colvals[152] = 166;
    colvals[153] = 167;
    colvals[154] = 168;
    colvals[155] = 37;
    colvals[156] = 169;
    colvals[157] = 170;
    colvals[158] = 38;
    colvals[159] = 171;
    colvals[160] = 39;
    colvals[161] = 144;
    colvals[162] = 172;
    colvals[163] = 173;
    colvals[164] = 40;
    colvals[165] = 176;
    colvals[166] = 177;
    colvals[167] = 41;
    colvals[168] = 178;
    colvals[169] = 179;
    colvals[170] = 42;
    colvals[171] = 146;
    colvals[172] = 181;
    colvals[173] = 182;
    colvals[174] = 43;
    colvals[175] = 191;
    colvals[176] = 192;
    colvals[177] = 44;
    colvals[178] = 193;
    colvals[179] = 194;
    colvals[180] = 45;
    colvals[181] = 195;
    colvals[182] = 196;
    colvals[183] = 46;
    colvals[184] = 189;
    colvals[185] = 190;
    colvals[186] = 197;
    colvals[187] = 198;
    colvals[188] = 199;
    colvals[189] = 200;
    colvals[190] = 201;
    colvals[191] = 202;
    colvals[192] = 203;
    colvals[193] = 204;
    colvals[194] = 205;
    colvals[195] = 47;
    colvals[196] = 48;
    colvals[197] = 151;
    colvals[198] = 209;
    colvals[199] = 210;
    colvals[200] = 49;
    colvals[201] = 152;
    colvals[202] = 211;
    colvals[203] = 212;
    colvals[204] = 50;
    colvals[205] = 153;
    colvals[206] = 213;
    colvals[207] = 214;
    colvals[208] = 51;
    colvals[209] = 52;
    colvals[210] = 53;
    colvals[211] = 54;
    colvals[212] = 55;
    colvals[213] = 56;
    colvals[214] = 59;
    colvals[215] = 61;
    colvals[216] = 64;
    colvals[217] = 65;
    colvals[218] = 68;
    colvals[219] = 69;
    colvals[220] = 74;
    colvals[221] = 75;
    colvals[222] = 76;
    colvals[223] = 77;
    colvals[224] = 78;
    colvals[225] = 80;
    colvals[226] = 81;
    colvals[227] = 86;
    colvals[228] = 90;
    colvals[229] = 93;
    colvals[230] = 94;
    colvals[231] = 95;
    colvals[232] = 96;
    colvals[233] = 97;
    colvals[234] = 98;
    colvals[235] = 99;
    colvals[236] = 100;
    colvals[237] = 101;
    colvals[238] = 103;
    colvals[239] = 104;
    colvals[240] = 106;
    colvals[241] = 107;
    colvals[242] = 112;
    colvals[243] = 113;
    colvals[244] = 114;
    colvals[245] = 115;
    colvals[246] = 119;
    colvals[247] = 122;
    colvals[248] = 128;
    colvals[249] = 129;
    colvals[250] = 131;
    colvals[251] = 132;
    colvals[252] = 133;
    colvals[253] = 138;
    colvals[254] = 140;
    colvals[255] = 141;
    colvals[256] = 143;
    colvals[257] = 147;
    colvals[258] = 148;
    colvals[259] = 154;
    colvals[260] = 156;
    colvals[261] = 157;
    colvals[262] = 158;
    colvals[263] = 159;
    colvals[264] = 160;
    colvals[265] = 161;
    colvals[266] = 162;
    colvals[267] = 163;
    colvals[268] = 164;
    colvals[269] = 166;
    colvals[270] = 167;
    colvals[271] = 169;
    colvals[272] = 172;
    colvals[273] = 174;
    colvals[274] = 175;
    colvals[275] = 176;
    colvals[276] = 177;
    colvals[277] = 179;
    colvals[278] = 180;
    colvals[279] = 181;
    colvals[280] = 182;
    colvals[281] = 183;
    colvals[282] = 184;
    colvals[283] = 185;
    colvals[284] = 186;
    colvals[285] = 187;
    colvals[286] = 189;
    colvals[287] = 191;
    colvals[288] = 192;
    colvals[289] = 193;
    colvals[290] = 194;
    colvals[291] = 195;
    colvals[292] = 196;
    colvals[293] = 197;
    colvals[294] = 198;
    colvals[295] = 199;
    colvals[296] = 201;
    colvals[297] = 207;
    colvals[298] = 209;
    colvals[299] = 211;
    colvals[300] = 213;
    colvals[301] = 51;
    colvals[302] = 52;
    colvals[303] = 53;
    colvals[304] = 54;
    colvals[305] = 55;
    colvals[306] = 62;
    colvals[307] = 64;
    colvals[308] = 75;
    colvals[309] = 76;
    colvals[310] = 77;
    colvals[311] = 78;
    colvals[312] = 80;
    colvals[313] = 82;
    colvals[314] = 86;
    colvals[315] = 88;
    colvals[316] = 93;
    colvals[317] = 94;
    colvals[318] = 95;
    colvals[319] = 96;
    colvals[320] = 97;
    colvals[321] = 98;
    colvals[322] = 100;
    colvals[323] = 101;
    colvals[324] = 103;
    colvals[325] = 107;
    colvals[326] = 109;
    colvals[327] = 112;
    colvals[328] = 114;
    colvals[329] = 125;
    colvals[330] = 128;
    colvals[331] = 131;
    colvals[332] = 138;
    colvals[333] = 140;
    colvals[334] = 147;
    colvals[335] = 154;
    colvals[336] = 156;
    colvals[337] = 159;
    colvals[338] = 161;
    colvals[339] = 162;
    colvals[340] = 164;
    colvals[341] = 166;
    colvals[342] = 169;
    colvals[343] = 172;
    colvals[344] = 174;
    colvals[345] = 176;
    colvals[346] = 177;
    colvals[347] = 180;
    colvals[348] = 181;
    colvals[349] = 183;
    colvals[350] = 185;
    colvals[351] = 189;
    colvals[352] = 191;
    colvals[353] = 193;
    colvals[354] = 195;
    colvals[355] = 197;
    colvals[356] = 199;
    colvals[357] = 201;
    colvals[358] = 206;
    colvals[359] = 209;
    colvals[360] = 211;
    colvals[361] = 213;
    colvals[362] = 0;
    colvals[363] = 51;
    colvals[364] = 52;
    colvals[365] = 53;
    colvals[366] = 54;
    colvals[367] = 55;
    colvals[368] = 56;
    colvals[369] = 57;
    colvals[370] = 58;
    colvals[371] = 64;
    colvals[372] = 65;
    colvals[373] = 68;
    colvals[374] = 71;
    colvals[375] = 72;
    colvals[376] = 74;
    colvals[377] = 75;
    colvals[378] = 76;
    colvals[379] = 77;
    colvals[380] = 90;
    colvals[381] = 93;
    colvals[382] = 94;
    colvals[383] = 95;
    colvals[384] = 96;
    colvals[385] = 98;
    colvals[386] = 100;
    colvals[387] = 101;
    colvals[388] = 102;
    colvals[389] = 104;
    colvals[390] = 107;
    colvals[391] = 108;
    colvals[392] = 113;
    colvals[393] = 119;
    colvals[394] = 122;
    colvals[395] = 128;
    colvals[396] = 129;
    colvals[397] = 131;
    colvals[398] = 132;
    colvals[399] = 138;
    colvals[400] = 140;
    colvals[401] = 143;
    colvals[402] = 156;
    colvals[403] = 157;
    colvals[404] = 159;
    colvals[405] = 160;
    colvals[406] = 163;
    colvals[407] = 164;
    colvals[408] = 165;
    colvals[409] = 166;
    colvals[410] = 167;
    colvals[411] = 169;
    colvals[412] = 174;
    colvals[413] = 175;
    colvals[414] = 176;
    colvals[415] = 177;
    colvals[416] = 179;
    colvals[417] = 183;
    colvals[418] = 184;
    colvals[419] = 185;
    colvals[420] = 186;
    colvals[421] = 191;
    colvals[422] = 193;
    colvals[423] = 194;
    colvals[424] = 195;
    colvals[425] = 196;
    colvals[426] = 207;
    colvals[427] = 51;
    colvals[428] = 52;
    colvals[429] = 53;
    colvals[430] = 54;
    colvals[431] = 55;
    colvals[432] = 56;
    colvals[433] = 57;
    colvals[434] = 71;
    colvals[435] = 75;
    colvals[436] = 76;
    colvals[437] = 77;
    colvals[438] = 88;
    colvals[439] = 94;
    colvals[440] = 96;
    colvals[441] = 100;
    colvals[442] = 102;
    colvals[443] = 103;
    colvals[444] = 104;
    colvals[445] = 112;
    colvals[446] = 113;
    colvals[447] = 131;
    colvals[448] = 138;
    colvals[449] = 156;
    colvals[450] = 157;
    colvals[451] = 159;
    colvals[452] = 162;
    colvals[453] = 164;
    colvals[454] = 169;
    colvals[455] = 174;
    colvals[456] = 175;
    colvals[457] = 176;
    colvals[458] = 177;
    colvals[459] = 183;
    colvals[460] = 184;
    colvals[461] = 185;
    colvals[462] = 1;
    colvals[463] = 51;
    colvals[464] = 52;
    colvals[465] = 55;
    colvals[466] = 56;
    colvals[467] = 57;
    colvals[468] = 58;
    colvals[469] = 72;
    colvals[470] = 76;
    colvals[471] = 77;
    colvals[472] = 83;
    colvals[473] = 90;
    colvals[474] = 93;
    colvals[475] = 94;
    colvals[476] = 96;
    colvals[477] = 100;
    colvals[478] = 101;
    colvals[479] = 102;
    colvals[480] = 103;
    colvals[481] = 104;
    colvals[482] = 108;
    colvals[483] = 112;
    colvals[484] = 113;
    colvals[485] = 114;
    colvals[486] = 119;
    colvals[487] = 125;
    colvals[488] = 128;
    colvals[489] = 129;
    colvals[490] = 132;
    colvals[491] = 138;
    colvals[492] = 140;
    colvals[493] = 143;
    colvals[494] = 156;
    colvals[495] = 157;
    colvals[496] = 159;
    colvals[497] = 160;
    colvals[498] = 161;
    colvals[499] = 163;
    colvals[500] = 164;
    colvals[501] = 165;
    colvals[502] = 166;
    colvals[503] = 169;
    colvals[504] = 174;
    colvals[505] = 175;
    colvals[506] = 176;
    colvals[507] = 179;
    colvals[508] = 183;
    colvals[509] = 184;
    colvals[510] = 185;
    colvals[511] = 190;
    colvals[512] = 51;
    colvals[513] = 52;
    colvals[514] = 53;
    colvals[515] = 54;
    colvals[516] = 55;
    colvals[517] = 56;
    colvals[518] = 57;
    colvals[519] = 59;
    colvals[520] = 60;
    colvals[521] = 72;
    colvals[522] = 75;
    colvals[523] = 76;
    colvals[524] = 77;
    colvals[525] = 78;
    colvals[526] = 80;
    colvals[527] = 81;
    colvals[528] = 88;
    colvals[529] = 90;
    colvals[530] = 94;
    colvals[531] = 96;
    colvals[532] = 100;
    colvals[533] = 102;
    colvals[534] = 103;
    colvals[535] = 104;
    colvals[536] = 107;
    colvals[537] = 108;
    colvals[538] = 112;
    colvals[539] = 113;
    colvals[540] = 119;
    colvals[541] = 122;
    colvals[542] = 125;
    colvals[543] = 128;
    colvals[544] = 129;
    colvals[545] = 131;
    colvals[546] = 132;
    colvals[547] = 138;
    colvals[548] = 140;
    colvals[549] = 143;
    colvals[550] = 156;
    colvals[551] = 157;
    colvals[552] = 159;
    colvals[553] = 160;
    colvals[554] = 162;
    colvals[555] = 163;
    colvals[556] = 164;
    colvals[557] = 165;
    colvals[558] = 166;
    colvals[559] = 169;
    colvals[560] = 174;
    colvals[561] = 175;
    colvals[562] = 179;
    colvals[563] = 184;
    colvals[564] = 185;
    colvals[565] = 2;
    colvals[566] = 51;
    colvals[567] = 52;
    colvals[568] = 53;
    colvals[569] = 55;
    colvals[570] = 57;
    colvals[571] = 58;
    colvals[572] = 59;
    colvals[573] = 60;
    colvals[574] = 66;
    colvals[575] = 75;
    colvals[576] = 77;
    colvals[577] = 80;
    colvals[578] = 82;
    colvals[579] = 89;
    colvals[580] = 93;
    colvals[581] = 101;
    colvals[582] = 102;
    colvals[583] = 103;
    colvals[584] = 104;
    colvals[585] = 107;
    colvals[586] = 113;
    colvals[587] = 114;
    colvals[588] = 129;
    colvals[589] = 131;
    colvals[590] = 138;
    colvals[591] = 154;
    colvals[592] = 156;
    colvals[593] = 166;
    colvals[594] = 169;
    colvals[595] = 174;
    colvals[596] = 175;
    colvals[597] = 176;
    colvals[598] = 177;
    colvals[599] = 183;
    colvals[600] = 189;
    colvals[601] = 212;
    colvals[602] = 52;
    colvals[603] = 53;
    colvals[604] = 54;
    colvals[605] = 55;
    colvals[606] = 56;
    colvals[607] = 57;
    colvals[608] = 58;
    colvals[609] = 59;
    colvals[610] = 60;
    colvals[611] = 66;
    colvals[612] = 75;
    colvals[613] = 76;
    colvals[614] = 80;
    colvals[615] = 81;
    colvals[616] = 82;
    colvals[617] = 83;
    colvals[618] = 88;
    colvals[619] = 89;
    colvals[620] = 90;
    colvals[621] = 100;
    colvals[622] = 102;
    colvals[623] = 103;
    colvals[624] = 104;
    colvals[625] = 107;
    colvals[626] = 108;
    colvals[627] = 112;
    colvals[628] = 113;
    colvals[629] = 114;
    colvals[630] = 119;
    colvals[631] = 128;
    colvals[632] = 129;
    colvals[633] = 131;
    colvals[634] = 132;
    colvals[635] = 138;
    colvals[636] = 140;
    colvals[637] = 143;
    colvals[638] = 154;
    colvals[639] = 156;
    colvals[640] = 160;
    colvals[641] = 163;
    colvals[642] = 164;
    colvals[643] = 165;
    colvals[644] = 166;
    colvals[645] = 167;
    colvals[646] = 169;
    colvals[647] = 174;
    colvals[648] = 175;
    colvals[649] = 177;
    colvals[650] = 179;
    colvals[651] = 184;
    colvals[652] = 189;
    colvals[653] = 203;
    colvals[654] = 3;
    colvals[655] = 51;
    colvals[656] = 52;
    colvals[657] = 58;
    colvals[658] = 59;
    colvals[659] = 60;
    colvals[660] = 61;
    colvals[661] = 62;
    colvals[662] = 77;
    colvals[663] = 80;
    colvals[664] = 93;
    colvals[665] = 101;
    colvals[666] = 102;
    colvals[667] = 138;
    colvals[668] = 156;
    colvals[669] = 174;
    colvals[670] = 176;
    colvals[671] = 183;
    colvals[672] = 203;
    colvals[673] = 4;
    colvals[674] = 58;
    colvals[675] = 60;
    colvals[676] = 61;
    colvals[677] = 63;
    colvals[678] = 75;
    colvals[679] = 80;
    colvals[680] = 81;
    colvals[681] = 83;
    colvals[682] = 88;
    colvals[683] = 93;
    colvals[684] = 100;
    colvals[685] = 102;
    colvals[686] = 104;
    colvals[687] = 122;
    colvals[688] = 138;
    colvals[689] = 156;
    colvals[690] = 174;
    colvals[691] = 175;
    colvals[692] = 183;
    colvals[693] = 186;
    colvals[694] = 203;
    colvals[695] = 212;
    colvals[696] = 5;
    colvals[697] = 51;
    colvals[698] = 58;
    colvals[699] = 61;
    colvals[700] = 62;
    colvals[701] = 63;
    colvals[702] = 80;
    colvals[703] = 100;
    colvals[704] = 156;
    colvals[705] = 174;
    colvals[706] = 183;
    colvals[707] = 190;
    colvals[708] = 203;
    colvals[709] = 6;
    colvals[710] = 52;
    colvals[711] = 62;
    colvals[712] = 63;
    colvals[713] = 100;
    colvals[714] = 119;
    colvals[715] = 122;
    colvals[716] = 132;
    colvals[717] = 136;
    colvals[718] = 166;
    colvals[719] = 190;
    colvals[720] = 60;
    colvals[721] = 62;
    colvals[722] = 63;
    colvals[723] = 100;
    colvals[724] = 122;
    colvals[725] = 132;
    colvals[726] = 136;
    colvals[727] = 166;
    colvals[728] = 51;
    colvals[729] = 55;
    colvals[730] = 64;
    colvals[731] = 66;
    colvals[732] = 67;
    colvals[733] = 71;
    colvals[734] = 74;
    colvals[735] = 100;
    colvals[736] = 102;
    colvals[737] = 106;
    colvals[738] = 119;
    colvals[739] = 138;
    colvals[740] = 156;
    colvals[741] = 161;
    colvals[742] = 174;
    colvals[743] = 52;
    colvals[744] = 53;
    colvals[745] = 54;
    colvals[746] = 56;
    colvals[747] = 58;
    colvals[748] = 64;
    colvals[749] = 65;
    colvals[750] = 76;
    colvals[751] = 93;
    colvals[752] = 100;
    colvals[753] = 102;
    colvals[754] = 112;
    colvals[755] = 114;
    colvals[756] = 125;
    colvals[757] = 128;
    colvals[758] = 138;
    colvals[759] = 140;
    colvals[760] = 156;
    colvals[761] = 157;
    colvals[762] = 161;
    colvals[763] = 162;
    colvals[764] = 163;
    colvals[765] = 166;
    colvals[766] = 57;
    colvals[767] = 66;
    colvals[768] = 94;
    colvals[769] = 100;
    colvals[770] = 101;
    colvals[771] = 128;
    colvals[772] = 157;
    colvals[773] = 161;
    colvals[774] = 54;
    colvals[775] = 58;
    colvals[776] = 64;
    colvals[777] = 65;
    colvals[778] = 67;
    colvals[779] = 76;
    colvals[780] = 100;
    colvals[781] = 112;
    colvals[782] = 119;
    colvals[783] = 128;
    colvals[784] = 156;
    colvals[785] = 164;
    colvals[786] = 51;
    colvals[787] = 52;
    colvals[788] = 53;
    colvals[789] = 54;
    colvals[790] = 55;
    colvals[791] = 56;
    colvals[792] = 68;
    colvals[793] = 75;
    colvals[794] = 76;
    colvals[795] = 82;
    colvals[796] = 100;
    colvals[797] = 125;
    colvals[798] = 128;
    colvals[799] = 138;
    colvals[800] = 7;
    colvals[801] = 51;
    colvals[802] = 57;
    colvals[803] = 59;
    colvals[804] = 69;
    colvals[805] = 75;
    colvals[806] = 156;
    colvals[807] = 58;
    colvals[808] = 60;
    colvals[809] = 70;
    colvals[810] = 81;
    colvals[811] = 82;
    colvals[812] = 83;
    colvals[813] = 88;
    colvals[814] = 100;
    colvals[815] = 122;
    colvals[816] = 132;
    colvals[817] = 53;
    colvals[818] = 71;
    colvals[819] = 73;
    colvals[820] = 74;
    colvals[821] = 100;
    colvals[822] = 128;
    colvals[823] = 138;
    colvals[824] = 156;
    colvals[825] = 174;
    colvals[826] = 8;
    colvals[827] = 51;
    colvals[828] = 53;
    colvals[829] = 57;
    colvals[830] = 69;
    colvals[831] = 72;
    colvals[832] = 138;
    colvals[833] = 156;
    colvals[834] = 9;
    colvals[835] = 72;
    colvals[836] = 73;
    colvals[837] = 156;
    colvals[838] = 174;
    colvals[839] = 52;
    colvals[840] = 68;
    colvals[841] = 74;
    colvals[842] = 100;
    colvals[843] = 112;
    colvals[844] = 125;
    colvals[845] = 128;
    colvals[846] = 51;
    colvals[847] = 52;
    colvals[848] = 53;
    colvals[849] = 54;
    colvals[850] = 55;
    colvals[851] = 56;
    colvals[852] = 57;
    colvals[853] = 58;
    colvals[854] = 60;
    colvals[855] = 75;
    colvals[856] = 76;
    colvals[857] = 77;
    colvals[858] = 78;
    colvals[859] = 80;
    colvals[860] = 81;
    colvals[861] = 86;
    colvals[862] = 88;
    colvals[863] = 90;
    colvals[864] = 93;
    colvals[865] = 94;
    colvals[866] = 96;
    colvals[867] = 97;
    colvals[868] = 100;
    colvals[869] = 101;
    colvals[870] = 102;
    colvals[871] = 103;
    colvals[872] = 104;
    colvals[873] = 107;
    colvals[874] = 108;
    colvals[875] = 113;
    colvals[876] = 114;
    colvals[877] = 119;
    colvals[878] = 120;
    colvals[879] = 122;
    colvals[880] = 125;
    colvals[881] = 128;
    colvals[882] = 129;
    colvals[883] = 130;
    colvals[884] = 131;
    colvals[885] = 132;
    colvals[886] = 136;
    colvals[887] = 138;
    colvals[888] = 142;
    colvals[889] = 143;
    colvals[890] = 147;
    colvals[891] = 148;
    colvals[892] = 154;
    colvals[893] = 156;
    colvals[894] = 157;
    colvals[895] = 158;
    colvals[896] = 159;
    colvals[897] = 160;
    colvals[898] = 162;
    colvals[899] = 163;
    colvals[900] = 164;
    colvals[901] = 165;
    colvals[902] = 166;
    colvals[903] = 167;
    colvals[904] = 169;
    colvals[905] = 174;
    colvals[906] = 175;
    colvals[907] = 176;
    colvals[908] = 177;
    colvals[909] = 178;
    colvals[910] = 179;
    colvals[911] = 181;
    colvals[912] = 183;
    colvals[913] = 184;
    colvals[914] = 185;
    colvals[915] = 186;
    colvals[916] = 189;
    colvals[917] = 190;
    colvals[918] = 198;
    colvals[919] = 207;
    colvals[920] = 211;
    colvals[921] = 51;
    colvals[922] = 52;
    colvals[923] = 53;
    colvals[924] = 54;
    colvals[925] = 55;
    colvals[926] = 56;
    colvals[927] = 57;
    colvals[928] = 58;
    colvals[929] = 75;
    colvals[930] = 76;
    colvals[931] = 77;
    colvals[932] = 78;
    colvals[933] = 80;
    colvals[934] = 81;
    colvals[935] = 86;
    colvals[936] = 88;
    colvals[937] = 90;
    colvals[938] = 93;
    colvals[939] = 94;
    colvals[940] = 96;
    colvals[941] = 97;
    colvals[942] = 100;
    colvals[943] = 101;
    colvals[944] = 102;
    colvals[945] = 103;
    colvals[946] = 104;
    colvals[947] = 107;
    colvals[948] = 108;
    colvals[949] = 112;
    colvals[950] = 113;
    colvals[951] = 114;
    colvals[952] = 119;
    colvals[953] = 128;
    colvals[954] = 129;
    colvals[955] = 131;
    colvals[956] = 132;
    colvals[957] = 133;
    colvals[958] = 138;
    colvals[959] = 140;
    colvals[960] = 143;
    colvals[961] = 154;
    colvals[962] = 156;
    colvals[963] = 157;
    colvals[964] = 159;
    colvals[965] = 160;
    colvals[966] = 162;
    colvals[967] = 163;
    colvals[968] = 164;
    colvals[969] = 165;
    colvals[970] = 166;
    colvals[971] = 169;
    colvals[972] = 174;
    colvals[973] = 175;
    colvals[974] = 176;
    colvals[975] = 177;
    colvals[976] = 179;
    colvals[977] = 181;
    colvals[978] = 183;
    colvals[979] = 184;
    colvals[980] = 185;
    colvals[981] = 189;
    colvals[982] = 51;
    colvals[983] = 52;
    colvals[984] = 54;
    colvals[985] = 56;
    colvals[986] = 57;
    colvals[987] = 60;
    colvals[988] = 75;
    colvals[989] = 76;
    colvals[990] = 77;
    colvals[991] = 78;
    colvals[992] = 79;
    colvals[993] = 80;
    colvals[994] = 81;
    colvals[995] = 86;
    colvals[996] = 87;
    colvals[997] = 88;
    colvals[998] = 89;
    colvals[999] = 90;
    colvals[1000] = 93;
    colvals[1001] = 94;
    colvals[1002] = 96;
    colvals[1003] = 100;
    colvals[1004] = 101;
    colvals[1005] = 102;
    colvals[1006] = 103;
    colvals[1007] = 104;
    colvals[1008] = 107;
    colvals[1009] = 108;
    colvals[1010] = 109;
    colvals[1011] = 113;
    colvals[1012] = 119;
    colvals[1013] = 120;
    colvals[1014] = 122;
    colvals[1015] = 129;
    colvals[1016] = 130;
    colvals[1017] = 131;
    colvals[1018] = 132;
    colvals[1019] = 138;
    colvals[1020] = 142;
    colvals[1021] = 143;
    colvals[1022] = 156;
    colvals[1023] = 157;
    colvals[1024] = 158;
    colvals[1025] = 159;
    colvals[1026] = 160;
    colvals[1027] = 163;
    colvals[1028] = 165;
    colvals[1029] = 166;
    colvals[1030] = 167;
    colvals[1031] = 169;
    colvals[1032] = 171;
    colvals[1033] = 174;
    colvals[1034] = 175;
    colvals[1035] = 176;
    colvals[1036] = 177;
    colvals[1037] = 178;
    colvals[1038] = 179;
    colvals[1039] = 183;
    colvals[1040] = 184;
    colvals[1041] = 185;
    colvals[1042] = 186;
    colvals[1043] = 207;
    colvals[1044] = 51;
    colvals[1045] = 52;
    colvals[1046] = 54;
    colvals[1047] = 56;
    colvals[1048] = 60;
    colvals[1049] = 75;
    colvals[1050] = 76;
    colvals[1051] = 77;
    colvals[1052] = 78;
    colvals[1053] = 79;
    colvals[1054] = 81;
    colvals[1055] = 88;
    colvals[1056] = 89;
    colvals[1057] = 90;
    colvals[1058] = 94;
    colvals[1059] = 96;
    colvals[1060] = 97;
    colvals[1061] = 100;
    colvals[1062] = 101;
    colvals[1063] = 102;
    colvals[1064] = 103;
    colvals[1065] = 104;
    colvals[1066] = 107;
    colvals[1067] = 108;
    colvals[1068] = 109;
    colvals[1069] = 112;
    colvals[1070] = 113;
    colvals[1071] = 114;
    colvals[1072] = 119;
    colvals[1073] = 120;
    colvals[1074] = 122;
    colvals[1075] = 129;
    colvals[1076] = 130;
    colvals[1077] = 131;
    colvals[1078] = 132;
    colvals[1079] = 138;
    colvals[1080] = 143;
    colvals[1081] = 148;
    colvals[1082] = 156;
    colvals[1083] = 157;
    colvals[1084] = 159;
    colvals[1085] = 160;
    colvals[1086] = 163;
    colvals[1087] = 165;
    colvals[1088] = 166;
    colvals[1089] = 169;
    colvals[1090] = 174;
    colvals[1091] = 175;
    colvals[1092] = 176;
    colvals[1093] = 177;
    colvals[1094] = 179;
    colvals[1095] = 181;
    colvals[1096] = 184;
    colvals[1097] = 185;
    colvals[1098] = 198;
    colvals[1099] = 10;
    colvals[1100] = 57;
    colvals[1101] = 59;
    colvals[1102] = 60;
    colvals[1103] = 79;
    colvals[1104] = 101;
    colvals[1105] = 138;
    colvals[1106] = 174;
    colvals[1107] = 183;
    colvals[1108] = 51;
    colvals[1109] = 52;
    colvals[1110] = 54;
    colvals[1111] = 56;
    colvals[1112] = 57;
    colvals[1113] = 59;
    colvals[1114] = 60;
    colvals[1115] = 61;
    colvals[1116] = 75;
    colvals[1117] = 77;
    colvals[1118] = 78;
    colvals[1119] = 79;
    colvals[1120] = 80;
    colvals[1121] = 81;
    colvals[1122] = 83;
    colvals[1123] = 85;
    colvals[1124] = 86;
    colvals[1125] = 87;
    colvals[1126] = 88;
    colvals[1127] = 89;
    colvals[1128] = 90;
    colvals[1129] = 93;
    colvals[1130] = 95;
    colvals[1131] = 96;
    colvals[1132] = 97;
    colvals[1133] = 99;
    colvals[1134] = 100;
    colvals[1135] = 101;
    colvals[1136] = 102;
    colvals[1137] = 103;
    colvals[1138] = 107;
    colvals[1139] = 108;
    colvals[1140] = 112;
    colvals[1141] = 113;
    colvals[1142] = 114;
    colvals[1143] = 119;
    colvals[1144] = 129;
    colvals[1145] = 131;
    colvals[1146] = 134;
    colvals[1147] = 138;
    colvals[1148] = 142;
    colvals[1149] = 154;
    colvals[1150] = 156;
    colvals[1151] = 157;
    colvals[1152] = 162;
    colvals[1153] = 164;
    colvals[1154] = 166;
    colvals[1155] = 167;
    colvals[1156] = 169;
    colvals[1157] = 171;
    colvals[1158] = 174;
    colvals[1159] = 176;
    colvals[1160] = 178;
    colvals[1161] = 181;
    colvals[1162] = 183;
    colvals[1163] = 185;
    colvals[1164] = 186;
    colvals[1165] = 190;
    colvals[1166] = 51;
    colvals[1167] = 52;
    colvals[1168] = 56;
    colvals[1169] = 60;
    colvals[1170] = 62;
    colvals[1171] = 75;
    colvals[1172] = 76;
    colvals[1173] = 77;
    colvals[1174] = 78;
    colvals[1175] = 80;
    colvals[1176] = 81;
    colvals[1177] = 83;
    colvals[1178] = 86;
    colvals[1179] = 88;
    colvals[1180] = 89;
    colvals[1181] = 90;
    colvals[1182] = 100;
    colvals[1183] = 101;
    colvals[1184] = 102;
    colvals[1185] = 103;
    colvals[1186] = 104;
    colvals[1187] = 107;
    colvals[1188] = 108;
    colvals[1189] = 112;
    colvals[1190] = 113;
    colvals[1191] = 114;
    colvals[1192] = 119;
    colvals[1193] = 122;
    colvals[1194] = 128;
    colvals[1195] = 129;
    colvals[1196] = 130;
    colvals[1197] = 131;
    colvals[1198] = 132;
    colvals[1199] = 138;
    colvals[1200] = 143;
    colvals[1201] = 147;
    colvals[1202] = 154;
    colvals[1203] = 157;
    colvals[1204] = 159;
    colvals[1205] = 160;
    colvals[1206] = 162;
    colvals[1207] = 163;
    colvals[1208] = 165;
    colvals[1209] = 166;
    colvals[1210] = 167;
    colvals[1211] = 169;
    colvals[1212] = 174;
    colvals[1213] = 175;
    colvals[1214] = 176;
    colvals[1215] = 179;
    colvals[1216] = 181;
    colvals[1217] = 183;
    colvals[1218] = 184;
    colvals[1219] = 185;
    colvals[1220] = 203;
    colvals[1221] = 211;
    colvals[1222] = 11;
    colvals[1223] = 51;
    colvals[1224] = 52;
    colvals[1225] = 60;
    colvals[1226] = 61;
    colvals[1227] = 70;
    colvals[1228] = 75;
    colvals[1229] = 82;
    colvals[1230] = 100;
    colvals[1231] = 122;
    colvals[1232] = 132;
    colvals[1233] = 138;
    colvals[1234] = 12;
    colvals[1235] = 58;
    colvals[1236] = 80;
    colvals[1237] = 81;
    colvals[1238] = 83;
    colvals[1239] = 85;
    colvals[1240] = 93;
    colvals[1241] = 100;
    colvals[1242] = 102;
    colvals[1243] = 119;
    colvals[1244] = 122;
    colvals[1245] = 130;
    colvals[1246] = 132;
    colvals[1247] = 133;
    colvals[1248] = 138;
    colvals[1249] = 160;
    colvals[1250] = 13;
    colvals[1251] = 84;
    colvals[1252] = 58;
    colvals[1253] = 81;
    colvals[1254] = 83;
    colvals[1255] = 85;
    colvals[1256] = 87;
    colvals[1257] = 100;
    colvals[1258] = 119;
    colvals[1259] = 122;
    colvals[1260] = 128;
    colvals[1261] = 130;
    colvals[1262] = 132;
    colvals[1263] = 133;
    colvals[1264] = 160;
    colvals[1265] = 14;
    colvals[1266] = 52;
    colvals[1267] = 76;
    colvals[1268] = 81;
    colvals[1269] = 86;
    colvals[1270] = 87;
    colvals[1271] = 89;
    colvals[1272] = 100;
    colvals[1273] = 102;
    colvals[1274] = 108;
    colvals[1275] = 119;
    colvals[1276] = 120;
    colvals[1277] = 122;
    colvals[1278] = 124;
    colvals[1279] = 132;
    colvals[1280] = 138;
    colvals[1281] = 157;
    colvals[1282] = 166;
    colvals[1283] = 175;
    colvals[1284] = 177;
    colvals[1285] = 188;
    colvals[1286] = 190;
    colvals[1287] = 76;
    colvals[1288] = 81;
    colvals[1289] = 86;
    colvals[1290] = 87;
    colvals[1291] = 89;
    colvals[1292] = 100;
    colvals[1293] = 107;
    colvals[1294] = 108;
    colvals[1295] = 112;
    colvals[1296] = 119;
    colvals[1297] = 120;
    colvals[1298] = 122;
    colvals[1299] = 128;
    colvals[1300] = 132;
    colvals[1301] = 166;
    colvals[1302] = 15;
    colvals[1303] = 51;
    colvals[1304] = 52;
    colvals[1305] = 53;
    colvals[1306] = 54;
    colvals[1307] = 55;
    colvals[1308] = 56;
    colvals[1309] = 57;
    colvals[1310] = 58;
    colvals[1311] = 59;
    colvals[1312] = 62;
    colvals[1313] = 75;
    colvals[1314] = 76;
    colvals[1315] = 77;
    colvals[1316] = 80;
    colvals[1317] = 81;
    colvals[1318] = 86;
    colvals[1319] = 88;
    colvals[1320] = 89;
    colvals[1321] = 90;
    colvals[1322] = 93;
    colvals[1323] = 95;
    colvals[1324] = 96;
    colvals[1325] = 97;
    colvals[1326] = 99;
    colvals[1327] = 100;
    colvals[1328] = 101;
    colvals[1329] = 102;
    colvals[1330] = 103;
    colvals[1331] = 104;
    colvals[1332] = 107;
    colvals[1333] = 108;
    colvals[1334] = 112;
    colvals[1335] = 113;
    colvals[1336] = 114;
    colvals[1337] = 119;
    colvals[1338] = 126;
    colvals[1339] = 128;
    colvals[1340] = 129;
    colvals[1341] = 131;
    colvals[1342] = 133;
    colvals[1343] = 134;
    colvals[1344] = 138;
    colvals[1345] = 140;
    colvals[1346] = 142;
    colvals[1347] = 143;
    colvals[1348] = 148;
    colvals[1349] = 154;
    colvals[1350] = 157;
    colvals[1351] = 159;
    colvals[1352] = 160;
    colvals[1353] = 162;
    colvals[1354] = 164;
    colvals[1355] = 166;
    colvals[1356] = 167;
    colvals[1357] = 174;
    colvals[1358] = 175;
    colvals[1359] = 176;
    colvals[1360] = 178;
    colvals[1361] = 181;
    colvals[1362] = 183;
    colvals[1363] = 184;
    colvals[1364] = 185;
    colvals[1365] = 186;
    colvals[1366] = 203;
    colvals[1367] = 57;
    colvals[1368] = 80;
    colvals[1369] = 81;
    colvals[1370] = 86;
    colvals[1371] = 88;
    colvals[1372] = 89;
    colvals[1373] = 90;
    colvals[1374] = 95;
    colvals[1375] = 96;
    colvals[1376] = 97;
    colvals[1377] = 100;
    colvals[1378] = 101;
    colvals[1379] = 102;
    colvals[1380] = 103;
    colvals[1381] = 104;
    colvals[1382] = 107;
    colvals[1383] = 112;
    colvals[1384] = 114;
    colvals[1385] = 119;
    colvals[1386] = 131;
    colvals[1387] = 138;
    colvals[1388] = 157;
    colvals[1389] = 166;
    colvals[1390] = 174;
    colvals[1391] = 175;
    colvals[1392] = 176;
    colvals[1393] = 181;
    colvals[1394] = 51;
    colvals[1395] = 53;
    colvals[1396] = 55;
    colvals[1397] = 75;
    colvals[1398] = 77;
    colvals[1399] = 81;
    colvals[1400] = 88;
    colvals[1401] = 89;
    colvals[1402] = 90;
    colvals[1403] = 95;
    colvals[1404] = 97;
    colvals[1405] = 100;
    colvals[1406] = 101;
    colvals[1407] = 103;
    colvals[1408] = 104;
    colvals[1409] = 107;
    colvals[1410] = 112;
    colvals[1411] = 114;
    colvals[1412] = 119;
    colvals[1413] = 126;
    colvals[1414] = 128;
    colvals[1415] = 131;
    colvals[1416] = 133;
    colvals[1417] = 140;
    colvals[1418] = 143;
    colvals[1419] = 154;
    colvals[1420] = 160;
    colvals[1421] = 162;
    colvals[1422] = 164;
    colvals[1423] = 166;
    colvals[1424] = 174;
    colvals[1425] = 183;
    colvals[1426] = 184;
    colvals[1427] = 185;
    colvals[1428] = 203;
    colvals[1429] = 91;
    colvals[1430] = 92;
    colvals[1431] = 100;
    colvals[1432] = 101;
    colvals[1433] = 102;
    colvals[1434] = 103;
    colvals[1435] = 105;
    colvals[1436] = 119;
    colvals[1437] = 126;
    colvals[1438] = 127;
    colvals[1439] = 176;
    colvals[1440] = 91;
    colvals[1441] = 92;
    colvals[1442] = 100;
    colvals[1443] = 101;
    colvals[1444] = 102;
    colvals[1445] = 103;
    colvals[1446] = 126;
    colvals[1447] = 138;
    colvals[1448] = 176;
    colvals[1449] = 51;
    colvals[1450] = 52;
    colvals[1451] = 53;
    colvals[1452] = 54;
    colvals[1453] = 55;
    colvals[1454] = 56;
    colvals[1455] = 57;
    colvals[1456] = 58;
    colvals[1457] = 60;
    colvals[1458] = 64;
    colvals[1459] = 65;
    colvals[1460] = 66;
    colvals[1461] = 71;
    colvals[1462] = 73;
    colvals[1463] = 75;
    colvals[1464] = 76;
    colvals[1465] = 77;
    colvals[1466] = 80;
    colvals[1467] = 83;
    colvals[1468] = 88;
    colvals[1469] = 93;
    colvals[1470] = 94;
    colvals[1471] = 95;
    colvals[1472] = 97;
    colvals[1473] = 98;
    colvals[1474] = 100;
    colvals[1475] = 101;
    colvals[1476] = 103;
    colvals[1477] = 104;
    colvals[1478] = 107;
    colvals[1479] = 112;
    colvals[1480] = 119;
    colvals[1481] = 125;
    colvals[1482] = 128;
    colvals[1483] = 129;
    colvals[1484] = 130;
    colvals[1485] = 131;
    colvals[1486] = 138;
    colvals[1487] = 140;
    colvals[1488] = 142;
    colvals[1489] = 143;
    colvals[1490] = 156;
    colvals[1491] = 157;
    colvals[1492] = 158;
    colvals[1493] = 159;
    colvals[1494] = 161;
    colvals[1495] = 162;
    colvals[1496] = 163;
    colvals[1497] = 164;
    colvals[1498] = 166;
    colvals[1499] = 169;
    colvals[1500] = 171;
    colvals[1501] = 172;
    colvals[1502] = 174;
    colvals[1503] = 175;
    colvals[1504] = 176;
    colvals[1505] = 179;
    colvals[1506] = 180;
    colvals[1507] = 183;
    colvals[1508] = 184;
    colvals[1509] = 185;
    colvals[1510] = 191;
    colvals[1511] = 192;
    colvals[1512] = 203;
    colvals[1513] = 51;
    colvals[1514] = 52;
    colvals[1515] = 53;
    colvals[1516] = 55;
    colvals[1517] = 75;
    colvals[1518] = 76;
    colvals[1519] = 77;
    colvals[1520] = 83;
    colvals[1521] = 93;
    colvals[1522] = 94;
    colvals[1523] = 95;
    colvals[1524] = 100;
    colvals[1525] = 101;
    colvals[1526] = 103;
    colvals[1527] = 104;
    colvals[1528] = 107;
    colvals[1529] = 112;
    colvals[1530] = 128;
    colvals[1531] = 131;
    colvals[1532] = 138;
    colvals[1533] = 140;
    colvals[1534] = 156;
    colvals[1535] = 157;
    colvals[1536] = 159;
    colvals[1537] = 161;
    colvals[1538] = 162;
    colvals[1539] = 164;
    colvals[1540] = 169;
    colvals[1541] = 174;
    colvals[1542] = 176;
    colvals[1543] = 180;
    colvals[1544] = 183;
    colvals[1545] = 185;
    colvals[1546] = 16;
    colvals[1547] = 51;
    colvals[1548] = 52;
    colvals[1549] = 53;
    colvals[1550] = 54;
    colvals[1551] = 55;
    colvals[1552] = 56;
    colvals[1553] = 57;
    colvals[1554] = 62;
    colvals[1555] = 64;
    colvals[1556] = 71;
    colvals[1557] = 73;
    colvals[1558] = 75;
    colvals[1559] = 76;
    colvals[1560] = 77;
    colvals[1561] = 78;
    colvals[1562] = 79;
    colvals[1563] = 80;
    colvals[1564] = 81;
    colvals[1565] = 82;
    colvals[1566] = 83;
    colvals[1567] = 86;
    colvals[1568] = 88;
    colvals[1569] = 89;
    colvals[1570] = 90;
    colvals[1571] = 93;
    colvals[1572] = 94;
    colvals[1573] = 95;
    colvals[1574] = 96;
    colvals[1575] = 97;
    colvals[1576] = 98;
    colvals[1577] = 100;
    colvals[1578] = 101;
    colvals[1579] = 102;
    colvals[1580] = 104;
    colvals[1581] = 105;
    colvals[1582] = 107;
    colvals[1583] = 108;
    colvals[1584] = 109;
    colvals[1585] = 112;
    colvals[1586] = 113;
    colvals[1587] = 114;
    colvals[1588] = 119;
    colvals[1589] = 120;
    colvals[1590] = 128;
    colvals[1591] = 129;
    colvals[1592] = 131;
    colvals[1593] = 132;
    colvals[1594] = 133;
    colvals[1595] = 134;
    colvals[1596] = 135;
    colvals[1597] = 138;
    colvals[1598] = 140;
    colvals[1599] = 141;
    colvals[1600] = 142;
    colvals[1601] = 143;
    colvals[1602] = 145;
    colvals[1603] = 147;
    colvals[1604] = 149;
    colvals[1605] = 156;
    colvals[1606] = 157;
    colvals[1607] = 159;
    colvals[1608] = 160;
    colvals[1609] = 162;
    colvals[1610] = 163;
    colvals[1611] = 164;
    colvals[1612] = 166;
    colvals[1613] = 167;
    colvals[1614] = 169;
    colvals[1615] = 171;
    colvals[1616] = 172;
    colvals[1617] = 174;
    colvals[1618] = 175;
    colvals[1619] = 176;
    colvals[1620] = 177;
    colvals[1621] = 178;
    colvals[1622] = 179;
    colvals[1623] = 180;
    colvals[1624] = 181;
    colvals[1625] = 182;
    colvals[1626] = 183;
    colvals[1627] = 184;
    colvals[1628] = 185;
    colvals[1629] = 186;
    colvals[1630] = 187;
    colvals[1631] = 189;
    colvals[1632] = 190;
    colvals[1633] = 191;
    colvals[1634] = 193;
    colvals[1635] = 195;
    colvals[1636] = 197;
    colvals[1637] = 199;
    colvals[1638] = 203;
    colvals[1639] = 204;
    colvals[1640] = 206;
    colvals[1641] = 207;
    colvals[1642] = 209;
    colvals[1643] = 211;
    colvals[1644] = 212;
    colvals[1645] = 213;
    colvals[1646] = 214;
    colvals[1647] = 51;
    colvals[1648] = 52;
    colvals[1649] = 53;
    colvals[1650] = 54;
    colvals[1651] = 55;
    colvals[1652] = 75;
    colvals[1653] = 76;
    colvals[1654] = 77;
    colvals[1655] = 79;
    colvals[1656] = 88;
    colvals[1657] = 94;
    colvals[1658] = 95;
    colvals[1659] = 96;
    colvals[1660] = 97;
    colvals[1661] = 99;
    colvals[1662] = 100;
    colvals[1663] = 101;
    colvals[1664] = 102;
    colvals[1665] = 103;
    colvals[1666] = 104;
    colvals[1667] = 107;
    colvals[1668] = 112;
    colvals[1669] = 114;
    colvals[1670] = 128;
    colvals[1671] = 131;
    colvals[1672] = 132;
    colvals[1673] = 138;
    colvals[1674] = 157;
    colvals[1675] = 159;
    colvals[1676] = 162;
    colvals[1677] = 164;
    colvals[1678] = 166;
    colvals[1679] = 169;
    colvals[1680] = 174;
    colvals[1681] = 175;
    colvals[1682] = 176;
    colvals[1683] = 177;
    colvals[1684] = 180;
    colvals[1685] = 181;
    colvals[1686] = 183;
    colvals[1687] = 185;
    colvals[1688] = 211;
    colvals[1689] = 213;
    colvals[1690] = 17;
    colvals[1691] = 51;
    colvals[1692] = 52;
    colvals[1693] = 75;
    colvals[1694] = 76;
    colvals[1695] = 77;
    colvals[1696] = 78;
    colvals[1697] = 83;
    colvals[1698] = 88;
    colvals[1699] = 89;
    colvals[1700] = 90;
    colvals[1701] = 95;
    colvals[1702] = 96;
    colvals[1703] = 97;
    colvals[1704] = 100;
    colvals[1705] = 101;
    colvals[1706] = 102;
    colvals[1707] = 104;
    colvals[1708] = 112;
    colvals[1709] = 119;
    colvals[1710] = 129;
    colvals[1711] = 131;
    colvals[1712] = 133;
    colvals[1713] = 134;
    colvals[1714] = 138;
    colvals[1715] = 142;
    colvals[1716] = 143;
    colvals[1717] = 156;
    colvals[1718] = 157;
    colvals[1719] = 160;
    colvals[1720] = 163;
    colvals[1721] = 166;
    colvals[1722] = 169;
    colvals[1723] = 171;
    colvals[1724] = 174;
    colvals[1725] = 175;
    colvals[1726] = 176;
    colvals[1727] = 178;
    colvals[1728] = 179;
    colvals[1729] = 180;
    colvals[1730] = 181;
    colvals[1731] = 183;
    colvals[1732] = 184;
    colvals[1733] = 189;
    colvals[1734] = 207;
    colvals[1735] = 212;
    colvals[1736] = 213;
    colvals[1737] = 214;
    colvals[1738] = 18;
    colvals[1739] = 51;
    colvals[1740] = 52;
    colvals[1741] = 53;
    colvals[1742] = 62;
    colvals[1743] = 75;
    colvals[1744] = 77;
    colvals[1745] = 98;
    colvals[1746] = 99;
    colvals[1747] = 100;
    colvals[1748] = 101;
    colvals[1749] = 102;
    colvals[1750] = 109;
    colvals[1751] = 110;
    colvals[1752] = 119;
    colvals[1753] = 121;
    colvals[1754] = 122;
    colvals[1755] = 132;
    colvals[1756] = 135;
    colvals[1757] = 136;
    colvals[1758] = 138;
    colvals[1759] = 146;
    colvals[1760] = 147;
    colvals[1761] = 154;
    colvals[1762] = 156;
    colvals[1763] = 166;
    colvals[1764] = 172;
    colvals[1765] = 174;
    colvals[1766] = 181;
    colvals[1767] = 182;
    colvals[1768] = 183;
    colvals[1769] = 185;
    colvals[1770] = 187;
    colvals[1771] = 189;
    colvals[1772] = 211;
    colvals[1773] = 51;
    colvals[1774] = 52;
    colvals[1775] = 53;
    colvals[1776] = 54;
    colvals[1777] = 75;
    colvals[1778] = 76;
    colvals[1779] = 88;
    colvals[1780] = 98;
    colvals[1781] = 99;
    colvals[1782] = 100;
    colvals[1783] = 102;
    colvals[1784] = 103;
    colvals[1785] = 109;
    colvals[1786] = 135;
    colvals[1787] = 138;
    colvals[1788] = 147;
    colvals[1789] = 148;
    colvals[1790] = 154;
    colvals[1791] = 157;
    colvals[1792] = 172;
    colvals[1793] = 174;
    colvals[1794] = 176;
    colvals[1795] = 181;
    colvals[1796] = 185;
    colvals[1797] = 186;
    colvals[1798] = 189;
    colvals[1799] = 211;
    colvals[1800] = 51;
    colvals[1801] = 52;
    colvals[1802] = 53;
    colvals[1803] = 54;
    colvals[1804] = 55;
    colvals[1805] = 56;
    colvals[1806] = 57;
    colvals[1807] = 58;
    colvals[1808] = 63;
    colvals[1809] = 65;
    colvals[1810] = 66;
    colvals[1811] = 67;
    colvals[1812] = 68;
    colvals[1813] = 70;
    colvals[1814] = 74;
    colvals[1815] = 75;
    colvals[1816] = 76;
    colvals[1817] = 77;
    colvals[1818] = 78;
    colvals[1819] = 80;
    colvals[1820] = 81;
    colvals[1821] = 85;
    colvals[1822] = 86;
    colvals[1823] = 87;
    colvals[1824] = 88;
    colvals[1825] = 89;
    colvals[1826] = 90;
    colvals[1827] = 91;
    colvals[1828] = 92;
    colvals[1829] = 94;
    colvals[1830] = 95;
    colvals[1831] = 96;
    colvals[1832] = 98;
    colvals[1833] = 99;
    colvals[1834] = 100;
    colvals[1835] = 101;
    colvals[1836] = 102;
    colvals[1837] = 103;
    colvals[1838] = 104;
    colvals[1839] = 105;
    colvals[1840] = 107;
    colvals[1841] = 108;
    colvals[1842] = 110;
    colvals[1843] = 111;
    colvals[1844] = 112;
    colvals[1845] = 113;
    colvals[1846] = 114;
    colvals[1847] = 115;
    colvals[1848] = 117;
    colvals[1849] = 119;
    colvals[1850] = 120;
    colvals[1851] = 121;
    colvals[1852] = 122;
    colvals[1853] = 123;
    colvals[1854] = 124;
    colvals[1855] = 126;
    colvals[1856] = 127;
    colvals[1857] = 129;
    colvals[1858] = 130;
    colvals[1859] = 131;
    colvals[1860] = 132;
    colvals[1861] = 133;
    colvals[1862] = 135;
    colvals[1863] = 136;
    colvals[1864] = 137;
    colvals[1865] = 138;
    colvals[1866] = 139;
    colvals[1867] = 143;
    colvals[1868] = 144;
    colvals[1869] = 145;
    colvals[1870] = 146;
    colvals[1871] = 148;
    colvals[1872] = 149;
    colvals[1873] = 150;
    colvals[1874] = 151;
    colvals[1875] = 152;
    colvals[1876] = 153;
    colvals[1877] = 154;
    colvals[1878] = 155;
    colvals[1879] = 156;
    colvals[1880] = 157;
    colvals[1881] = 159;
    colvals[1882] = 160;
    colvals[1883] = 161;
    colvals[1884] = 162;
    colvals[1885] = 163;
    colvals[1886] = 164;
    colvals[1887] = 165;
    colvals[1888] = 166;
    colvals[1889] = 167;
    colvals[1890] = 168;
    colvals[1891] = 169;
    colvals[1892] = 170;
    colvals[1893] = 173;
    colvals[1894] = 174;
    colvals[1895] = 175;
    colvals[1896] = 176;
    colvals[1897] = 177;
    colvals[1898] = 179;
    colvals[1899] = 181;
    colvals[1900] = 182;
    colvals[1901] = 183;
    colvals[1902] = 184;
    colvals[1903] = 185;
    colvals[1904] = 186;
    colvals[1905] = 187;
    colvals[1906] = 188;
    colvals[1907] = 189;
    colvals[1908] = 190;
    colvals[1909] = 192;
    colvals[1910] = 194;
    colvals[1911] = 196;
    colvals[1912] = 198;
    colvals[1913] = 199;
    colvals[1914] = 200;
    colvals[1915] = 201;
    colvals[1916] = 202;
    colvals[1917] = 204;
    colvals[1918] = 205;
    colvals[1919] = 206;
    colvals[1920] = 207;
    colvals[1921] = 208;
    colvals[1922] = 210;
    colvals[1923] = 211;
    colvals[1924] = 212;
    colvals[1925] = 214;
    colvals[1926] = 51;
    colvals[1927] = 52;
    colvals[1928] = 53;
    colvals[1929] = 54;
    colvals[1930] = 55;
    colvals[1931] = 56;
    colvals[1932] = 57;
    colvals[1933] = 58;
    colvals[1934] = 59;
    colvals[1935] = 60;
    colvals[1936] = 61;
    colvals[1937] = 63;
    colvals[1938] = 64;
    colvals[1939] = 66;
    colvals[1940] = 67;
    colvals[1941] = 68;
    colvals[1942] = 69;
    colvals[1943] = 70;
    colvals[1944] = 72;
    colvals[1945] = 75;
    colvals[1946] = 76;
    colvals[1947] = 77;
    colvals[1948] = 78;
    colvals[1949] = 79;
    colvals[1950] = 80;
    colvals[1951] = 81;
    colvals[1952] = 85;
    colvals[1953] = 86;
    colvals[1954] = 87;
    colvals[1955] = 88;
    colvals[1956] = 89;
    colvals[1957] = 90;
    colvals[1958] = 91;
    colvals[1959] = 92;
    colvals[1960] = 93;
    colvals[1961] = 94;
    colvals[1962] = 95;
    colvals[1963] = 96;
    colvals[1964] = 97;
    colvals[1965] = 98;
    colvals[1966] = 99;
    colvals[1967] = 100;
    colvals[1968] = 101;
    colvals[1969] = 102;
    colvals[1970] = 103;
    colvals[1971] = 104;
    colvals[1972] = 105;
    colvals[1973] = 106;
    colvals[1974] = 107;
    colvals[1975] = 108;
    colvals[1976] = 109;
    colvals[1977] = 110;
    colvals[1978] = 111;
    colvals[1979] = 112;
    colvals[1980] = 113;
    colvals[1981] = 114;
    colvals[1982] = 115;
    colvals[1983] = 116;
    colvals[1984] = 117;
    colvals[1985] = 118;
    colvals[1986] = 119;
    colvals[1987] = 120;
    colvals[1988] = 121;
    colvals[1989] = 122;
    colvals[1990] = 123;
    colvals[1991] = 124;
    colvals[1992] = 125;
    colvals[1993] = 126;
    colvals[1994] = 127;
    colvals[1995] = 128;
    colvals[1996] = 129;
    colvals[1997] = 130;
    colvals[1998] = 131;
    colvals[1999] = 132;
    colvals[2000] = 133;
    colvals[2001] = 135;
    colvals[2002] = 136;
    colvals[2003] = 137;
    colvals[2004] = 138;
    colvals[2005] = 139;
    colvals[2006] = 140;
    colvals[2007] = 142;
    colvals[2008] = 143;
    colvals[2009] = 144;
    colvals[2010] = 145;
    colvals[2011] = 146;
    colvals[2012] = 147;
    colvals[2013] = 148;
    colvals[2014] = 149;
    colvals[2015] = 150;
    colvals[2016] = 151;
    colvals[2017] = 152;
    colvals[2018] = 153;
    colvals[2019] = 154;
    colvals[2020] = 156;
    colvals[2021] = 157;
    colvals[2022] = 158;
    colvals[2023] = 159;
    colvals[2024] = 160;
    colvals[2025] = 161;
    colvals[2026] = 162;
    colvals[2027] = 163;
    colvals[2028] = 164;
    colvals[2029] = 165;
    colvals[2030] = 166;
    colvals[2031] = 167;
    colvals[2032] = 168;
    colvals[2033] = 169;
    colvals[2034] = 171;
    colvals[2035] = 172;
    colvals[2036] = 174;
    colvals[2037] = 175;
    colvals[2038] = 176;
    colvals[2039] = 177;
    colvals[2040] = 178;
    colvals[2041] = 179;
    colvals[2042] = 180;
    colvals[2043] = 181;
    colvals[2044] = 183;
    colvals[2045] = 184;
    colvals[2046] = 185;
    colvals[2047] = 186;
    colvals[2048] = 187;
    colvals[2049] = 189;
    colvals[2050] = 190;
    colvals[2051] = 191;
    colvals[2052] = 193;
    colvals[2053] = 195;
    colvals[2054] = 197;
    colvals[2055] = 198;
    colvals[2056] = 199;
    colvals[2057] = 200;
    colvals[2058] = 201;
    colvals[2059] = 202;
    colvals[2060] = 203;
    colvals[2061] = 204;
    colvals[2062] = 205;
    colvals[2063] = 206;
    colvals[2064] = 207;
    colvals[2065] = 208;
    colvals[2066] = 209;
    colvals[2067] = 210;
    colvals[2068] = 211;
    colvals[2069] = 213;
    colvals[2070] = 214;
    colvals[2071] = 53;
    colvals[2072] = 55;
    colvals[2073] = 57;
    colvals[2074] = 59;
    colvals[2075] = 60;
    colvals[2076] = 64;
    colvals[2077] = 75;
    colvals[2078] = 76;
    colvals[2079] = 77;
    colvals[2080] = 78;
    colvals[2081] = 80;
    colvals[2082] = 83;
    colvals[2083] = 86;
    colvals[2084] = 88;
    colvals[2085] = 91;
    colvals[2086] = 92;
    colvals[2087] = 94;
    colvals[2088] = 96;
    colvals[2089] = 97;
    colvals[2090] = 98;
    colvals[2091] = 100;
    colvals[2092] = 101;
    colvals[2093] = 102;
    colvals[2094] = 103;
    colvals[2095] = 104;
    colvals[2096] = 107;
    colvals[2097] = 109;
    colvals[2098] = 112;
    colvals[2099] = 114;
    colvals[2100] = 116;
    colvals[2101] = 118;
    colvals[2102] = 119;
    colvals[2103] = 126;
    colvals[2104] = 128;
    colvals[2105] = 129;
    colvals[2106] = 131;
    colvals[2107] = 135;
    colvals[2108] = 137;
    colvals[2109] = 138;
    colvals[2110] = 140;
    colvals[2111] = 141;
    colvals[2112] = 142;
    colvals[2113] = 147;
    colvals[2114] = 148;
    colvals[2115] = 149;
    colvals[2116] = 154;
    colvals[2117] = 162;
    colvals[2118] = 163;
    colvals[2119] = 164;
    colvals[2120] = 166;
    colvals[2121] = 169;
    colvals[2122] = 171;
    colvals[2123] = 172;
    colvals[2124] = 174;
    colvals[2125] = 175;
    colvals[2126] = 176;
    colvals[2127] = 181;
    colvals[2128] = 183;
    colvals[2129] = 185;
    colvals[2130] = 187;
    colvals[2131] = 189;
    colvals[2132] = 191;
    colvals[2133] = 193;
    colvals[2134] = 195;
    colvals[2135] = 197;
    colvals[2136] = 199;
    colvals[2137] = 201;
    colvals[2138] = 203;
    colvals[2139] = 206;
    colvals[2140] = 209;
    colvals[2141] = 211;
    colvals[2142] = 213;
    colvals[2143] = 51;
    colvals[2144] = 52;
    colvals[2145] = 53;
    colvals[2146] = 54;
    colvals[2147] = 55;
    colvals[2148] = 56;
    colvals[2149] = 57;
    colvals[2150] = 58;
    colvals[2151] = 59;
    colvals[2152] = 60;
    colvals[2153] = 61;
    colvals[2154] = 62;
    colvals[2155] = 64;
    colvals[2156] = 75;
    colvals[2157] = 76;
    colvals[2158] = 77;
    colvals[2159] = 78;
    colvals[2160] = 80;
    colvals[2161] = 81;
    colvals[2162] = 82;
    colvals[2163] = 83;
    colvals[2164] = 86;
    colvals[2165] = 87;
    colvals[2166] = 88;
    colvals[2167] = 89;
    colvals[2168] = 90;
    colvals[2169] = 91;
    colvals[2170] = 92;
    colvals[2171] = 93;
    colvals[2172] = 94;
    colvals[2173] = 95;
    colvals[2174] = 96;
    colvals[2175] = 97;
    colvals[2176] = 98;
    colvals[2177] = 99;
    colvals[2178] = 100;
    colvals[2179] = 101;
    colvals[2180] = 102;
    colvals[2181] = 103;
    colvals[2182] = 104;
    colvals[2183] = 106;
    colvals[2184] = 107;
    colvals[2185] = 108;
    colvals[2186] = 109;
    colvals[2187] = 111;
    colvals[2188] = 112;
    colvals[2189] = 113;
    colvals[2190] = 114;
    colvals[2191] = 115;
    colvals[2192] = 118;
    colvals[2193] = 119;
    colvals[2194] = 120;
    colvals[2195] = 121;
    colvals[2196] = 122;
    colvals[2197] = 123;
    colvals[2198] = 126;
    colvals[2199] = 127;
    colvals[2200] = 128;
    colvals[2201] = 129;
    colvals[2202] = 131;
    colvals[2203] = 134;
    colvals[2204] = 135;
    colvals[2205] = 138;
    colvals[2206] = 139;
    colvals[2207] = 140;
    colvals[2208] = 142;
    colvals[2209] = 145;
    colvals[2210] = 147;
    colvals[2211] = 148;
    colvals[2212] = 149;
    colvals[2213] = 154;
    colvals[2214] = 156;
    colvals[2215] = 157;
    colvals[2216] = 158;
    colvals[2217] = 159;
    colvals[2218] = 162;
    colvals[2219] = 163;
    colvals[2220] = 164;
    colvals[2221] = 165;
    colvals[2222] = 166;
    colvals[2223] = 167;
    colvals[2224] = 168;
    colvals[2225] = 169;
    colvals[2226] = 171;
    colvals[2227] = 172;
    colvals[2228] = 174;
    colvals[2229] = 175;
    colvals[2230] = 176;
    colvals[2231] = 178;
    colvals[2232] = 179;
    colvals[2233] = 181;
    colvals[2234] = 183;
    colvals[2235] = 184;
    colvals[2236] = 185;
    colvals[2237] = 186;
    colvals[2238] = 187;
    colvals[2239] = 189;
    colvals[2240] = 190;
    colvals[2241] = 197;
    colvals[2242] = 198;
    colvals[2243] = 199;
    colvals[2244] = 200;
    colvals[2245] = 201;
    colvals[2246] = 202;
    colvals[2247] = 203;
    colvals[2248] = 204;
    colvals[2249] = 205;
    colvals[2250] = 206;
    colvals[2251] = 207;
    colvals[2252] = 209;
    colvals[2253] = 211;
    colvals[2254] = 213;
    colvals[2255] = 214;
    colvals[2256] = 51;
    colvals[2257] = 53;
    colvals[2258] = 55;
    colvals[2259] = 57;
    colvals[2260] = 60;
    colvals[2261] = 75;
    colvals[2262] = 77;
    colvals[2263] = 88;
    colvals[2264] = 93;
    colvals[2265] = 95;
    colvals[2266] = 97;
    colvals[2267] = 100;
    colvals[2268] = 101;
    colvals[2269] = 102;
    colvals[2270] = 103;
    colvals[2271] = 104;
    colvals[2272] = 107;
    colvals[2273] = 112;
    colvals[2274] = 114;
    colvals[2275] = 119;
    colvals[2276] = 128;
    colvals[2277] = 131;
    colvals[2278] = 137;
    colvals[2279] = 138;
    colvals[2280] = 139;
    colvals[2281] = 156;
    colvals[2282] = 158;
    colvals[2283] = 162;
    colvals[2284] = 164;
    colvals[2285] = 166;
    colvals[2286] = 169;
    colvals[2287] = 174;
    colvals[2288] = 176;
    colvals[2289] = 183;
    colvals[2290] = 90;
    colvals[2291] = 95;
    colvals[2292] = 100;
    colvals[2293] = 103;
    colvals[2294] = 105;
    colvals[2295] = 112;
    colvals[2296] = 119;
    colvals[2297] = 126;
    colvals[2298] = 127;
    colvals[2299] = 19;
    colvals[2300] = 51;
    colvals[2301] = 61;
    colvals[2302] = 80;
    colvals[2303] = 101;
    colvals[2304] = 106;
    colvals[2305] = 156;
    colvals[2306] = 174;
    colvals[2307] = 20;
    colvals[2308] = 52;
    colvals[2309] = 56;
    colvals[2310] = 58;
    colvals[2311] = 59;
    colvals[2312] = 60;
    colvals[2313] = 61;
    colvals[2314] = 75;
    colvals[2315] = 76;
    colvals[2316] = 77;
    colvals[2317] = 78;
    colvals[2318] = 80;
    colvals[2319] = 81;
    colvals[2320] = 86;
    colvals[2321] = 87;
    colvals[2322] = 89;
    colvals[2323] = 90;
    colvals[2324] = 93;
    colvals[2325] = 94;
    colvals[2326] = 96;
    colvals[2327] = 100;
    colvals[2328] = 101;
    colvals[2329] = 102;
    colvals[2330] = 104;
    colvals[2331] = 107;
    colvals[2332] = 108;
    colvals[2333] = 112;
    colvals[2334] = 113;
    colvals[2335] = 114;
    colvals[2336] = 119;
    colvals[2337] = 120;
    colvals[2338] = 122;
    colvals[2339] = 123;
    colvals[2340] = 128;
    colvals[2341] = 129;
    colvals[2342] = 130;
    colvals[2343] = 131;
    colvals[2344] = 132;
    colvals[2345] = 134;
    colvals[2346] = 138;
    colvals[2347] = 140;
    colvals[2348] = 142;
    colvals[2349] = 143;
    colvals[2350] = 154;
    colvals[2351] = 157;
    colvals[2352] = 159;
    colvals[2353] = 160;
    colvals[2354] = 163;
    colvals[2355] = 164;
    colvals[2356] = 165;
    colvals[2357] = 166;
    colvals[2358] = 167;
    colvals[2359] = 169;
    colvals[2360] = 171;
    colvals[2361] = 174;
    colvals[2362] = 175;
    colvals[2363] = 176;
    colvals[2364] = 177;
    colvals[2365] = 178;
    colvals[2366] = 179;
    colvals[2367] = 183;
    colvals[2368] = 184;
    colvals[2369] = 185;
    colvals[2370] = 186;
    colvals[2371] = 188;
    colvals[2372] = 189;
    colvals[2373] = 207;
    colvals[2374] = 212;
    colvals[2375] = 52;
    colvals[2376] = 53;
    colvals[2377] = 55;
    colvals[2378] = 58;
    colvals[2379] = 75;
    colvals[2380] = 76;
    colvals[2381] = 77;
    colvals[2382] = 78;
    colvals[2383] = 81;
    colvals[2384] = 86;
    colvals[2385] = 88;
    colvals[2386] = 89;
    colvals[2387] = 90;
    colvals[2388] = 94;
    colvals[2389] = 96;
    colvals[2390] = 97;
    colvals[2391] = 100;
    colvals[2392] = 102;
    colvals[2393] = 104;
    colvals[2394] = 107;
    colvals[2395] = 108;
    colvals[2396] = 112;
    colvals[2397] = 113;
    colvals[2398] = 119;
    colvals[2399] = 128;
    colvals[2400] = 129;
    colvals[2401] = 131;
    colvals[2402] = 132;
    colvals[2403] = 138;
    colvals[2404] = 140;
    colvals[2405] = 143;
    colvals[2406] = 154;
    colvals[2407] = 157;
    colvals[2408] = 159;
    colvals[2409] = 160;
    colvals[2410] = 162;
    colvals[2411] = 163;
    colvals[2412] = 164;
    colvals[2413] = 165;
    colvals[2414] = 166;
    colvals[2415] = 169;
    colvals[2416] = 174;
    colvals[2417] = 175;
    colvals[2418] = 176;
    colvals[2419] = 177;
    colvals[2420] = 179;
    colvals[2421] = 183;
    colvals[2422] = 184;
    colvals[2423] = 185;
    colvals[2424] = 189;
    colvals[2425] = 21;
    colvals[2426] = 52;
    colvals[2427] = 80;
    colvals[2428] = 100;
    colvals[2429] = 102;
    colvals[2430] = 109;
    colvals[2431] = 110;
    colvals[2432] = 119;
    colvals[2433] = 121;
    colvals[2434] = 132;
    colvals[2435] = 138;
    colvals[2436] = 185;
    colvals[2437] = 57;
    colvals[2438] = 60;
    colvals[2439] = 78;
    colvals[2440] = 80;
    colvals[2441] = 81;
    colvals[2442] = 100;
    colvals[2443] = 102;
    colvals[2444] = 109;
    colvals[2445] = 110;
    colvals[2446] = 119;
    colvals[2447] = 135;
    colvals[2448] = 147;
    colvals[2449] = 181;
    colvals[2450] = 186;
    colvals[2451] = 212;
    colvals[2452] = 100;
    colvals[2453] = 111;
    colvals[2454] = 119;
    colvals[2455] = 142;
    colvals[2456] = 165;
    colvals[2457] = 176;
    colvals[2458] = 22;
    colvals[2459] = 52;
    colvals[2460] = 53;
    colvals[2461] = 54;
    colvals[2462] = 55;
    colvals[2463] = 57;
    colvals[2464] = 58;
    colvals[2465] = 59;
    colvals[2466] = 60;
    colvals[2467] = 61;
    colvals[2468] = 62;
    colvals[2469] = 63;
    colvals[2470] = 65;
    colvals[2471] = 74;
    colvals[2472] = 75;
    colvals[2473] = 76;
    colvals[2474] = 77;
    colvals[2475] = 78;
    colvals[2476] = 80;
    colvals[2477] = 81;
    colvals[2478] = 82;
    colvals[2479] = 83;
    colvals[2480] = 86;
    colvals[2481] = 87;
    colvals[2482] = 88;
    colvals[2483] = 89;
    colvals[2484] = 90;
    colvals[2485] = 94;
    colvals[2486] = 96;
    colvals[2487] = 98;
    colvals[2488] = 100;
    colvals[2489] = 101;
    colvals[2490] = 102;
    colvals[2491] = 103;
    colvals[2492] = 104;
    colvals[2493] = 105;
    colvals[2494] = 107;
    colvals[2495] = 108;
    colvals[2496] = 112;
    colvals[2497] = 113;
    colvals[2498] = 114;
    colvals[2499] = 115;
    colvals[2500] = 119;
    colvals[2501] = 120;
    colvals[2502] = 122;
    colvals[2503] = 128;
    colvals[2504] = 129;
    colvals[2505] = 131;
    colvals[2506] = 132;
    colvals[2507] = 133;
    colvals[2508] = 134;
    colvals[2509] = 138;
    colvals[2510] = 140;
    colvals[2511] = 142;
    colvals[2512] = 143;
    colvals[2513] = 146;
    colvals[2514] = 148;
    colvals[2515] = 149;
    colvals[2516] = 151;
    colvals[2517] = 153;
    colvals[2518] = 154;
    colvals[2519] = 157;
    colvals[2520] = 159;
    colvals[2521] = 160;
    colvals[2522] = 162;
    colvals[2523] = 163;
    colvals[2524] = 164;
    colvals[2525] = 165;
    colvals[2526] = 166;
    colvals[2527] = 167;
    colvals[2528] = 169;
    colvals[2529] = 174;
    colvals[2530] = 175;
    colvals[2531] = 176;
    colvals[2532] = 178;
    colvals[2533] = 179;
    colvals[2534] = 181;
    colvals[2535] = 183;
    colvals[2536] = 184;
    colvals[2537] = 185;
    colvals[2538] = 187;
    colvals[2539] = 189;
    colvals[2540] = 190;
    colvals[2541] = 197;
    colvals[2542] = 198;
    colvals[2543] = 199;
    colvals[2544] = 204;
    colvals[2545] = 205;
    colvals[2546] = 206;
    colvals[2547] = 212;
    colvals[2548] = 51;
    colvals[2549] = 53;
    colvals[2550] = 55;
    colvals[2551] = 57;
    colvals[2552] = 75;
    colvals[2553] = 77;
    colvals[2554] = 88;
    colvals[2555] = 90;
    colvals[2556] = 95;
    colvals[2557] = 96;
    colvals[2558] = 100;
    colvals[2559] = 102;
    colvals[2560] = 103;
    colvals[2561] = 104;
    colvals[2562] = 107;
    colvals[2563] = 112;
    colvals[2564] = 113;
    colvals[2565] = 114;
    colvals[2566] = 119;
    colvals[2567] = 128;
    colvals[2568] = 129;
    colvals[2569] = 131;
    colvals[2570] = 132;
    colvals[2571] = 138;
    colvals[2572] = 140;
    colvals[2573] = 143;
    colvals[2574] = 154;
    colvals[2575] = 156;
    colvals[2576] = 157;
    colvals[2577] = 159;
    colvals[2578] = 160;
    colvals[2579] = 162;
    colvals[2580] = 163;
    colvals[2581] = 164;
    colvals[2582] = 166;
    colvals[2583] = 169;
    colvals[2584] = 174;
    colvals[2585] = 175;
    colvals[2586] = 176;
    colvals[2587] = 179;
    colvals[2588] = 181;
    colvals[2589] = 183;
    colvals[2590] = 184;
    colvals[2591] = 185;
    colvals[2592] = 189;
    colvals[2593] = 213;
    colvals[2594] = 23;
    colvals[2595] = 52;
    colvals[2596] = 58;
    colvals[2597] = 65;
    colvals[2598] = 76;
    colvals[2599] = 78;
    colvals[2600] = 80;
    colvals[2601] = 81;
    colvals[2602] = 89;
    colvals[2603] = 90;
    colvals[2604] = 96;
    colvals[2605] = 100;
    colvals[2606] = 101;
    colvals[2607] = 102;
    colvals[2608] = 103;
    colvals[2609] = 104;
    colvals[2610] = 107;
    colvals[2611] = 112;
    colvals[2612] = 113;
    colvals[2613] = 114;
    colvals[2614] = 115;
    colvals[2615] = 119;
    colvals[2616] = 120;
    colvals[2617] = 122;
    colvals[2618] = 123;
    colvals[2619] = 128;
    colvals[2620] = 130;
    colvals[2621] = 131;
    colvals[2622] = 132;
    colvals[2623] = 138;
    colvals[2624] = 140;
    colvals[2625] = 147;
    colvals[2626] = 148;
    colvals[2627] = 150;
    colvals[2628] = 151;
    colvals[2629] = 154;
    colvals[2630] = 157;
    colvals[2631] = 159;
    colvals[2632] = 165;
    colvals[2633] = 166;
    colvals[2634] = 167;
    colvals[2635] = 169;
    colvals[2636] = 174;
    colvals[2637] = 175;
    colvals[2638] = 177;
    colvals[2639] = 182;
    colvals[2640] = 183;
    colvals[2641] = 184;
    colvals[2642] = 185;
    colvals[2643] = 186;
    colvals[2644] = 189;
    colvals[2645] = 212;
    colvals[2646] = 51;
    colvals[2647] = 52;
    colvals[2648] = 58;
    colvals[2649] = 89;
    colvals[2650] = 96;
    colvals[2651] = 100;
    colvals[2652] = 101;
    colvals[2653] = 102;
    colvals[2654] = 103;
    colvals[2655] = 104;
    colvals[2656] = 107;
    colvals[2657] = 112;
    colvals[2658] = 113;
    colvals[2659] = 114;
    colvals[2660] = 115;
    colvals[2661] = 119;
    colvals[2662] = 123;
    colvals[2663] = 131;
    colvals[2664] = 132;
    colvals[2665] = 138;
    colvals[2666] = 147;
    colvals[2667] = 148;
    colvals[2668] = 154;
    colvals[2669] = 156;
    colvals[2670] = 157;
    colvals[2671] = 159;
    colvals[2672] = 165;
    colvals[2673] = 166;
    colvals[2674] = 169;
    colvals[2675] = 174;
    colvals[2676] = 175;
    colvals[2677] = 177;
    colvals[2678] = 182;
    colvals[2679] = 184;
    colvals[2680] = 185;
    colvals[2681] = 186;
    colvals[2682] = 189;
    colvals[2683] = 24;
    colvals[2684] = 102;
    colvals[2685] = 116;
    colvals[2686] = 138;
    colvals[2687] = 86;
    colvals[2688] = 100;
    colvals[2689] = 102;
    colvals[2690] = 116;
    colvals[2691] = 117;
    colvals[2692] = 119;
    colvals[2693] = 122;
    colvals[2694] = 123;
    colvals[2695] = 132;
    colvals[2696] = 149;
    colvals[2697] = 185;
    colvals[2698] = 188;
    colvals[2699] = 25;
    colvals[2700] = 102;
    colvals[2701] = 118;
    colvals[2702] = 138;
    colvals[2703] = 174;
    colvals[2704] = 201;
    colvals[2705] = 51;
    colvals[2706] = 53;
    colvals[2707] = 55;
    colvals[2708] = 62;
    colvals[2709] = 64;
    colvals[2710] = 75;
    colvals[2711] = 77;
    colvals[2712] = 80;
    colvals[2713] = 83;
    colvals[2714] = 86;
    colvals[2715] = 88;
    colvals[2716] = 91;
    colvals[2717] = 93;
    colvals[2718] = 95;
    colvals[2719] = 97;
    colvals[2720] = 98;
    colvals[2721] = 100;
    colvals[2722] = 103;
    colvals[2723] = 104;
    colvals[2724] = 107;
    colvals[2725] = 109;
    colvals[2726] = 112;
    colvals[2727] = 114;
    colvals[2728] = 119;
    colvals[2729] = 126;
    colvals[2730] = 128;
    colvals[2731] = 131;
    colvals[2732] = 134;
    colvals[2733] = 135;
    colvals[2734] = 139;
    colvals[2735] = 140;
    colvals[2736] = 142;
    colvals[2737] = 147;
    colvals[2738] = 149;
    colvals[2739] = 154;
    colvals[2740] = 158;
    colvals[2741] = 162;
    colvals[2742] = 163;
    colvals[2743] = 164;
    colvals[2744] = 166;
    colvals[2745] = 169;
    colvals[2746] = 171;
    colvals[2747] = 172;
    colvals[2748] = 174;
    colvals[2749] = 176;
    colvals[2750] = 179;
    colvals[2751] = 181;
    colvals[2752] = 183;
    colvals[2753] = 185;
    colvals[2754] = 187;
    colvals[2755] = 189;
    colvals[2756] = 197;
    colvals[2757] = 199;
    colvals[2758] = 201;
    colvals[2759] = 203;
    colvals[2760] = 206;
    colvals[2761] = 209;
    colvals[2762] = 211;
    colvals[2763] = 213;
    colvals[2764] = 52;
    colvals[2765] = 56;
    colvals[2766] = 60;
    colvals[2767] = 62;
    colvals[2768] = 75;
    colvals[2769] = 76;
    colvals[2770] = 78;
    colvals[2771] = 81;
    colvals[2772] = 86;
    colvals[2773] = 88;
    colvals[2774] = 89;
    colvals[2775] = 90;
    colvals[2776] = 100;
    colvals[2777] = 102;
    colvals[2778] = 107;
    colvals[2779] = 108;
    colvals[2780] = 112;
    colvals[2781] = 113;
    colvals[2782] = 114;
    colvals[2783] = 119;
    colvals[2784] = 120;
    colvals[2785] = 122;
    colvals[2786] = 123;
    colvals[2787] = 128;
    colvals[2788] = 129;
    colvals[2789] = 130;
    colvals[2790] = 131;
    colvals[2791] = 132;
    colvals[2792] = 140;
    colvals[2793] = 143;
    colvals[2794] = 157;
    colvals[2795] = 160;
    colvals[2796] = 162;
    colvals[2797] = 163;
    colvals[2798] = 164;
    colvals[2799] = 165;
    colvals[2800] = 166;
    colvals[2801] = 174;
    colvals[2802] = 175;
    colvals[2803] = 176;
    colvals[2804] = 177;
    colvals[2805] = 179;
    colvals[2806] = 184;
    colvals[2807] = 212;
    colvals[2808] = 60;
    colvals[2809] = 78;
    colvals[2810] = 81;
    colvals[2811] = 88;
    colvals[2812] = 100;
    colvals[2813] = 109;
    colvals[2814] = 114;
    colvals[2815] = 119;
    colvals[2816] = 121;
    colvals[2817] = 132;
    colvals[2818] = 148;
    colvals[2819] = 181;
    colvals[2820] = 186;
    colvals[2821] = 212;
    colvals[2822] = 51;
    colvals[2823] = 53;
    colvals[2824] = 58;
    colvals[2825] = 60;
    colvals[2826] = 62;
    colvals[2827] = 75;
    colvals[2828] = 76;
    colvals[2829] = 77;
    colvals[2830] = 82;
    colvals[2831] = 83;
    colvals[2832] = 86;
    colvals[2833] = 88;
    colvals[2834] = 89;
    colvals[2835] = 90;
    colvals[2836] = 98;
    colvals[2837] = 100;
    colvals[2838] = 103;
    colvals[2839] = 104;
    colvals[2840] = 105;
    colvals[2841] = 107;
    colvals[2842] = 108;
    colvals[2843] = 112;
    colvals[2844] = 113;
    colvals[2845] = 114;
    colvals[2846] = 115;
    colvals[2847] = 119;
    colvals[2848] = 120;
    colvals[2849] = 122;
    colvals[2850] = 128;
    colvals[2851] = 129;
    colvals[2852] = 131;
    colvals[2853] = 132;
    colvals[2854] = 133;
    colvals[2855] = 134;
    colvals[2856] = 140;
    colvals[2857] = 143;
    colvals[2858] = 146;
    colvals[2859] = 148;
    colvals[2860] = 149;
    colvals[2861] = 153;
    colvals[2862] = 160;
    colvals[2863] = 162;
    colvals[2864] = 163;
    colvals[2865] = 164;
    colvals[2866] = 165;
    colvals[2867] = 166;
    colvals[2868] = 174;
    colvals[2869] = 179;
    colvals[2870] = 183;
    colvals[2871] = 184;
    colvals[2872] = 187;
    colvals[2873] = 189;
    colvals[2874] = 197;
    colvals[2875] = 198;
    colvals[2876] = 199;
    colvals[2877] = 204;
    colvals[2878] = 205;
    colvals[2879] = 206;
    colvals[2880] = 58;
    colvals[2881] = 76;
    colvals[2882] = 78;
    colvals[2883] = 89;
    colvals[2884] = 90;
    colvals[2885] = 100;
    colvals[2886] = 101;
    colvals[2887] = 103;
    colvals[2888] = 107;
    colvals[2889] = 113;
    colvals[2890] = 114;
    colvals[2891] = 115;
    colvals[2892] = 119;
    colvals[2893] = 120;
    colvals[2894] = 122;
    colvals[2895] = 123;
    colvals[2896] = 128;
    colvals[2897] = 130;
    colvals[2898] = 132;
    colvals[2899] = 140;
    colvals[2900] = 148;
    colvals[2901] = 150;
    colvals[2902] = 151;
    colvals[2903] = 165;
    colvals[2904] = 166;
    colvals[2905] = 184;
    colvals[2906] = 185;
    colvals[2907] = 87;
    colvals[2908] = 100;
    colvals[2909] = 107;
    colvals[2910] = 119;
    colvals[2911] = 122;
    colvals[2912] = 124;
    colvals[2913] = 132;
    colvals[2914] = 134;
    colvals[2915] = 26;
    colvals[2916] = 52;
    colvals[2917] = 55;
    colvals[2918] = 57;
    colvals[2919] = 69;
    colvals[2920] = 74;
    colvals[2921] = 93;
    colvals[2922] = 112;
    colvals[2923] = 125;
    colvals[2924] = 128;
    colvals[2925] = 138;
    colvals[2926] = 140;
    colvals[2927] = 156;
    colvals[2928] = 161;
    colvals[2929] = 27;
    colvals[2930] = 90;
    colvals[2931] = 91;
    colvals[2932] = 95;
    colvals[2933] = 100;
    colvals[2934] = 101;
    colvals[2935] = 102;
    colvals[2936] = 103;
    colvals[2937] = 105;
    colvals[2938] = 112;
    colvals[2939] = 119;
    colvals[2940] = 126;
    colvals[2941] = 138;
    colvals[2942] = 91;
    colvals[2943] = 92;
    colvals[2944] = 100;
    colvals[2945] = 102;
    colvals[2946] = 103;
    colvals[2947] = 119;
    colvals[2948] = 126;
    colvals[2949] = 127;
    colvals[2950] = 28;
    colvals[2951] = 51;
    colvals[2952] = 52;
    colvals[2953] = 53;
    colvals[2954] = 55;
    colvals[2955] = 56;
    colvals[2956] = 57;
    colvals[2957] = 58;
    colvals[2958] = 60;
    colvals[2959] = 65;
    colvals[2960] = 66;
    colvals[2961] = 68;
    colvals[2962] = 75;
    colvals[2963] = 76;
    colvals[2964] = 77;
    colvals[2965] = 80;
    colvals[2966] = 81;
    colvals[2967] = 83;
    colvals[2968] = 87;
    colvals[2969] = 88;
    colvals[2970] = 90;
    colvals[2971] = 93;
    colvals[2972] = 94;
    colvals[2973] = 96;
    colvals[2974] = 100;
    colvals[2975] = 101;
    colvals[2976] = 102;
    colvals[2977] = 103;
    colvals[2978] = 104;
    colvals[2979] = 106;
    colvals[2980] = 107;
    colvals[2981] = 108;
    colvals[2982] = 112;
    colvals[2983] = 113;
    colvals[2984] = 114;
    colvals[2985] = 119;
    colvals[2986] = 120;
    colvals[2987] = 122;
    colvals[2988] = 123;
    colvals[2989] = 125;
    colvals[2990] = 128;
    colvals[2991] = 129;
    colvals[2992] = 130;
    colvals[2993] = 131;
    colvals[2994] = 132;
    colvals[2995] = 135;
    colvals[2996] = 138;
    colvals[2997] = 140;
    colvals[2998] = 142;
    colvals[2999] = 143;
    colvals[3000] = 148;
    colvals[3001] = 151;
    colvals[3002] = 156;
    colvals[3003] = 157;
    colvals[3004] = 158;
    colvals[3005] = 159;
    colvals[3006] = 160;
    colvals[3007] = 161;
    colvals[3008] = 162;
    colvals[3009] = 163;
    colvals[3010] = 164;
    colvals[3011] = 165;
    colvals[3012] = 166;
    colvals[3013] = 169;
    colvals[3014] = 174;
    colvals[3015] = 175;
    colvals[3016] = 176;
    colvals[3017] = 179;
    colvals[3018] = 180;
    colvals[3019] = 183;
    colvals[3020] = 184;
    colvals[3021] = 185;
    colvals[3022] = 203;
    colvals[3023] = 51;
    colvals[3024] = 52;
    colvals[3025] = 53;
    colvals[3026] = 55;
    colvals[3027] = 57;
    colvals[3028] = 66;
    colvals[3029] = 75;
    colvals[3030] = 76;
    colvals[3031] = 77;
    colvals[3032] = 78;
    colvals[3033] = 88;
    colvals[3034] = 93;
    colvals[3035] = 94;
    colvals[3036] = 95;
    colvals[3037] = 96;
    colvals[3038] = 97;
    colvals[3039] = 100;
    colvals[3040] = 101;
    colvals[3041] = 102;
    colvals[3042] = 103;
    colvals[3043] = 104;
    colvals[3044] = 107;
    colvals[3045] = 112;
    colvals[3046] = 119;
    colvals[3047] = 128;
    colvals[3048] = 129;
    colvals[3049] = 131;
    colvals[3050] = 140;
    colvals[3051] = 143;
    colvals[3052] = 156;
    colvals[3053] = 157;
    colvals[3054] = 159;
    colvals[3055] = 162;
    colvals[3056] = 163;
    colvals[3057] = 164;
    colvals[3058] = 166;
    colvals[3059] = 169;
    colvals[3060] = 176;
    colvals[3061] = 179;
    colvals[3062] = 183;
    colvals[3063] = 184;
    colvals[3064] = 185;
    colvals[3065] = 56;
    colvals[3066] = 58;
    colvals[3067] = 65;
    colvals[3068] = 75;
    colvals[3069] = 76;
    colvals[3070] = 77;
    colvals[3071] = 81;
    colvals[3072] = 83;
    colvals[3073] = 88;
    colvals[3074] = 90;
    colvals[3075] = 100;
    colvals[3076] = 103;
    colvals[3077] = 107;
    colvals[3078] = 108;
    colvals[3079] = 113;
    colvals[3080] = 114;
    colvals[3081] = 119;
    colvals[3082] = 120;
    colvals[3083] = 122;
    colvals[3084] = 123;
    colvals[3085] = 128;
    colvals[3086] = 129;
    colvals[3087] = 130;
    colvals[3088] = 131;
    colvals[3089] = 132;
    colvals[3090] = 140;
    colvals[3091] = 143;
    colvals[3092] = 148;
    colvals[3093] = 151;
    colvals[3094] = 157;
    colvals[3095] = 160;
    colvals[3096] = 162;
    colvals[3097] = 163;
    colvals[3098] = 164;
    colvals[3099] = 165;
    colvals[3100] = 166;
    colvals[3101] = 179;
    colvals[3102] = 184;
    colvals[3103] = 51;
    colvals[3104] = 52;
    colvals[3105] = 53;
    colvals[3106] = 54;
    colvals[3107] = 55;
    colvals[3108] = 56;
    colvals[3109] = 57;
    colvals[3110] = 58;
    colvals[3111] = 59;
    colvals[3112] = 60;
    colvals[3113] = 75;
    colvals[3114] = 76;
    colvals[3115] = 77;
    colvals[3116] = 78;
    colvals[3117] = 80;
    colvals[3118] = 81;
    colvals[3119] = 86;
    colvals[3120] = 90;
    colvals[3121] = 93;
    colvals[3122] = 94;
    colvals[3123] = 96;
    colvals[3124] = 97;
    colvals[3125] = 100;
    colvals[3126] = 101;
    colvals[3127] = 102;
    colvals[3128] = 104;
    colvals[3129] = 107;
    colvals[3130] = 108;
    colvals[3131] = 112;
    colvals[3132] = 113;
    colvals[3133] = 115;
    colvals[3134] = 119;
    colvals[3135] = 120;
    colvals[3136] = 124;
    colvals[3137] = 128;
    colvals[3138] = 129;
    colvals[3139] = 131;
    colvals[3140] = 132;
    colvals[3141] = 138;
    colvals[3142] = 140;
    colvals[3143] = 142;
    colvals[3144] = 143;
    colvals[3145] = 154;
    colvals[3146] = 156;
    colvals[3147] = 157;
    colvals[3148] = 159;
    colvals[3149] = 160;
    colvals[3150] = 163;
    colvals[3151] = 164;
    colvals[3152] = 165;
    colvals[3153] = 166;
    colvals[3154] = 167;
    colvals[3155] = 169;
    colvals[3156] = 174;
    colvals[3157] = 175;
    colvals[3158] = 176;
    colvals[3159] = 177;
    colvals[3160] = 178;
    colvals[3161] = 179;
    colvals[3162] = 181;
    colvals[3163] = 183;
    colvals[3164] = 184;
    colvals[3165] = 185;
    colvals[3166] = 186;
    colvals[3167] = 207;
    colvals[3168] = 212;
    colvals[3169] = 51;
    colvals[3170] = 52;
    colvals[3171] = 53;
    colvals[3172] = 54;
    colvals[3173] = 55;
    colvals[3174] = 56;
    colvals[3175] = 57;
    colvals[3176] = 58;
    colvals[3177] = 62;
    colvals[3178] = 65;
    colvals[3179] = 74;
    colvals[3180] = 75;
    colvals[3181] = 76;
    colvals[3182] = 77;
    colvals[3183] = 78;
    colvals[3184] = 81;
    colvals[3185] = 82;
    colvals[3186] = 83;
    colvals[3187] = 86;
    colvals[3188] = 88;
    colvals[3189] = 89;
    colvals[3190] = 90;
    colvals[3191] = 94;
    colvals[3192] = 95;
    colvals[3193] = 96;
    colvals[3194] = 97;
    colvals[3195] = 98;
    colvals[3196] = 100;
    colvals[3197] = 102;
    colvals[3198] = 103;
    colvals[3199] = 104;
    colvals[3200] = 105;
    colvals[3201] = 107;
    colvals[3202] = 108;
    colvals[3203] = 109;
    colvals[3204] = 112;
    colvals[3205] = 113;
    colvals[3206] = 114;
    colvals[3207] = 115;
    colvals[3208] = 119;
    colvals[3209] = 122;
    colvals[3210] = 128;
    colvals[3211] = 129;
    colvals[3212] = 131;
    colvals[3213] = 132;
    colvals[3214] = 133;
    colvals[3215] = 134;
    colvals[3216] = 136;
    colvals[3217] = 138;
    colvals[3218] = 140;
    colvals[3219] = 143;
    colvals[3220] = 145;
    colvals[3221] = 147;
    colvals[3222] = 149;
    colvals[3223] = 154;
    colvals[3224] = 157;
    colvals[3225] = 159;
    colvals[3226] = 160;
    colvals[3227] = 162;
    colvals[3228] = 163;
    colvals[3229] = 164;
    colvals[3230] = 165;
    colvals[3231] = 166;
    colvals[3232] = 167;
    colvals[3233] = 172;
    colvals[3234] = 174;
    colvals[3235] = 175;
    colvals[3236] = 176;
    colvals[3237] = 177;
    colvals[3238] = 179;
    colvals[3239] = 181;
    colvals[3240] = 183;
    colvals[3241] = 184;
    colvals[3242] = 185;
    colvals[3243] = 186;
    colvals[3244] = 187;
    colvals[3245] = 189;
    colvals[3246] = 197;
    colvals[3247] = 199;
    colvals[3248] = 203;
    colvals[3249] = 204;
    colvals[3250] = 206;
    colvals[3251] = 207;
    colvals[3252] = 209;
    colvals[3253] = 211;
    colvals[3254] = 212;
    colvals[3255] = 51;
    colvals[3256] = 83;
    colvals[3257] = 88;
    colvals[3258] = 89;
    colvals[3259] = 90;
    colvals[3260] = 95;
    colvals[3261] = 97;
    colvals[3262] = 100;
    colvals[3263] = 104;
    colvals[3264] = 112;
    colvals[3265] = 119;
    colvals[3266] = 129;
    colvals[3267] = 132;
    colvals[3268] = 133;
    colvals[3269] = 134;
    colvals[3270] = 138;
    colvals[3271] = 143;
    colvals[3272] = 160;
    colvals[3273] = 163;
    colvals[3274] = 166;
    colvals[3275] = 174;
    colvals[3276] = 179;
    colvals[3277] = 183;
    colvals[3278] = 184;
    colvals[3279] = 29;
    colvals[3280] = 100;
    colvals[3281] = 119;
    colvals[3282] = 122;
    colvals[3283] = 124;
    colvals[3284] = 132;
    colvals[3285] = 134;
    colvals[3286] = 138;
    colvals[3287] = 57;
    colvals[3288] = 60;
    colvals[3289] = 77;
    colvals[3290] = 100;
    colvals[3291] = 101;
    colvals[3292] = 102;
    colvals[3293] = 110;
    colvals[3294] = 119;
    colvals[3295] = 135;
    colvals[3296] = 138;
    colvals[3297] = 156;
    colvals[3298] = 174;
    colvals[3299] = 185;
    colvals[3300] = 212;
    colvals[3301] = 51;
    colvals[3302] = 52;
    colvals[3303] = 57;
    colvals[3304] = 60;
    colvals[3305] = 62;
    colvals[3306] = 65;
    colvals[3307] = 76;
    colvals[3308] = 77;
    colvals[3309] = 78;
    colvals[3310] = 81;
    colvals[3311] = 88;
    colvals[3312] = 98;
    colvals[3313] = 99;
    colvals[3314] = 100;
    colvals[3315] = 103;
    colvals[3316] = 114;
    colvals[3317] = 115;
    colvals[3318] = 119;
    colvals[3319] = 122;
    colvals[3320] = 132;
    colvals[3321] = 135;
    colvals[3322] = 136;
    colvals[3323] = 166;
    colvals[3324] = 174;
    colvals[3325] = 181;
    colvals[3326] = 185;
    colvals[3327] = 186;
    colvals[3328] = 212;
    colvals[3329] = 51;
    colvals[3330] = 53;
    colvals[3331] = 55;
    colvals[3332] = 57;
    colvals[3333] = 59;
    colvals[3334] = 60;
    colvals[3335] = 64;
    colvals[3336] = 71;
    colvals[3337] = 72;
    colvals[3338] = 75;
    colvals[3339] = 77;
    colvals[3340] = 79;
    colvals[3341] = 80;
    colvals[3342] = 82;
    colvals[3343] = 83;
    colvals[3344] = 86;
    colvals[3345] = 88;
    colvals[3346] = 93;
    colvals[3347] = 95;
    colvals[3348] = 97;
    colvals[3349] = 98;
    colvals[3350] = 100;
    colvals[3351] = 101;
    colvals[3352] = 102;
    colvals[3353] = 103;
    colvals[3354] = 104;
    colvals[3355] = 107;
    colvals[3356] = 109;
    colvals[3357] = 112;
    colvals[3358] = 114;
    colvals[3359] = 116;
    colvals[3360] = 118;
    colvals[3361] = 125;
    colvals[3362] = 126;
    colvals[3363] = 128;
    colvals[3364] = 131;
    colvals[3365] = 134;
    colvals[3366] = 135;
    colvals[3367] = 137;
    colvals[3368] = 138;
    colvals[3369] = 139;
    colvals[3370] = 140;
    colvals[3371] = 142;
    colvals[3372] = 147;
    colvals[3373] = 149;
    colvals[3374] = 158;
    colvals[3375] = 161;
    colvals[3376] = 162;
    colvals[3377] = 164;
    colvals[3378] = 166;
    colvals[3379] = 169;
    colvals[3380] = 172;
    colvals[3381] = 176;
    colvals[3382] = 180;
    colvals[3383] = 181;
    colvals[3384] = 183;
    colvals[3385] = 187;
    colvals[3386] = 189;
    colvals[3387] = 191;
    colvals[3388] = 193;
    colvals[3389] = 195;
    colvals[3390] = 197;
    colvals[3391] = 199;
    colvals[3392] = 201;
    colvals[3393] = 203;
    colvals[3394] = 206;
    colvals[3395] = 209;
    colvals[3396] = 211;
    colvals[3397] = 213;
    colvals[3398] = 51;
    colvals[3399] = 53;
    colvals[3400] = 55;
    colvals[3401] = 57;
    colvals[3402] = 59;
    colvals[3403] = 60;
    colvals[3404] = 64;
    colvals[3405] = 71;
    colvals[3406] = 72;
    colvals[3407] = 75;
    colvals[3408] = 77;
    colvals[3409] = 79;
    colvals[3410] = 80;
    colvals[3411] = 82;
    colvals[3412] = 83;
    colvals[3413] = 86;
    colvals[3414] = 88;
    colvals[3415] = 93;
    colvals[3416] = 95;
    colvals[3417] = 97;
    colvals[3418] = 98;
    colvals[3419] = 100;
    colvals[3420] = 101;
    colvals[3421] = 103;
    colvals[3422] = 107;
    colvals[3423] = 109;
    colvals[3424] = 112;
    colvals[3425] = 114;
    colvals[3426] = 116;
    colvals[3427] = 118;
    colvals[3428] = 125;
    colvals[3429] = 126;
    colvals[3430] = 128;
    colvals[3431] = 131;
    colvals[3432] = 134;
    colvals[3433] = 135;
    colvals[3434] = 137;
    colvals[3435] = 138;
    colvals[3436] = 140;
    colvals[3437] = 142;
    colvals[3438] = 147;
    colvals[3439] = 149;
    colvals[3440] = 158;
    colvals[3441] = 161;
    colvals[3442] = 162;
    colvals[3443] = 164;
    colvals[3444] = 166;
    colvals[3445] = 169;
    colvals[3446] = 172;
    colvals[3447] = 176;
    colvals[3448] = 180;
    colvals[3449] = 181;
    colvals[3450] = 183;
    colvals[3451] = 187;
    colvals[3452] = 189;
    colvals[3453] = 191;
    colvals[3454] = 193;
    colvals[3455] = 195;
    colvals[3456] = 197;
    colvals[3457] = 199;
    colvals[3458] = 201;
    colvals[3459] = 203;
    colvals[3460] = 206;
    colvals[3461] = 209;
    colvals[3462] = 211;
    colvals[3463] = 213;
    colvals[3464] = 100;
    colvals[3465] = 101;
    colvals[3466] = 102;
    colvals[3467] = 103;
    colvals[3468] = 104;
    colvals[3469] = 131;
    colvals[3470] = 137;
    colvals[3471] = 138;
    colvals[3472] = 139;
    colvals[3473] = 30;
    colvals[3474] = 51;
    colvals[3475] = 52;
    colvals[3476] = 55;
    colvals[3477] = 56;
    colvals[3478] = 58;
    colvals[3479] = 75;
    colvals[3480] = 76;
    colvals[3481] = 77;
    colvals[3482] = 83;
    colvals[3483] = 85;
    colvals[3484] = 90;
    colvals[3485] = 93;
    colvals[3486] = 100;
    colvals[3487] = 101;
    colvals[3488] = 102;
    colvals[3489] = 107;
    colvals[3490] = 108;
    colvals[3491] = 113;
    colvals[3492] = 114;
    colvals[3493] = 119;
    colvals[3494] = 120;
    colvals[3495] = 122;
    colvals[3496] = 123;
    colvals[3497] = 129;
    colvals[3498] = 130;
    colvals[3499] = 132;
    colvals[3500] = 138;
    colvals[3501] = 140;
    colvals[3502] = 141;
    colvals[3503] = 143;
    colvals[3504] = 148;
    colvals[3505] = 156;
    colvals[3506] = 160;
    colvals[3507] = 163;
    colvals[3508] = 164;
    colvals[3509] = 165;
    colvals[3510] = 166;
    colvals[3511] = 179;
    colvals[3512] = 184;
    colvals[3513] = 31;
    colvals[3514] = 51;
    colvals[3515] = 77;
    colvals[3516] = 102;
    colvals[3517] = 141;
    colvals[3518] = 169;
    colvals[3519] = 32;
    colvals[3520] = 75;
    colvals[3521] = 77;
    colvals[3522] = 80;
    colvals[3523] = 93;
    colvals[3524] = 95;
    colvals[3525] = 100;
    colvals[3526] = 101;
    colvals[3527] = 102;
    colvals[3528] = 111;
    colvals[3529] = 119;
    colvals[3530] = 131;
    colvals[3531] = 138;
    colvals[3532] = 142;
    colvals[3533] = 143;
    colvals[3534] = 156;
    colvals[3535] = 162;
    colvals[3536] = 164;
    colvals[3537] = 169;
    colvals[3538] = 171;
    colvals[3539] = 174;
    colvals[3540] = 176;
    colvals[3541] = 183;
    colvals[3542] = 51;
    colvals[3543] = 53;
    colvals[3544] = 55;
    colvals[3545] = 75;
    colvals[3546] = 77;
    colvals[3547] = 88;
    colvals[3548] = 93;
    colvals[3549] = 95;
    colvals[3550] = 97;
    colvals[3551] = 100;
    colvals[3552] = 104;
    colvals[3553] = 107;
    colvals[3554] = 112;
    colvals[3555] = 113;
    colvals[3556] = 119;
    colvals[3557] = 128;
    colvals[3558] = 131;
    colvals[3559] = 140;
    colvals[3560] = 143;
    colvals[3561] = 156;
    colvals[3562] = 158;
    colvals[3563] = 162;
    colvals[3564] = 163;
    colvals[3565] = 164;
    colvals[3566] = 165;
    colvals[3567] = 166;
    colvals[3568] = 167;
    colvals[3569] = 169;
    colvals[3570] = 174;
    colvals[3571] = 176;
    colvals[3572] = 177;
    colvals[3573] = 179;
    colvals[3574] = 183;
    colvals[3575] = 184;
    colvals[3576] = 185;
    colvals[3577] = 100;
    colvals[3578] = 119;
    colvals[3579] = 132;
    colvals[3580] = 144;
    colvals[3581] = 165;
    colvals[3582] = 172;
    colvals[3583] = 185;
    colvals[3584] = 52;
    colvals[3585] = 95;
    colvals[3586] = 96;
    colvals[3587] = 100;
    colvals[3588] = 103;
    colvals[3589] = 112;
    colvals[3590] = 119;
    colvals[3591] = 145;
    colvals[3592] = 76;
    colvals[3593] = 81;
    colvals[3594] = 89;
    colvals[3595] = 100;
    colvals[3596] = 112;
    colvals[3597] = 119;
    colvals[3598] = 132;
    colvals[3599] = 146;
    colvals[3600] = 181;
    colvals[3601] = 211;
    colvals[3602] = 51;
    colvals[3603] = 52;
    colvals[3604] = 75;
    colvals[3605] = 80;
    colvals[3606] = 81;
    colvals[3607] = 88;
    colvals[3608] = 96;
    colvals[3609] = 98;
    colvals[3610] = 100;
    colvals[3611] = 101;
    colvals[3612] = 102;
    colvals[3613] = 103;
    colvals[3614] = 107;
    colvals[3615] = 112;
    colvals[3616] = 113;
    colvals[3617] = 114;
    colvals[3618] = 115;
    colvals[3619] = 116;
    colvals[3620] = 117;
    colvals[3621] = 119;
    colvals[3622] = 123;
    colvals[3623] = 131;
    colvals[3624] = 132;
    colvals[3625] = 135;
    colvals[3626] = 138;
    colvals[3627] = 147;
    colvals[3628] = 148;
    colvals[3629] = 149;
    colvals[3630] = 150;
    colvals[3631] = 151;
    colvals[3632] = 154;
    colvals[3633] = 156;
    colvals[3634] = 157;
    colvals[3635] = 162;
    colvals[3636] = 165;
    colvals[3637] = 166;
    colvals[3638] = 167;
    colvals[3639] = 169;
    colvals[3640] = 172;
    colvals[3641] = 174;
    colvals[3642] = 181;
    colvals[3643] = 183;
    colvals[3644] = 185;
    colvals[3645] = 186;
    colvals[3646] = 187;
    colvals[3647] = 189;
    colvals[3648] = 210;
    colvals[3649] = 211;
    colvals[3650] = 51;
    colvals[3651] = 75;
    colvals[3652] = 76;
    colvals[3653] = 88;
    colvals[3654] = 90;
    colvals[3655] = 100;
    colvals[3656] = 101;
    colvals[3657] = 102;
    colvals[3658] = 103;
    colvals[3659] = 104;
    colvals[3660] = 108;
    colvals[3661] = 112;
    colvals[3662] = 113;
    colvals[3663] = 114;
    colvals[3664] = 115;
    colvals[3665] = 116;
    colvals[3666] = 119;
    colvals[3667] = 128;
    colvals[3668] = 129;
    colvals[3669] = 131;
    colvals[3670] = 132;
    colvals[3671] = 138;
    colvals[3672] = 140;
    colvals[3673] = 143;
    colvals[3674] = 147;
    colvals[3675] = 148;
    colvals[3676] = 154;
    colvals[3677] = 156;
    colvals[3678] = 157;
    colvals[3679] = 159;
    colvals[3680] = 160;
    colvals[3681] = 163;
    colvals[3682] = 165;
    colvals[3683] = 166;
    colvals[3684] = 169;
    colvals[3685] = 174;
    colvals[3686] = 175;
    colvals[3687] = 179;
    colvals[3688] = 181;
    colvals[3689] = 184;
    colvals[3690] = 185;
    colvals[3691] = 186;
    colvals[3692] = 189;
    colvals[3693] = 100;
    colvals[3694] = 102;
    colvals[3695] = 117;
    colvals[3696] = 119;
    colvals[3697] = 122;
    colvals[3698] = 132;
    colvals[3699] = 138;
    colvals[3700] = 149;
    colvals[3701] = 100;
    colvals[3702] = 102;
    colvals[3703] = 114;
    colvals[3704] = 116;
    colvals[3705] = 119;
    colvals[3706] = 122;
    colvals[3707] = 132;
    colvals[3708] = 138;
    colvals[3709] = 148;
    colvals[3710] = 149;
    colvals[3711] = 150;
    colvals[3712] = 186;
    colvals[3713] = 187;
    colvals[3714] = 100;
    colvals[3715] = 112;
    colvals[3716] = 114;
    colvals[3717] = 119;
    colvals[3718] = 128;
    colvals[3719] = 132;
    colvals[3720] = 151;
    colvals[3721] = 166;
    colvals[3722] = 185;
    colvals[3723] = 200;
    colvals[3724] = 209;
    colvals[3725] = 100;
    colvals[3726] = 113;
    colvals[3727] = 119;
    colvals[3728] = 132;
    colvals[3729] = 152;
    colvals[3730] = 185;
    colvals[3731] = 211;
    colvals[3732] = 100;
    colvals[3733] = 103;
    colvals[3734] = 112;
    colvals[3735] = 113;
    colvals[3736] = 119;
    colvals[3737] = 153;
    colvals[3738] = 166;
    colvals[3739] = 213;
    colvals[3740] = 214;
    colvals[3741] = 33;
    colvals[3742] = 52;
    colvals[3743] = 58;
    colvals[3744] = 76;
    colvals[3745] = 81;
    colvals[3746] = 90;
    colvals[3747] = 99;
    colvals[3748] = 100;
    colvals[3749] = 102;
    colvals[3750] = 108;
    colvals[3751] = 113;
    colvals[3752] = 115;
    colvals[3753] = 119;
    colvals[3754] = 132;
    colvals[3755] = 148;
    colvals[3756] = 154;
    colvals[3757] = 155;
    colvals[3758] = 157;
    colvals[3759] = 159;
    colvals[3760] = 167;
    colvals[3761] = 170;
    colvals[3762] = 177;
    colvals[3763] = 186;
    colvals[3764] = 190;
    colvals[3765] = 207;
    colvals[3766] = 212;
    colvals[3767] = 52;
    colvals[3768] = 58;
    colvals[3769] = 76;
    colvals[3770] = 81;
    colvals[3771] = 90;
    colvals[3772] = 99;
    colvals[3773] = 100;
    colvals[3774] = 102;
    colvals[3775] = 108;
    colvals[3776] = 113;
    colvals[3777] = 115;
    colvals[3778] = 119;
    colvals[3779] = 132;
    colvals[3780] = 148;
    colvals[3781] = 154;
    colvals[3782] = 155;
    colvals[3783] = 157;
    colvals[3784] = 159;
    colvals[3785] = 167;
    colvals[3786] = 170;
    colvals[3787] = 177;
    colvals[3788] = 186;
    colvals[3789] = 190;
    colvals[3790] = 207;
    colvals[3791] = 212;
    colvals[3792] = 51;
    colvals[3793] = 52;
    colvals[3794] = 53;
    colvals[3795] = 54;
    colvals[3796] = 55;
    colvals[3797] = 56;
    colvals[3798] = 58;
    colvals[3799] = 59;
    colvals[3800] = 60;
    colvals[3801] = 61;
    colvals[3802] = 64;
    colvals[3803] = 65;
    colvals[3804] = 66;
    colvals[3805] = 69;
    colvals[3806] = 71;
    colvals[3807] = 72;
    colvals[3808] = 73;
    colvals[3809] = 75;
    colvals[3810] = 76;
    colvals[3811] = 77;
    colvals[3812] = 78;
    colvals[3813] = 80;
    colvals[3814] = 88;
    colvals[3815] = 93;
    colvals[3816] = 94;
    colvals[3817] = 95;
    colvals[3818] = 96;
    colvals[3819] = 97;
    colvals[3820] = 98;
    colvals[3821] = 100;
    colvals[3822] = 101;
    colvals[3823] = 103;
    colvals[3824] = 104;
    colvals[3825] = 106;
    colvals[3826] = 107;
    colvals[3827] = 108;
    colvals[3828] = 112;
    colvals[3829] = 113;
    colvals[3830] = 114;
    colvals[3831] = 115;
    colvals[3832] = 128;
    colvals[3833] = 131;
    colvals[3834] = 135;
    colvals[3835] = 138;
    colvals[3836] = 140;
    colvals[3837] = 142;
    colvals[3838] = 147;
    colvals[3839] = 148;
    colvals[3840] = 154;
    colvals[3841] = 156;
    colvals[3842] = 157;
    colvals[3843] = 158;
    colvals[3844] = 159;
    colvals[3845] = 160;
    colvals[3846] = 161;
    colvals[3847] = 162;
    colvals[3848] = 163;
    colvals[3849] = 164;
    colvals[3850] = 165;
    colvals[3851] = 166;
    colvals[3852] = 167;
    colvals[3853] = 169;
    colvals[3854] = 170;
    colvals[3855] = 171;
    colvals[3856] = 172;
    colvals[3857] = 173;
    colvals[3858] = 174;
    colvals[3859] = 175;
    colvals[3860] = 176;
    colvals[3861] = 177;
    colvals[3862] = 178;
    colvals[3863] = 181;
    colvals[3864] = 183;
    colvals[3865] = 184;
    colvals[3866] = 185;
    colvals[3867] = 187;
    colvals[3868] = 189;
    colvals[3869] = 191;
    colvals[3870] = 192;
    colvals[3871] = 207;
    colvals[3872] = 211;
    colvals[3873] = 212;
    colvals[3874] = 53;
    colvals[3875] = 55;
    colvals[3876] = 75;
    colvals[3877] = 77;
    colvals[3878] = 86;
    colvals[3879] = 88;
    colvals[3880] = 93;
    colvals[3881] = 95;
    colvals[3882] = 97;
    colvals[3883] = 100;
    colvals[3884] = 103;
    colvals[3885] = 107;
    colvals[3886] = 112;
    colvals[3887] = 114;
    colvals[3888] = 128;
    colvals[3889] = 131;
    colvals[3890] = 138;
    colvals[3891] = 154;
    colvals[3892] = 156;
    colvals[3893] = 157;
    colvals[3894] = 158;
    colvals[3895] = 159;
    colvals[3896] = 161;
    colvals[3897] = 162;
    colvals[3898] = 164;
    colvals[3899] = 166;
    colvals[3900] = 169;
    colvals[3901] = 172;
    colvals[3902] = 176;
    colvals[3903] = 181;
    colvals[3904] = 183;
    colvals[3905] = 34;
    colvals[3906] = 51;
    colvals[3907] = 53;
    colvals[3908] = 55;
    colvals[3909] = 75;
    colvals[3910] = 77;
    colvals[3911] = 83;
    colvals[3912] = 88;
    colvals[3913] = 93;
    colvals[3914] = 95;
    colvals[3915] = 97;
    colvals[3916] = 100;
    colvals[3917] = 104;
    colvals[3918] = 107;
    colvals[3919] = 112;
    colvals[3920] = 114;
    colvals[3921] = 119;
    colvals[3922] = 128;
    colvals[3923] = 131;
    colvals[3924] = 138;
    colvals[3925] = 140;
    colvals[3926] = 143;
    colvals[3927] = 154;
    colvals[3928] = 156;
    colvals[3929] = 157;
    colvals[3930] = 158;
    colvals[3931] = 159;
    colvals[3932] = 160;
    colvals[3933] = 161;
    colvals[3934] = 162;
    colvals[3935] = 163;
    colvals[3936] = 164;
    colvals[3937] = 166;
    colvals[3938] = 169;
    colvals[3939] = 171;
    colvals[3940] = 172;
    colvals[3941] = 174;
    colvals[3942] = 175;
    colvals[3943] = 176;
    colvals[3944] = 179;
    colvals[3945] = 180;
    colvals[3946] = 181;
    colvals[3947] = 183;
    colvals[3948] = 184;
    colvals[3949] = 185;
    colvals[3950] = 51;
    colvals[3951] = 53;
    colvals[3952] = 55;
    colvals[3953] = 75;
    colvals[3954] = 77;
    colvals[3955] = 88;
    colvals[3956] = 93;
    colvals[3957] = 94;
    colvals[3958] = 95;
    colvals[3959] = 100;
    colvals[3960] = 103;
    colvals[3961] = 107;
    colvals[3962] = 112;
    colvals[3963] = 114;
    colvals[3964] = 128;
    colvals[3965] = 131;
    colvals[3966] = 138;
    colvals[3967] = 154;
    colvals[3968] = 156;
    colvals[3969] = 157;
    colvals[3970] = 158;
    colvals[3971] = 159;
    colvals[3972] = 162;
    colvals[3973] = 163;
    colvals[3974] = 164;
    colvals[3975] = 166;
    colvals[3976] = 169;
    colvals[3977] = 174;
    colvals[3978] = 176;
    colvals[3979] = 181;
    colvals[3980] = 183;
    colvals[3981] = 185;
    colvals[3982] = 51;
    colvals[3983] = 53;
    colvals[3984] = 55;
    colvals[3985] = 75;
    colvals[3986] = 77;
    colvals[3987] = 83;
    colvals[3988] = 88;
    colvals[3989] = 95;
    colvals[3990] = 97;
    colvals[3991] = 100;
    colvals[3992] = 103;
    colvals[3993] = 104;
    colvals[3994] = 107;
    colvals[3995] = 112;
    colvals[3996] = 119;
    colvals[3997] = 128;
    colvals[3998] = 131;
    colvals[3999] = 140;
    colvals[4000] = 143;
    colvals[4001] = 156;
    colvals[4002] = 157;
    colvals[4003] = 158;
    colvals[4004] = 159;
    colvals[4005] = 160;
    colvals[4006] = 162;
    colvals[4007] = 163;
    colvals[4008] = 164;
    colvals[4009] = 165;
    colvals[4010] = 166;
    colvals[4011] = 169;
    colvals[4012] = 174;
    colvals[4013] = 179;
    colvals[4014] = 183;
    colvals[4015] = 184;
    colvals[4016] = 185;
    colvals[4017] = 35;
    colvals[4018] = 51;
    colvals[4019] = 52;
    colvals[4020] = 55;
    colvals[4021] = 57;
    colvals[4022] = 66;
    colvals[4023] = 93;
    colvals[4024] = 101;
    colvals[4025] = 128;
    colvals[4026] = 138;
    colvals[4027] = 140;
    colvals[4028] = 156;
    colvals[4029] = 157;
    colvals[4030] = 161;
    colvals[4031] = 183;
    colvals[4032] = 51;
    colvals[4033] = 52;
    colvals[4034] = 53;
    colvals[4035] = 54;
    colvals[4036] = 55;
    colvals[4037] = 59;
    colvals[4038] = 61;
    colvals[4039] = 75;
    colvals[4040] = 76;
    colvals[4041] = 77;
    colvals[4042] = 80;
    colvals[4043] = 81;
    colvals[4044] = 86;
    colvals[4045] = 88;
    colvals[4046] = 90;
    colvals[4047] = 93;
    colvals[4048] = 94;
    colvals[4049] = 95;
    colvals[4050] = 96;
    colvals[4051] = 100;
    colvals[4052] = 101;
    colvals[4053] = 102;
    colvals[4054] = 103;
    colvals[4055] = 104;
    colvals[4056] = 106;
    colvals[4057] = 107;
    colvals[4058] = 108;
    colvals[4059] = 112;
    colvals[4060] = 113;
    colvals[4061] = 114;
    colvals[4062] = 119;
    colvals[4063] = 128;
    colvals[4064] = 129;
    colvals[4065] = 131;
    colvals[4066] = 132;
    colvals[4067] = 138;
    colvals[4068] = 140;
    colvals[4069] = 141;
    colvals[4070] = 142;
    colvals[4071] = 143;
    colvals[4072] = 147;
    colvals[4073] = 156;
    colvals[4074] = 157;
    colvals[4075] = 158;
    colvals[4076] = 159;
    colvals[4077] = 160;
    colvals[4078] = 162;
    colvals[4079] = 163;
    colvals[4080] = 164;
    colvals[4081] = 165;
    colvals[4082] = 166;
    colvals[4083] = 167;
    colvals[4084] = 169;
    colvals[4085] = 171;
    colvals[4086] = 172;
    colvals[4087] = 174;
    colvals[4088] = 175;
    colvals[4089] = 176;
    colvals[4090] = 177;
    colvals[4091] = 178;
    colvals[4092] = 179;
    colvals[4093] = 180;
    colvals[4094] = 183;
    colvals[4095] = 184;
    colvals[4096] = 185;
    colvals[4097] = 186;
    colvals[4098] = 51;
    colvals[4099] = 53;
    colvals[4100] = 55;
    colvals[4101] = 75;
    colvals[4102] = 77;
    colvals[4103] = 93;
    colvals[4104] = 94;
    colvals[4105] = 95;
    colvals[4106] = 96;
    colvals[4107] = 97;
    colvals[4108] = 100;
    colvals[4109] = 102;
    colvals[4110] = 103;
    colvals[4111] = 104;
    colvals[4112] = 107;
    colvals[4113] = 112;
    colvals[4114] = 114;
    colvals[4115] = 128;
    colvals[4116] = 131;
    colvals[4117] = 138;
    colvals[4118] = 140;
    colvals[4119] = 156;
    colvals[4120] = 157;
    colvals[4121] = 158;
    colvals[4122] = 159;
    colvals[4123] = 162;
    colvals[4124] = 163;
    colvals[4125] = 164;
    colvals[4126] = 166;
    colvals[4127] = 169;
    colvals[4128] = 174;
    colvals[4129] = 175;
    colvals[4130] = 176;
    colvals[4131] = 183;
    colvals[4132] = 185;
    colvals[4133] = 51;
    colvals[4134] = 52;
    colvals[4135] = 54;
    colvals[4136] = 56;
    colvals[4137] = 58;
    colvals[4138] = 75;
    colvals[4139] = 76;
    colvals[4140] = 77;
    colvals[4141] = 80;
    colvals[4142] = 88;
    colvals[4143] = 90;
    colvals[4144] = 93;
    colvals[4145] = 94;
    colvals[4146] = 96;
    colvals[4147] = 100;
    colvals[4148] = 101;
    colvals[4149] = 102;
    colvals[4150] = 103;
    colvals[4151] = 104;
    colvals[4152] = 107;
    colvals[4153] = 108;
    colvals[4154] = 112;
    colvals[4155] = 113;
    colvals[4156] = 114;
    colvals[4157] = 119;
    colvals[4158] = 120;
    colvals[4159] = 122;
    colvals[4160] = 128;
    colvals[4161] = 129;
    colvals[4162] = 130;
    colvals[4163] = 131;
    colvals[4164] = 132;
    colvals[4165] = 138;
    colvals[4166] = 142;
    colvals[4167] = 143;
    colvals[4168] = 157;
    colvals[4169] = 159;
    colvals[4170] = 160;
    colvals[4171] = 162;
    colvals[4172] = 163;
    colvals[4173] = 164;
    colvals[4174] = 165;
    colvals[4175] = 166;
    colvals[4176] = 167;
    colvals[4177] = 168;
    colvals[4178] = 169;
    colvals[4179] = 174;
    colvals[4180] = 175;
    colvals[4181] = 177;
    colvals[4182] = 179;
    colvals[4183] = 183;
    colvals[4184] = 184;
    colvals[4185] = 185;
    colvals[4186] = 53;
    colvals[4187] = 54;
    colvals[4188] = 55;
    colvals[4189] = 75;
    colvals[4190] = 77;
    colvals[4191] = 90;
    colvals[4192] = 94;
    colvals[4193] = 96;
    colvals[4194] = 100;
    colvals[4195] = 102;
    colvals[4196] = 103;
    colvals[4197] = 104;
    colvals[4198] = 107;
    colvals[4199] = 112;
    colvals[4200] = 113;
    colvals[4201] = 114;
    colvals[4202] = 119;
    colvals[4203] = 128;
    colvals[4204] = 129;
    colvals[4205] = 131;
    colvals[4206] = 132;
    colvals[4207] = 138;
    colvals[4208] = 140;
    colvals[4209] = 141;
    colvals[4210] = 143;
    colvals[4211] = 156;
    colvals[4212] = 157;
    colvals[4213] = 159;
    colvals[4214] = 160;
    colvals[4215] = 162;
    colvals[4216] = 163;
    colvals[4217] = 164;
    colvals[4218] = 165;
    colvals[4219] = 166;
    colvals[4220] = 169;
    colvals[4221] = 174;
    colvals[4222] = 175;
    colvals[4223] = 176;
    colvals[4224] = 177;
    colvals[4225] = 179;
    colvals[4226] = 184;
    colvals[4227] = 185;
    colvals[4228] = 36;
    colvals[4229] = 52;
    colvals[4230] = 56;
    colvals[4231] = 58;
    colvals[4232] = 63;
    colvals[4233] = 65;
    colvals[4234] = 76;
    colvals[4235] = 78;
    colvals[4236] = 80;
    colvals[4237] = 81;
    colvals[4238] = 87;
    colvals[4239] = 88;
    colvals[4240] = 89;
    colvals[4241] = 90;
    colvals[4242] = 93;
    colvals[4243] = 96;
    colvals[4244] = 100;
    colvals[4245] = 101;
    colvals[4246] = 102;
    colvals[4247] = 103;
    colvals[4248] = 104;
    colvals[4249] = 108;
    colvals[4250] = 113;
    colvals[4251] = 114;
    colvals[4252] = 115;
    colvals[4253] = 119;
    colvals[4254] = 120;
    colvals[4255] = 122;
    colvals[4256] = 123;
    colvals[4257] = 129;
    colvals[4258] = 130;
    colvals[4259] = 131;
    colvals[4260] = 132;
    colvals[4261] = 133;
    colvals[4262] = 136;
    colvals[4263] = 138;
    colvals[4264] = 143;
    colvals[4265] = 148;
    colvals[4266] = 151;
    colvals[4267] = 153;
    colvals[4268] = 154;
    colvals[4269] = 157;
    colvals[4270] = 159;
    colvals[4271] = 160;
    colvals[4272] = 162;
    colvals[4273] = 163;
    colvals[4274] = 164;
    colvals[4275] = 165;
    colvals[4276] = 166;
    colvals[4277] = 167;
    colvals[4278] = 168;
    colvals[4279] = 169;
    colvals[4280] = 174;
    colvals[4281] = 175;
    colvals[4282] = 177;
    colvals[4283] = 179;
    colvals[4284] = 182;
    colvals[4285] = 183;
    colvals[4286] = 184;
    colvals[4287] = 186;
    colvals[4288] = 189;
    colvals[4289] = 198;
    colvals[4290] = 208;
    colvals[4291] = 212;
    colvals[4292] = 52;
    colvals[4293] = 53;
    colvals[4294] = 56;
    colvals[4295] = 58;
    colvals[4296] = 75;
    colvals[4297] = 76;
    colvals[4298] = 77;
    colvals[4299] = 88;
    colvals[4300] = 89;
    colvals[4301] = 90;
    colvals[4302] = 96;
    colvals[4303] = 100;
    colvals[4304] = 102;
    colvals[4305] = 103;
    colvals[4306] = 104;
    colvals[4307] = 107;
    colvals[4308] = 108;
    colvals[4309] = 112;
    colvals[4310] = 113;
    colvals[4311] = 114;
    colvals[4312] = 115;
    colvals[4313] = 119;
    colvals[4314] = 120;
    colvals[4315] = 122;
    colvals[4316] = 129;
    colvals[4317] = 130;
    colvals[4318] = 131;
    colvals[4319] = 132;
    colvals[4320] = 138;
    colvals[4321] = 143;
    colvals[4322] = 148;
    colvals[4323] = 154;
    colvals[4324] = 157;
    colvals[4325] = 159;
    colvals[4326] = 160;
    colvals[4327] = 162;
    colvals[4328] = 163;
    colvals[4329] = 164;
    colvals[4330] = 165;
    colvals[4331] = 166;
    colvals[4332] = 167;
    colvals[4333] = 169;
    colvals[4334] = 174;
    colvals[4335] = 175;
    colvals[4336] = 177;
    colvals[4337] = 179;
    colvals[4338] = 182;
    colvals[4339] = 183;
    colvals[4340] = 184;
    colvals[4341] = 186;
    colvals[4342] = 189;
    colvals[4343] = 212;
    colvals[4344] = 56;
    colvals[4345] = 58;
    colvals[4346] = 63;
    colvals[4347] = 75;
    colvals[4348] = 76;
    colvals[4349] = 78;
    colvals[4350] = 81;
    colvals[4351] = 87;
    colvals[4352] = 88;
    colvals[4353] = 89;
    colvals[4354] = 90;
    colvals[4355] = 100;
    colvals[4356] = 103;
    colvals[4357] = 107;
    colvals[4358] = 108;
    colvals[4359] = 112;
    colvals[4360] = 113;
    colvals[4361] = 114;
    colvals[4362] = 115;
    colvals[4363] = 119;
    colvals[4364] = 120;
    colvals[4365] = 122;
    colvals[4366] = 123;
    colvals[4367] = 130;
    colvals[4368] = 131;
    colvals[4369] = 132;
    colvals[4370] = 133;
    colvals[4371] = 136;
    colvals[4372] = 143;
    colvals[4373] = 148;
    colvals[4374] = 151;
    colvals[4375] = 153;
    colvals[4376] = 160;
    colvals[4377] = 162;
    colvals[4378] = 163;
    colvals[4379] = 164;
    colvals[4380] = 165;
    colvals[4381] = 166;
    colvals[4382] = 167;
    colvals[4383] = 168;
    colvals[4384] = 179;
    colvals[4385] = 183;
    colvals[4386] = 184;
    colvals[4387] = 198;
    colvals[4388] = 208;
    colvals[4389] = 37;
    colvals[4390] = 51;
    colvals[4391] = 52;
    colvals[4392] = 53;
    colvals[4393] = 54;
    colvals[4394] = 55;
    colvals[4395] = 56;
    colvals[4396] = 57;
    colvals[4397] = 58;
    colvals[4398] = 75;
    colvals[4399] = 76;
    colvals[4400] = 77;
    colvals[4401] = 78;
    colvals[4402] = 80;
    colvals[4403] = 81;
    colvals[4404] = 86;
    colvals[4405] = 88;
    colvals[4406] = 93;
    colvals[4407] = 94;
    colvals[4408] = 95;
    colvals[4409] = 96;
    colvals[4410] = 97;
    colvals[4411] = 100;
    colvals[4412] = 101;
    colvals[4413] = 102;
    colvals[4414] = 104;
    colvals[4415] = 107;
    colvals[4416] = 108;
    colvals[4417] = 111;
    colvals[4418] = 112;
    colvals[4419] = 113;
    colvals[4420] = 115;
    colvals[4421] = 119;
    colvals[4422] = 128;
    colvals[4423] = 129;
    colvals[4424] = 131;
    colvals[4425] = 138;
    colvals[4426] = 140;
    colvals[4427] = 142;
    colvals[4428] = 143;
    colvals[4429] = 148;
    colvals[4430] = 154;
    colvals[4431] = 156;
    colvals[4432] = 157;
    colvals[4433] = 158;
    colvals[4434] = 159;
    colvals[4435] = 162;
    colvals[4436] = 163;
    colvals[4437] = 164;
    colvals[4438] = 165;
    colvals[4439] = 166;
    colvals[4440] = 167;
    colvals[4441] = 169;
    colvals[4442] = 170;
    colvals[4443] = 171;
    colvals[4444] = 172;
    colvals[4445] = 174;
    colvals[4446] = 176;
    colvals[4447] = 177;
    colvals[4448] = 179;
    colvals[4449] = 180;
    colvals[4450] = 181;
    colvals[4451] = 183;
    colvals[4452] = 184;
    colvals[4453] = 185;
    colvals[4454] = 186;
    colvals[4455] = 188;
    colvals[4456] = 189;
    colvals[4457] = 207;
    colvals[4458] = 211;
    colvals[4459] = 52;
    colvals[4460] = 54;
    colvals[4461] = 56;
    colvals[4462] = 58;
    colvals[4463] = 76;
    colvals[4464] = 78;
    colvals[4465] = 81;
    colvals[4466] = 86;
    colvals[4467] = 93;
    colvals[4468] = 94;
    colvals[4469] = 95;
    colvals[4470] = 96;
    colvals[4471] = 97;
    colvals[4472] = 100;
    colvals[4473] = 102;
    colvals[4474] = 104;
    colvals[4475] = 107;
    colvals[4476] = 108;
    colvals[4477] = 113;
    colvals[4478] = 115;
    colvals[4479] = 119;
    colvals[4480] = 128;
    colvals[4481] = 129;
    colvals[4482] = 138;
    colvals[4483] = 142;
    colvals[4484] = 143;
    colvals[4485] = 148;
    colvals[4486] = 154;
    colvals[4487] = 156;
    colvals[4488] = 157;
    colvals[4489] = 158;
    colvals[4490] = 159;
    colvals[4491] = 162;
    colvals[4492] = 163;
    colvals[4493] = 165;
    colvals[4494] = 167;
    colvals[4495] = 169;
    colvals[4496] = 170;
    colvals[4497] = 171;
    colvals[4498] = 173;
    colvals[4499] = 174;
    colvals[4500] = 175;
    colvals[4501] = 176;
    colvals[4502] = 177;
    colvals[4503] = 184;
    colvals[4504] = 186;
    colvals[4505] = 188;
    colvals[4506] = 189;
    colvals[4507] = 207;
    colvals[4508] = 38;
    colvals[4509] = 77;
    colvals[4510] = 80;
    colvals[4511] = 93;
    colvals[4512] = 95;
    colvals[4513] = 101;
    colvals[4514] = 102;
    colvals[4515] = 119;
    colvals[4516] = 142;
    colvals[4517] = 156;
    colvals[4518] = 162;
    colvals[4519] = 169;
    colvals[4520] = 171;
    colvals[4521] = 174;
    colvals[4522] = 175;
    colvals[4523] = 176;
    colvals[4524] = 180;
    colvals[4525] = 183;
    colvals[4526] = 39;
    colvals[4527] = 51;
    colvals[4528] = 52;
    colvals[4529] = 93;
    colvals[4530] = 100;
    colvals[4531] = 101;
    colvals[4532] = 102;
    colvals[4533] = 119;
    colvals[4534] = 132;
    colvals[4535] = 138;
    colvals[4536] = 144;
    colvals[4537] = 147;
    colvals[4538] = 156;
    colvals[4539] = 162;
    colvals[4540] = 169;
    colvals[4541] = 172;
    colvals[4542] = 174;
    colvals[4543] = 185;
    colvals[4544] = 187;
    colvals[4545] = 211;
    colvals[4546] = 52;
    colvals[4547] = 100;
    colvals[4548] = 102;
    colvals[4549] = 115;
    colvals[4550] = 148;
    colvals[4551] = 156;
    colvals[4552] = 162;
    colvals[4553] = 163;
    colvals[4554] = 172;
    colvals[4555] = 173;
    colvals[4556] = 174;
    colvals[4557] = 185;
    colvals[4558] = 186;
    colvals[4559] = 212;
    colvals[4560] = 51;
    colvals[4561] = 52;
    colvals[4562] = 53;
    colvals[4563] = 54;
    colvals[4564] = 55;
    colvals[4565] = 56;
    colvals[4566] = 57;
    colvals[4567] = 58;
    colvals[4568] = 59;
    colvals[4569] = 60;
    colvals[4570] = 61;
    colvals[4571] = 64;
    colvals[4572] = 71;
    colvals[4573] = 73;
    colvals[4574] = 75;
    colvals[4575] = 76;
    colvals[4576] = 77;
    colvals[4577] = 78;
    colvals[4578] = 80;
    colvals[4579] = 81;
    colvals[4580] = 88;
    colvals[4581] = 89;
    colvals[4582] = 90;
    colvals[4583] = 93;
    colvals[4584] = 94;
    colvals[4585] = 95;
    colvals[4586] = 96;
    colvals[4587] = 97;
    colvals[4588] = 98;
    colvals[4589] = 99;
    colvals[4590] = 100;
    colvals[4591] = 101;
    colvals[4592] = 102;
    colvals[4593] = 103;
    colvals[4594] = 104;
    colvals[4595] = 106;
    colvals[4596] = 107;
    colvals[4597] = 108;
    colvals[4598] = 112;
    colvals[4599] = 113;
    colvals[4600] = 114;
    colvals[4601] = 115;
    colvals[4602] = 119;
    colvals[4603] = 122;
    colvals[4604] = 128;
    colvals[4605] = 131;
    colvals[4606] = 133;
    colvals[4607] = 135;
    colvals[4608] = 136;
    colvals[4609] = 138;
    colvals[4610] = 140;
    colvals[4611] = 142;
    colvals[4612] = 147;
    colvals[4613] = 148;
    colvals[4614] = 153;
    colvals[4615] = 156;
    colvals[4616] = 157;
    colvals[4617] = 158;
    colvals[4618] = 159;
    colvals[4619] = 160;
    colvals[4620] = 162;
    colvals[4621] = 163;
    colvals[4622] = 164;
    colvals[4623] = 165;
    colvals[4624] = 166;
    colvals[4625] = 167;
    colvals[4626] = 169;
    colvals[4627] = 170;
    colvals[4628] = 171;
    colvals[4629] = 172;
    colvals[4630] = 173;
    colvals[4631] = 174;
    colvals[4632] = 175;
    colvals[4633] = 176;
    colvals[4634] = 177;
    colvals[4635] = 178;
    colvals[4636] = 179;
    colvals[4637] = 180;
    colvals[4638] = 181;
    colvals[4639] = 182;
    colvals[4640] = 183;
    colvals[4641] = 184;
    colvals[4642] = 185;
    colvals[4643] = 186;
    colvals[4644] = 187;
    colvals[4645] = 189;
    colvals[4646] = 190;
    colvals[4647] = 191;
    colvals[4648] = 192;
    colvals[4649] = 193;
    colvals[4650] = 195;
    colvals[4651] = 197;
    colvals[4652] = 198;
    colvals[4653] = 199;
    colvals[4654] = 200;
    colvals[4655] = 201;
    colvals[4656] = 202;
    colvals[4657] = 203;
    colvals[4658] = 206;
    colvals[4659] = 207;
    colvals[4660] = 211;
    colvals[4661] = 212;
    colvals[4662] = 213;
    colvals[4663] = 214;
    colvals[4664] = 51;
    colvals[4665] = 52;
    colvals[4666] = 53;
    colvals[4667] = 55;
    colvals[4668] = 57;
    colvals[4669] = 60;
    colvals[4670] = 75;
    colvals[4671] = 76;
    colvals[4672] = 77;
    colvals[4673] = 86;
    colvals[4674] = 88;
    colvals[4675] = 93;
    colvals[4676] = 94;
    colvals[4677] = 95;
    colvals[4678] = 96;
    colvals[4679] = 97;
    colvals[4680] = 100;
    colvals[4681] = 101;
    colvals[4682] = 102;
    colvals[4683] = 103;
    colvals[4684] = 107;
    colvals[4685] = 112;
    colvals[4686] = 114;
    colvals[4687] = 128;
    colvals[4688] = 131;
    colvals[4689] = 138;
    colvals[4690] = 157;
    colvals[4691] = 158;
    colvals[4692] = 159;
    colvals[4693] = 162;
    colvals[4694] = 164;
    colvals[4695] = 166;
    colvals[4696] = 169;
    colvals[4697] = 171;
    colvals[4698] = 174;
    colvals[4699] = 175;
    colvals[4700] = 176;
    colvals[4701] = 177;
    colvals[4702] = 180;
    colvals[4703] = 181;
    colvals[4704] = 183;
    colvals[4705] = 184;
    colvals[4706] = 206;
    colvals[4707] = 211;
    colvals[4708] = 213;
    colvals[4709] = 40;
    colvals[4710] = 51;
    colvals[4711] = 52;
    colvals[4712] = 53;
    colvals[4713] = 54;
    colvals[4714] = 55;
    colvals[4715] = 57;
    colvals[4716] = 59;
    colvals[4717] = 75;
    colvals[4718] = 76;
    colvals[4719] = 77;
    colvals[4720] = 78;
    colvals[4721] = 80;
    colvals[4722] = 81;
    colvals[4723] = 86;
    colvals[4724] = 88;
    colvals[4725] = 89;
    colvals[4726] = 92;
    colvals[4727] = 93;
    colvals[4728] = 94;
    colvals[4729] = 95;
    colvals[4730] = 96;
    colvals[4731] = 97;
    colvals[4732] = 99;
    colvals[4733] = 100;
    colvals[4734] = 101;
    colvals[4735] = 102;
    colvals[4736] = 103;
    colvals[4737] = 104;
    colvals[4738] = 107;
    colvals[4739] = 108;
    colvals[4740] = 112;
    colvals[4741] = 113;
    colvals[4742] = 114;
    colvals[4743] = 119;
    colvals[4744] = 128;
    colvals[4745] = 129;
    colvals[4746] = 131;
    colvals[4747] = 133;
    colvals[4748] = 138;
    colvals[4749] = 140;
    colvals[4750] = 142;
    colvals[4751] = 154;
    colvals[4752] = 156;
    colvals[4753] = 157;
    colvals[4754] = 158;
    colvals[4755] = 159;
    colvals[4756] = 162;
    colvals[4757] = 163;
    colvals[4758] = 164;
    colvals[4759] = 165;
    colvals[4760] = 166;
    colvals[4761] = 169;
    colvals[4762] = 171;
    colvals[4763] = 174;
    colvals[4764] = 175;
    colvals[4765] = 176;
    colvals[4766] = 177;
    colvals[4767] = 178;
    colvals[4768] = 179;
    colvals[4769] = 180;
    colvals[4770] = 183;
    colvals[4771] = 184;
    colvals[4772] = 185;
    colvals[4773] = 186;
    colvals[4774] = 189;
    colvals[4775] = 200;
    colvals[4776] = 207;
    colvals[4777] = 210;
    colvals[4778] = 211;
    colvals[4779] = 213;
    colvals[4780] = 214;
    colvals[4781] = 51;
    colvals[4782] = 53;
    colvals[4783] = 57;
    colvals[4784] = 75;
    colvals[4785] = 77;
    colvals[4786] = 86;
    colvals[4787] = 89;
    colvals[4788] = 92;
    colvals[4789] = 94;
    colvals[4790] = 96;
    colvals[4791] = 97;
    colvals[4792] = 100;
    colvals[4793] = 102;
    colvals[4794] = 104;
    colvals[4795] = 107;
    colvals[4796] = 113;
    colvals[4797] = 114;
    colvals[4798] = 129;
    colvals[4799] = 131;
    colvals[4800] = 138;
    colvals[4801] = 154;
    colvals[4802] = 156;
    colvals[4803] = 157;
    colvals[4804] = 159;
    colvals[4805] = 162;
    colvals[4806] = 163;
    colvals[4807] = 164;
    colvals[4808] = 166;
    colvals[4809] = 169;
    colvals[4810] = 174;
    colvals[4811] = 175;
    colvals[4812] = 176;
    colvals[4813] = 177;
    colvals[4814] = 183;
    colvals[4815] = 184;
    colvals[4816] = 185;
    colvals[4817] = 189;
    colvals[4818] = 214;
    colvals[4819] = 41;
    colvals[4820] = 59;
    colvals[4821] = 75;
    colvals[4822] = 80;
    colvals[4823] = 88;
    colvals[4824] = 95;
    colvals[4825] = 101;
    colvals[4826] = 103;
    colvals[4827] = 108;
    colvals[4828] = 131;
    colvals[4829] = 156;
    colvals[4830] = 174;
    colvals[4831] = 176;
    colvals[4832] = 178;
    colvals[4833] = 183;
    colvals[4834] = 51;
    colvals[4835] = 53;
    colvals[4836] = 55;
    colvals[4837] = 75;
    colvals[4838] = 77;
    colvals[4839] = 93;
    colvals[4840] = 95;
    colvals[4841] = 97;
    colvals[4842] = 100;
    colvals[4843] = 103;
    colvals[4844] = 104;
    colvals[4845] = 107;
    colvals[4846] = 112;
    colvals[4847] = 119;
    colvals[4848] = 128;
    colvals[4849] = 131;
    colvals[4850] = 140;
    colvals[4851] = 158;
    colvals[4852] = 162;
    colvals[4853] = 163;
    colvals[4854] = 164;
    colvals[4855] = 166;
    colvals[4856] = 169;
    colvals[4857] = 174;
    colvals[4858] = 176;
    colvals[4859] = 177;
    colvals[4860] = 179;
    colvals[4861] = 183;
    colvals[4862] = 185;
    colvals[4863] = 51;
    colvals[4864] = 52;
    colvals[4865] = 75;
    colvals[4866] = 93;
    colvals[4867] = 101;
    colvals[4868] = 106;
    colvals[4869] = 128;
    colvals[4870] = 131;
    colvals[4871] = 138;
    colvals[4872] = 156;
    colvals[4873] = 161;
    colvals[4874] = 169;
    colvals[4875] = 171;
    colvals[4876] = 174;
    colvals[4877] = 176;
    colvals[4878] = 180;
    colvals[4879] = 183;
    colvals[4880] = 42;
    colvals[4881] = 51;
    colvals[4882] = 52;
    colvals[4883] = 75;
    colvals[4884] = 76;
    colvals[4885] = 78;
    colvals[4886] = 81;
    colvals[4887] = 89;
    colvals[4888] = 98;
    colvals[4889] = 100;
    colvals[4890] = 101;
    colvals[4891] = 102;
    colvals[4892] = 112;
    colvals[4893] = 113;
    colvals[4894] = 114;
    colvals[4895] = 119;
    colvals[4896] = 131;
    colvals[4897] = 132;
    colvals[4898] = 135;
    colvals[4899] = 138;
    colvals[4900] = 146;
    colvals[4901] = 157;
    colvals[4902] = 159;
    colvals[4903] = 166;
    colvals[4904] = 174;
    colvals[4905] = 175;
    colvals[4906] = 181;
    colvals[4907] = 182;
    colvals[4908] = 183;
    colvals[4909] = 185;
    colvals[4910] = 186;
    colvals[4911] = 190;
    colvals[4912] = 211;
    colvals[4913] = 212;
    colvals[4914] = 52;
    colvals[4915] = 89;
    colvals[4916] = 99;
    colvals[4917] = 100;
    colvals[4918] = 102;
    colvals[4919] = 113;
    colvals[4920] = 114;
    colvals[4921] = 136;
    colvals[4922] = 157;
    colvals[4923] = 159;
    colvals[4924] = 166;
    colvals[4925] = 174;
    colvals[4926] = 175;
    colvals[4927] = 176;
    colvals[4928] = 181;
    colvals[4929] = 182;
    colvals[4930] = 51;
    colvals[4931] = 52;
    colvals[4932] = 53;
    colvals[4933] = 54;
    colvals[4934] = 55;
    colvals[4935] = 57;
    colvals[4936] = 59;
    colvals[4937] = 60;
    colvals[4938] = 61;
    colvals[4939] = 62;
    colvals[4940] = 63;
    colvals[4941] = 65;
    colvals[4942] = 75;
    colvals[4943] = 76;
    colvals[4944] = 77;
    colvals[4945] = 78;
    colvals[4946] = 80;
    colvals[4947] = 81;
    colvals[4948] = 86;
    colvals[4949] = 87;
    colvals[4950] = 88;
    colvals[4951] = 89;
    colvals[4952] = 90;
    colvals[4953] = 93;
    colvals[4954] = 94;
    colvals[4955] = 95;
    colvals[4956] = 96;
    colvals[4957] = 97;
    colvals[4958] = 98;
    colvals[4959] = 100;
    colvals[4960] = 101;
    colvals[4961] = 102;
    colvals[4962] = 103;
    colvals[4963] = 104;
    colvals[4964] = 107;
    colvals[4965] = 112;
    colvals[4966] = 113;
    colvals[4967] = 114;
    colvals[4968] = 115;
    colvals[4969] = 119;
    colvals[4970] = 120;
    colvals[4971] = 122;
    colvals[4972] = 128;
    colvals[4973] = 129;
    colvals[4974] = 131;
    colvals[4975] = 132;
    colvals[4976] = 133;
    colvals[4977] = 138;
    colvals[4978] = 140;
    colvals[4979] = 142;
    colvals[4980] = 143;
    colvals[4981] = 146;
    colvals[4982] = 147;
    colvals[4983] = 148;
    colvals[4984] = 153;
    colvals[4985] = 156;
    colvals[4986] = 157;
    colvals[4987] = 159;
    colvals[4988] = 160;
    colvals[4989] = 161;
    colvals[4990] = 162;
    colvals[4991] = 163;
    colvals[4992] = 164;
    colvals[4993] = 165;
    colvals[4994] = 166;
    colvals[4995] = 167;
    colvals[4996] = 169;
    colvals[4997] = 171;
    colvals[4998] = 174;
    colvals[4999] = 175;
    colvals[5000] = 176;
    colvals[5001] = 178;
    colvals[5002] = 179;
    colvals[5003] = 180;
    colvals[5004] = 183;
    colvals[5005] = 184;
    colvals[5006] = 185;
    colvals[5007] = 186;
    colvals[5008] = 189;
    colvals[5009] = 190;
    colvals[5010] = 200;
    colvals[5011] = 203;
    colvals[5012] = 208;
    colvals[5013] = 211;
    colvals[5014] = 213;
    colvals[5015] = 214;
    colvals[5016] = 51;
    colvals[5017] = 53;
    colvals[5018] = 54;
    colvals[5019] = 55;
    colvals[5020] = 75;
    colvals[5021] = 77;
    colvals[5022] = 86;
    colvals[5023] = 88;
    colvals[5024] = 93;
    colvals[5025] = 94;
    colvals[5026] = 95;
    colvals[5027] = 96;
    colvals[5028] = 97;
    colvals[5029] = 100;
    colvals[5030] = 102;
    colvals[5031] = 103;
    colvals[5032] = 104;
    colvals[5033] = 107;
    colvals[5034] = 112;
    colvals[5035] = 113;
    colvals[5036] = 114;
    colvals[5037] = 119;
    colvals[5038] = 128;
    colvals[5039] = 131;
    colvals[5040] = 138;
    colvals[5041] = 140;
    colvals[5042] = 156;
    colvals[5043] = 157;
    colvals[5044] = 158;
    colvals[5045] = 159;
    colvals[5046] = 160;
    colvals[5047] = 162;
    colvals[5048] = 163;
    colvals[5049] = 164;
    colvals[5050] = 166;
    colvals[5051] = 169;
    colvals[5052] = 174;
    colvals[5053] = 175;
    colvals[5054] = 176;
    colvals[5055] = 179;
    colvals[5056] = 183;
    colvals[5057] = 184;
    colvals[5058] = 185;
    colvals[5059] = 189;
    colvals[5060] = 197;
    colvals[5061] = 206;
    colvals[5062] = 51;
    colvals[5063] = 52;
    colvals[5064] = 53;
    colvals[5065] = 54;
    colvals[5066] = 56;
    colvals[5067] = 75;
    colvals[5068] = 76;
    colvals[5069] = 77;
    colvals[5070] = 78;
    colvals[5071] = 80;
    colvals[5072] = 81;
    colvals[5073] = 88;
    colvals[5074] = 90;
    colvals[5075] = 93;
    colvals[5076] = 94;
    colvals[5077] = 96;
    colvals[5078] = 98;
    colvals[5079] = 99;
    colvals[5080] = 100;
    colvals[5081] = 101;
    colvals[5082] = 102;
    colvals[5083] = 103;
    colvals[5084] = 108;
    colvals[5085] = 109;
    colvals[5086] = 112;
    colvals[5087] = 113;
    colvals[5088] = 114;
    colvals[5089] = 115;
    colvals[5090] = 119;
    colvals[5091] = 123;
    colvals[5092] = 128;
    colvals[5093] = 129;
    colvals[5094] = 131;
    colvals[5095] = 132;
    colvals[5096] = 135;
    colvals[5097] = 136;
    colvals[5098] = 138;
    colvals[5099] = 140;
    colvals[5100] = 143;
    colvals[5101] = 147;
    colvals[5102] = 148;
    colvals[5103] = 149;
    colvals[5104] = 150;
    colvals[5105] = 154;
    colvals[5106] = 156;
    colvals[5107] = 159;
    colvals[5108] = 160;
    colvals[5109] = 162;
    colvals[5110] = 163;
    colvals[5111] = 165;
    colvals[5112] = 166;
    colvals[5113] = 169;
    colvals[5114] = 172;
    colvals[5115] = 173;
    colvals[5116] = 174;
    colvals[5117] = 176;
    colvals[5118] = 177;
    colvals[5119] = 179;
    colvals[5120] = 181;
    colvals[5121] = 182;
    colvals[5122] = 183;
    colvals[5123] = 184;
    colvals[5124] = 185;
    colvals[5125] = 186;
    colvals[5126] = 187;
    colvals[5127] = 188;
    colvals[5128] = 189;
    colvals[5129] = 191;
    colvals[5130] = 197;
    colvals[5131] = 200;
    colvals[5132] = 207;
    colvals[5133] = 209;
    colvals[5134] = 210;
    colvals[5135] = 211;
    colvals[5136] = 212;
    colvals[5137] = 213;
    colvals[5138] = 214;
    colvals[5139] = 51;
    colvals[5140] = 52;
    colvals[5141] = 53;
    colvals[5142] = 54;
    colvals[5143] = 56;
    colvals[5144] = 60;
    colvals[5145] = 75;
    colvals[5146] = 76;
    colvals[5147] = 77;
    colvals[5148] = 80;
    colvals[5149] = 88;
    colvals[5150] = 94;
    colvals[5151] = 96;
    colvals[5152] = 98;
    colvals[5153] = 99;
    colvals[5154] = 100;
    colvals[5155] = 101;
    colvals[5156] = 102;
    colvals[5157] = 103;
    colvals[5158] = 104;
    colvals[5159] = 107;
    colvals[5160] = 108;
    colvals[5161] = 109;
    colvals[5162] = 113;
    colvals[5163] = 114;
    colvals[5164] = 115;
    colvals[5165] = 129;
    colvals[5166] = 131;
    colvals[5167] = 138;
    colvals[5168] = 147;
    colvals[5169] = 148;
    colvals[5170] = 149;
    colvals[5171] = 154;
    colvals[5172] = 157;
    colvals[5173] = 159;
    colvals[5174] = 162;
    colvals[5175] = 163;
    colvals[5176] = 165;
    colvals[5177] = 166;
    colvals[5178] = 169;
    colvals[5179] = 172;
    colvals[5180] = 174;
    colvals[5181] = 175;
    colvals[5182] = 176;
    colvals[5183] = 177;
    colvals[5184] = 181;
    colvals[5185] = 183;
    colvals[5186] = 184;
    colvals[5187] = 185;
    colvals[5188] = 186;
    colvals[5189] = 187;
    colvals[5190] = 189;
    colvals[5191] = 191;
    colvals[5192] = 197;
    colvals[5193] = 209;
    colvals[5194] = 211;
    colvals[5195] = 213;
    colvals[5196] = 51;
    colvals[5197] = 100;
    colvals[5198] = 101;
    colvals[5199] = 102;
    colvals[5200] = 114;
    colvals[5201] = 119;
    colvals[5202] = 122;
    colvals[5203] = 132;
    colvals[5204] = 138;
    colvals[5205] = 147;
    colvals[5206] = 150;
    colvals[5207] = 156;
    colvals[5208] = 169;
    colvals[5209] = 174;
    colvals[5210] = 185;
    colvals[5211] = 187;
    colvals[5212] = 188;
    colvals[5213] = 211;
    colvals[5214] = 86;
    colvals[5215] = 100;
    colvals[5216] = 102;
    colvals[5217] = 114;
    colvals[5218] = 138;
    colvals[5219] = 149;
    colvals[5220] = 169;
    colvals[5221] = 181;
    colvals[5222] = 186;
    colvals[5223] = 187;
    colvals[5224] = 188;
    colvals[5225] = 212;
    colvals[5226] = 52;
    colvals[5227] = 57;
    colvals[5228] = 58;
    colvals[5229] = 75;
    colvals[5230] = 76;
    colvals[5231] = 95;
    colvals[5232] = 97;
    colvals[5233] = 99;
    colvals[5234] = 100;
    colvals[5235] = 102;
    colvals[5236] = 108;
    colvals[5237] = 112;
    colvals[5238] = 113;
    colvals[5239] = 115;
    colvals[5240] = 119;
    colvals[5241] = 122;
    colvals[5242] = 132;
    colvals[5243] = 138;
    colvals[5244] = 148;
    colvals[5245] = 151;
    colvals[5246] = 154;
    colvals[5247] = 156;
    colvals[5248] = 166;
    colvals[5249] = 167;
    colvals[5250] = 169;
    colvals[5251] = 170;
    colvals[5252] = 174;
    colvals[5253] = 176;
    colvals[5254] = 177;
    colvals[5255] = 183;
    colvals[5256] = 184;
    colvals[5257] = 186;
    colvals[5258] = 189;
    colvals[5259] = 190;
    colvals[5260] = 191;
    colvals[5261] = 192;
    colvals[5262] = 193;
    colvals[5263] = 194;
    colvals[5264] = 197;
    colvals[5265] = 198;
    colvals[5266] = 200;
    colvals[5267] = 206;
    colvals[5268] = 207;
    colvals[5269] = 208;
    colvals[5270] = 209;
    colvals[5271] = 210;
    colvals[5272] = 51;
    colvals[5273] = 52;
    colvals[5274] = 55;
    colvals[5275] = 58;
    colvals[5276] = 62;
    colvals[5277] = 75;
    colvals[5278] = 76;
    colvals[5279] = 77;
    colvals[5280] = 86;
    colvals[5281] = 95;
    colvals[5282] = 99;
    colvals[5283] = 100;
    colvals[5284] = 101;
    colvals[5285] = 102;
    colvals[5286] = 103;
    colvals[5287] = 108;
    colvals[5288] = 112;
    colvals[5289] = 113;
    colvals[5290] = 115;
    colvals[5291] = 138;
    colvals[5292] = 148;
    colvals[5293] = 154;
    colvals[5294] = 156;
    colvals[5295] = 167;
    colvals[5296] = 170;
    colvals[5297] = 174;
    colvals[5298] = 177;
    colvals[5299] = 181;
    colvals[5300] = 183;
    colvals[5301] = 185;
    colvals[5302] = 186;
    colvals[5303] = 189;
    colvals[5304] = 190;
    colvals[5305] = 191;
    colvals[5306] = 192;
    colvals[5307] = 193;
    colvals[5308] = 197;
    colvals[5309] = 198;
    colvals[5310] = 199;
    colvals[5311] = 203;
    colvals[5312] = 206;
    colvals[5313] = 207;
    colvals[5314] = 209;
    colvals[5315] = 210;
    colvals[5316] = 43;
    colvals[5317] = 51;
    colvals[5318] = 52;
    colvals[5319] = 100;
    colvals[5320] = 102;
    colvals[5321] = 138;
    colvals[5322] = 156;
    colvals[5323] = 174;
    colvals[5324] = 186;
    colvals[5325] = 191;
    colvals[5326] = 193;
    colvals[5327] = 194;
    colvals[5328] = 195;
    colvals[5329] = 196;
    colvals[5330] = 197;
    colvals[5331] = 51;
    colvals[5332] = 52;
    colvals[5333] = 53;
    colvals[5334] = 75;
    colvals[5335] = 100;
    colvals[5336] = 102;
    colvals[5337] = 156;
    colvals[5338] = 174;
    colvals[5339] = 186;
    colvals[5340] = 190;
    colvals[5341] = 191;
    colvals[5342] = 192;
    colvals[5343] = 197;
    colvals[5344] = 198;
    colvals[5345] = 199;
    colvals[5346] = 207;
    colvals[5347] = 209;
    colvals[5348] = 44;
    colvals[5349] = 52;
    colvals[5350] = 57;
    colvals[5351] = 100;
    colvals[5352] = 102;
    colvals[5353] = 138;
    colvals[5354] = 174;
    colvals[5355] = 189;
    colvals[5356] = 193;
    colvals[5357] = 195;
    colvals[5358] = 196;
    colvals[5359] = 52;
    colvals[5360] = 55;
    colvals[5361] = 58;
    colvals[5362] = 100;
    colvals[5363] = 102;
    colvals[5364] = 138;
    colvals[5365] = 189;
    colvals[5366] = 190;
    colvals[5367] = 193;
    colvals[5368] = 194;
    colvals[5369] = 195;
    colvals[5370] = 45;
    colvals[5371] = 52;
    colvals[5372] = 102;
    colvals[5373] = 138;
    colvals[5374] = 174;
    colvals[5375] = 195;
    colvals[5376] = 52;
    colvals[5377] = 100;
    colvals[5378] = 102;
    colvals[5379] = 195;
    colvals[5380] = 196;
    colvals[5381] = 51;
    colvals[5382] = 52;
    colvals[5383] = 100;
    colvals[5384] = 102;
    colvals[5385] = 119;
    colvals[5386] = 122;
    colvals[5387] = 132;
    colvals[5388] = 138;
    colvals[5389] = 174;
    colvals[5390] = 184;
    colvals[5391] = 186;
    colvals[5392] = 197;
    colvals[5393] = 199;
    colvals[5394] = 200;
    colvals[5395] = 201;
    colvals[5396] = 202;
    colvals[5397] = 203;
    colvals[5398] = 51;
    colvals[5399] = 58;
    colvals[5400] = 75;
    colvals[5401] = 100;
    colvals[5402] = 101;
    colvals[5403] = 102;
    colvals[5404] = 103;
    colvals[5405] = 112;
    colvals[5406] = 119;
    colvals[5407] = 122;
    colvals[5408] = 132;
    colvals[5409] = 138;
    colvals[5410] = 166;
    colvals[5411] = 174;
    colvals[5412] = 184;
    colvals[5413] = 186;
    colvals[5414] = 189;
    colvals[5415] = 190;
    colvals[5416] = 197;
    colvals[5417] = 198;
    colvals[5418] = 199;
    colvals[5419] = 201;
    colvals[5420] = 203;
    colvals[5421] = 52;
    colvals[5422] = 100;
    colvals[5423] = 102;
    colvals[5424] = 119;
    colvals[5425] = 122;
    colvals[5426] = 132;
    colvals[5427] = 138;
    colvals[5428] = 174;
    colvals[5429] = 199;
    colvals[5430] = 201;
    colvals[5431] = 202;
    colvals[5432] = 203;
    colvals[5433] = 204;
    colvals[5434] = 52;
    colvals[5435] = 58;
    colvals[5436] = 100;
    colvals[5437] = 102;
    colvals[5438] = 103;
    colvals[5439] = 119;
    colvals[5440] = 122;
    colvals[5441] = 132;
    colvals[5442] = 138;
    colvals[5443] = 174;
    colvals[5444] = 176;
    colvals[5445] = 184;
    colvals[5446] = 185;
    colvals[5447] = 190;
    colvals[5448] = 197;
    colvals[5449] = 199;
    colvals[5450] = 200;
    colvals[5451] = 201;
    colvals[5452] = 203;
    colvals[5453] = 52;
    colvals[5454] = 93;
    colvals[5455] = 95;
    colvals[5456] = 100;
    colvals[5457] = 102;
    colvals[5458] = 112;
    colvals[5459] = 119;
    colvals[5460] = 138;
    colvals[5461] = 174;
    colvals[5462] = 201;
    colvals[5463] = 203;
    colvals[5464] = 204;
    colvals[5465] = 205;
    colvals[5466] = 52;
    colvals[5467] = 58;
    colvals[5468] = 81;
    colvals[5469] = 90;
    colvals[5470] = 100;
    colvals[5471] = 102;
    colvals[5472] = 103;
    colvals[5473] = 119;
    colvals[5474] = 122;
    colvals[5475] = 132;
    colvals[5476] = 174;
    colvals[5477] = 198;
    colvals[5478] = 199;
    colvals[5479] = 201;
    colvals[5480] = 202;
    colvals[5481] = 203;
    colvals[5482] = 46;
    colvals[5483] = 58;
    colvals[5484] = 81;
    colvals[5485] = 90;
    colvals[5486] = 93;
    colvals[5487] = 100;
    colvals[5488] = 102;
    colvals[5489] = 112;
    colvals[5490] = 119;
    colvals[5491] = 132;
    colvals[5492] = 138;
    colvals[5493] = 174;
    colvals[5494] = 203;
    colvals[5495] = 205;
    colvals[5496] = 95;
    colvals[5497] = 100;
    colvals[5498] = 102;
    colvals[5499] = 103;
    colvals[5500] = 112;
    colvals[5501] = 119;
    colvals[5502] = 201;
    colvals[5503] = 203;
    colvals[5504] = 204;
    colvals[5505] = 100;
    colvals[5506] = 103;
    colvals[5507] = 112;
    colvals[5508] = 119;
    colvals[5509] = 132;
    colvals[5510] = 202;
    colvals[5511] = 203;
    colvals[5512] = 204;
    colvals[5513] = 205;
    colvals[5514] = 47;
    colvals[5515] = 52;
    colvals[5516] = 95;
    colvals[5517] = 97;
    colvals[5518] = 100;
    colvals[5519] = 102;
    colvals[5520] = 118;
    colvals[5521] = 119;
    colvals[5522] = 122;
    colvals[5523] = 131;
    colvals[5524] = 132;
    colvals[5525] = 138;
    colvals[5526] = 154;
    colvals[5527] = 166;
    colvals[5528] = 169;
    colvals[5529] = 174;
    colvals[5530] = 176;
    colvals[5531] = 183;
    colvals[5532] = 184;
    colvals[5533] = 189;
    colvals[5534] = 191;
    colvals[5535] = 197;
    colvals[5536] = 199;
    colvals[5537] = 206;
    colvals[5538] = 207;
    colvals[5539] = 208;
    colvals[5540] = 210;
    colvals[5541] = 51;
    colvals[5542] = 53;
    colvals[5543] = 75;
    colvals[5544] = 77;
    colvals[5545] = 95;
    colvals[5546] = 100;
    colvals[5547] = 102;
    colvals[5548] = 103;
    colvals[5549] = 131;
    colvals[5550] = 154;
    colvals[5551] = 156;
    colvals[5552] = 169;
    colvals[5553] = 174;
    colvals[5554] = 176;
    colvals[5555] = 183;
    colvals[5556] = 185;
    colvals[5557] = 190;
    colvals[5558] = 192;
    colvals[5559] = 198;
    colvals[5560] = 206;
    colvals[5561] = 207;
    colvals[5562] = 210;
    colvals[5563] = 62;
    colvals[5564] = 86;
    colvals[5565] = 100;
    colvals[5566] = 102;
    colvals[5567] = 103;
    colvals[5568] = 112;
    colvals[5569] = 118;
    colvals[5570] = 119;
    colvals[5571] = 122;
    colvals[5572] = 132;
    colvals[5573] = 138;
    colvals[5574] = 151;
    colvals[5575] = 166;
    colvals[5576] = 174;
    colvals[5577] = 176;
    colvals[5578] = 184;
    colvals[5579] = 190;
    colvals[5580] = 200;
    colvals[5581] = 202;
    colvals[5582] = 206;
    colvals[5583] = 207;
    colvals[5584] = 208;
    colvals[5585] = 48;
    colvals[5586] = 52;
    colvals[5587] = 100;
    colvals[5588] = 102;
    colvals[5589] = 114;
    colvals[5590] = 119;
    colvals[5591] = 128;
    colvals[5592] = 132;
    colvals[5593] = 138;
    colvals[5594] = 151;
    colvals[5595] = 166;
    colvals[5596] = 186;
    colvals[5597] = 209;
    colvals[5598] = 52;
    colvals[5599] = 100;
    colvals[5600] = 101;
    colvals[5601] = 102;
    colvals[5602] = 176;
    colvals[5603] = 181;
    colvals[5604] = 186;
    colvals[5605] = 190;
    colvals[5606] = 197;
    colvals[5607] = 209;
    colvals[5608] = 210;
    colvals[5609] = 49;
    colvals[5610] = 51;
    colvals[5611] = 52;
    colvals[5612] = 75;
    colvals[5613] = 81;
    colvals[5614] = 98;
    colvals[5615] = 100;
    colvals[5616] = 101;
    colvals[5617] = 102;
    colvals[5618] = 119;
    colvals[5619] = 132;
    colvals[5620] = 138;
    colvals[5621] = 147;
    colvals[5622] = 152;
    colvals[5623] = 153;
    colvals[5624] = 154;
    colvals[5625] = 156;
    colvals[5626] = 166;
    colvals[5627] = 169;
    colvals[5628] = 172;
    colvals[5629] = 174;
    colvals[5630] = 176;
    colvals[5631] = 181;
    colvals[5632] = 182;
    colvals[5633] = 183;
    colvals[5634] = 185;
    colvals[5635] = 187;
    colvals[5636] = 207;
    colvals[5637] = 210;
    colvals[5638] = 211;
    colvals[5639] = 212;
    colvals[5640] = 213;
    colvals[5641] = 214;
    colvals[5642] = 52;
    colvals[5643] = 57;
    colvals[5644] = 60;
    colvals[5645] = 95;
    colvals[5646] = 96;
    colvals[5647] = 100;
    colvals[5648] = 101;
    colvals[5649] = 102;
    colvals[5650] = 114;
    colvals[5651] = 115;
    colvals[5652] = 138;
    colvals[5653] = 148;
    colvals[5654] = 154;
    colvals[5655] = 156;
    colvals[5656] = 166;
    colvals[5657] = 174;
    colvals[5658] = 175;
    colvals[5659] = 176;
    colvals[5660] = 177;
    colvals[5661] = 181;
    colvals[5662] = 183;
    colvals[5663] = 184;
    colvals[5664] = 185;
    colvals[5665] = 186;
    colvals[5666] = 210;
    colvals[5667] = 211;
    colvals[5668] = 212;
    colvals[5669] = 213;
    colvals[5670] = 214;
    colvals[5671] = 50;
    colvals[5672] = 51;
    colvals[5673] = 52;
    colvals[5674] = 96;
    colvals[5675] = 100;
    colvals[5676] = 102;
    colvals[5677] = 112;
    colvals[5678] = 113;
    colvals[5679] = 119;
    colvals[5680] = 138;
    colvals[5681] = 153;
    colvals[5682] = 166;
    colvals[5683] = 174;
    colvals[5684] = 175;
    colvals[5685] = 176;
    colvals[5686] = 183;
    colvals[5687] = 185;
    colvals[5688] = 211;
    colvals[5689] = 213;
    colvals[5690] = 214;
    colvals[5691] = 95;
    colvals[5692] = 100;
    colvals[5693] = 101;
    colvals[5694] = 102;
    colvals[5695] = 103;
    colvals[5696] = 138;
    colvals[5697] = 175;
    colvals[5698] = 176;
    colvals[5699] = 213;
    colvals[5700] = 214;
    
    // value of each non-zero element
    data[0] = 0.0 - k[2315] - k[2316] - k[2317] - k[2318];
    data[1] = 0.0 + k[2209];
    data[2] = 0.0 + k[2228];
    data[3] = 0.0 + k[2292];
    data[4] = 0.0 - k[2323] - k[2324] - k[2325] - k[2326];
    data[5] = 0.0 + k[2224];
    data[6] = 0.0 + k[2242];
    data[7] = 0.0 - k[2327] - k[2328] - k[2329] - k[2330];
    data[8] = 0.0 + k[2298];
    data[9] = 0.0 + k[2277];
    data[10] = 0.0 - k[2339] - k[2340] - k[2341] - k[2342];
    data[11] = 0.0 + k[2146];
    data[12] = 0.0 - k[2351] - k[2352] - k[2353] - k[2354];
    data[13] = 0.0 + k[2300];
    data[14] = 0.0 - k[2359] - k[2360] - k[2361] - k[2362];
    data[15] = 0.0 + k[2301];
    data[16] = 0.0 - k[2439] - k[2440] - k[2441] - k[2442];
    data[17] = 0.0 + k[2150];
    data[18] = 0.0 + k[2151];
    data[19] = 0.0 - k[2399] - k[2400] - k[2401] - k[2402];
    data[20] = 0.0 + k[2199];
    data[21] = 0.0 - k[2463] - k[2464] - k[2465] - k[2466];
    data[22] = 0.0 + k[2200];
    data[23] = 0.0 - k[2491] - k[2492] - k[2493] - k[2494];
    data[24] = 0.0 + k[2166];
    data[25] = 0.0 + k[2165];
    data[26] = 0.0 - k[2415] - k[2416] - k[2417] - k[2418];
    data[27] = 0.0 + k[2147];
    data[28] = 0.0 - k[2403] - k[2404] - k[2405] - k[2406];
    data[29] = 0.0 + k[2148];
    data[30] = 0.0 + k[2149];
    data[31] = 0.0 - k[2411] - k[2412] - k[2413] - k[2414];
    data[32] = 0.0 + k[2299];
    data[33] = 0.0 - k[2419] - k[2420] - k[2421] - k[2422];
    data[34] = 0.0 + k[2159];
    data[35] = 0.0 + k[2160];
    data[36] = 0.0 + k[2162];
    data[37] = 0.0 + k[2155];
    data[38] = 0.0 + k[2156];
    data[39] = 0.0 - k[2379] - k[2380] - k[2381] - k[2382];
    data[40] = 0.0 + k[2154];
    data[41] = 0.0 + k[2152];
    data[42] = 0.0 + k[2296];
    data[43] = 0.0 + k[2202];
    data[44] = 0.0 - k[2303] - k[2304] - k[2305] - k[2306];
    data[45] = 0.0 + k[2206];
    data[46] = 0.0 + k[2221];
    data[47] = 0.0 + k[2210];
    data[48] = 0.0 + k[2231];
    data[49] = 0.0 + k[2213];
    data[50] = 0.0 + k[2237];
    data[51] = 0.0 + k[2216];
    data[52] = 0.0 + k[2245];
    data[53] = 0.0 + k[2217];
    data[54] = 0.0 + k[2249];
    data[55] = 0.0 + k[2248];
    data[56] = 0.0 - k[2343] - k[2344] - k[2345] - k[2346];
    data[57] = 0.0 + k[2207];
    data[58] = 0.0 + k[2234];
    data[59] = 0.0 - k[2435] - k[2436] - k[2437] - k[2438];
    data[60] = 0.0 + k[2215];
    data[61] = 0.0 + k[2247];
    data[62] = 0.0 - k[2431] - k[2432] - k[2433] - k[2434];
    data[63] = 0.0 - k[2355] - k[2356] - k[2357] - k[2358];
    data[64] = 0.0 + k[2275];
    data[65] = 0.0 - k[2367] - k[2368] - k[2369] - k[2370];
    data[66] = 0.0 + k[2169];
    data[67] = 0.0 + k[2208];
    data[68] = 0.0 + k[2244];
    data[69] = 0.0 + k[2218];
    data[70] = 0.0 + k[2240];
    data[71] = 0.0 + k[2168];
    data[72] = 0.0 - k[2443] - k[2444] - k[2445] - k[2446];
    data[73] = 0.0 + k[2255];
    data[74] = 0.0 + k[2266];
    data[75] = 0.0 + k[2263];
    data[76] = 0.0 + k[2279];
    data[77] = 0.0 + k[2282];
    data[78] = 0.0 + k[2258];
    data[79] = 0.0 + k[2268];
    data[80] = 0.0 - k[2311] - k[2312] - k[2313] - k[2314];
    data[81] = 0.0 + k[2214];
    data[82] = 0.0 + k[2239];
    data[83] = 0.0 + k[2246];
    data[84] = 0.0 + k[2252];
    data[85] = 0.0 + k[2227];
    data[86] = 0.0 + k[2211];
    data[87] = 0.0 + k[2233];
    data[88] = 0.0 - k[2391] - k[2392] - k[2393] - k[2394];
    data[89] = 0.0 + k[2257];
    data[90] = 0.0 + k[2293];
    data[91] = 0.0 + k[2278];
    data[92] = 0.0 + k[2254];
    data[93] = 0.0 + k[2265];
    data[94] = 0.0 + k[2261];
    data[95] = 0.0 + k[2264];
    data[96] = 0.0 - k[2503] - k[2504] - k[2505] - k[2506];
    data[97] = 0.0 + k[2203];
    data[98] = 0.0 + k[2280];
    data[99] = 0.0 + k[2204];
    data[100] = 0.0 + k[2205];
    data[101] = 0.0 + k[2201];
    data[102] = 0.0 + k[2281];
    data[103] = 0.0 - k[2455] - k[2456] - k[2457] - k[2458];
    data[104] = 0.0 + k[2192];
    data[105] = 0.0 + k[2171];
    data[106] = 0.0 + k[2187];
    data[107] = 0.0 + k[2188];
    data[108] = 0.0 - k[2467] - k[2468] - k[2469] - k[2470];
    data[109] = 0.0 + k[2161];
    data[110] = 0.0 + k[2145];
    data[111] = 0.0 - k[2395] - k[2396] - k[2397] - k[2398];
    data[112] = 0.0 + k[2194];
    data[113] = 0.0 + k[2196];
    data[114] = 0.0 + k[2198];
    data[115] = 0.0 + k[2195];
    data[116] = 0.0 + k[2197];
    data[117] = 0.0 - k[2331] - k[2332] - k[2333] - k[2334];
    data[118] = 0.0 + k[2220];
    data[119] = 0.0 + k[2235];
    data[120] = 0.0 + k[2223];
    data[121] = 0.0 + k[2241];
    data[122] = 0.0 + k[2289];
    data[123] = 0.0 - k[2479] - k[2480] - k[2481] - k[2482];
    data[124] = 0.0 + k[2167];
    data[125] = 0.0 + k[2153];
    data[126] = 0.0 - k[2335] - k[2336] - k[2337] - k[2338];
    data[127] = 0.0 + k[2297];
    data[128] = 0.0 - k[2423] - k[2424] - k[2425] - k[2426];
    data[129] = 0.0 + k[2157];
    data[130] = 0.0 + k[2158];
    data[131] = 0.0 - k[2371] - k[2372] - k[2373] - k[2374];
    data[132] = 0.0 + k[2273];
    data[133] = 0.0 + k[2271];
    data[134] = 0.0 + k[2272];
    data[135] = 0.0 - k[2319] - k[2320] - k[2321] - k[2322];
    data[136] = 0.0 + k[2287];
    data[137] = 0.0 + k[2286];
    data[138] = 0.0 - k[2347] - k[2348] - k[2349] - k[2350];
    data[139] = 0.0 + k[2219];
    data[140] = 0.0 + k[2230];
    data[141] = 0.0 + k[2290];
    data[142] = 0.0 - k[2471] - k[2472] - k[2473] - k[2474];
    data[143] = 0.0 + k[2163];
    data[144] = 0.0 + k[2164];
    data[145] = 0.0 - k[2307] - k[2308] - k[2309] - k[2310];
    data[146] = 0.0 + k[2251];
    data[147] = 0.0 + k[2226];
    data[148] = 0.0 + k[2222];
    data[149] = 0.0 + k[2232];
    data[150] = 0.0 + k[2250];
    data[151] = 0.0 + k[2238];
    data[152] = 0.0 + k[2225];
    data[153] = 0.0 + k[2243];
    data[154] = 0.0 + k[2288];
    data[155] = 0.0 - k[2363] - k[2364] - k[2365] - k[2366];
    data[156] = 0.0 + k[2212];
    data[157] = 0.0 + k[2236];
    data[158] = 0.0 - k[2447] - k[2448] - k[2449] - k[2450];
    data[159] = 0.0 + k[2270];
    data[160] = 0.0 - k[2451] - k[2452] - k[2453] - k[2454];
    data[161] = 0.0 + k[2291];
    data[162] = 0.0 + k[2262];
    data[163] = 0.0 + k[2276];
    data[164] = 0.0 - k[2375] - k[2376] - k[2377] - k[2378];
    data[165] = 0.0 + k[2253];
    data[166] = 0.0 + k[2229];
    data[167] = 0.0 - k[2387] - k[2388] - k[2389] - k[2390];
    data[168] = 0.0 + k[2294];
    data[169] = 0.0 + k[2274];
    data[170] = 0.0 - k[2487] - k[2488] - k[2489] - k[2490];
    data[171] = 0.0 + k[2285];
    data[172] = 0.0 + k[2259];
    data[173] = 0.0 + k[2269];
    data[174] = 0.0 - k[2407] - k[2408] - k[2409] - k[2410];
    data[175] = 0.0 + k[2181];
    data[176] = 0.0 + k[2183];
    data[177] = 0.0 - k[2475] - k[2476] - k[2477] - k[2478];
    data[178] = 0.0 + k[2182];
    data[179] = 0.0 + k[2184];
    data[180] = 0.0 - k[2495] - k[2496] - k[2497] - k[2498];
    data[181] = 0.0 + k[2185];
    data[182] = 0.0 + k[2186];
    data[183] = 0.0 - k[2383] - k[2384] - k[2385] - k[2386];
    data[184] = 0.0 + k[2170];
    data[185] = 0.0 + k[2173];
    data[186] = 0.0 + k[2172];
    data[187] = 0.0 + k[2174];
    data[188] = 0.0 + k[2175];
    data[189] = 0.0 + k[2176];
    data[190] = 0.0 + k[2177];
    data[191] = 0.0 + k[2178];
    data[192] = 0.0 + k[2179];
    data[193] = 0.0 + k[2180];
    data[194] = 0.0 + k[2193];
    data[195] = 0.0 - k[2427] - k[2428] - k[2429] - k[2430];
    data[196] = 0.0 - k[2483] - k[2484] - k[2485] - k[2486];
    data[197] = 0.0 + k[2191];
    data[198] = 0.0 + k[2189];
    data[199] = 0.0 + k[2190];
    data[200] = 0.0 - k[2459] - k[2460] - k[2461] - k[2462];
    data[201] = 0.0 + k[2283];
    data[202] = 0.0 + k[2256];
    data[203] = 0.0 + k[2267];
    data[204] = 0.0 - k[2499] - k[2500] - k[2501] - k[2502];
    data[205] = 0.0 + k[2295];
    data[206] = 0.0 + k[2260];
    data[207] = 0.0 + k[2284];
    data[208] = 0.0 - k[50]*y[IDX_C2II] - k[51]*y[IDX_CNII] -
        k[52]*y[IDX_COII] - k[53]*y[IDX_N2II] - k[54]*y[IDX_O2II] -
        k[209]*y[IDX_HeII] - k[356] - k[379] - k[685]*y[IDX_C2HII] -
        k[686]*y[IDX_CHII] - k[687]*y[IDX_CH2II] - k[688]*y[IDX_CH3II] -
        k[689]*y[IDX_CH5II] - k[690]*y[IDX_H2OII] - k[691]*y[IDX_H2SII] -
        k[692]*y[IDX_H3OII] - k[693]*y[IDX_HCNII] - k[694]*y[IDX_HCOII] -
        k[695]*y[IDX_HCO2II] - k[696]*y[IDX_HNOII] - k[697]*y[IDX_HSII] -
        k[698]*y[IDX_N2HII] - k[699]*y[IDX_NHII] - k[700]*y[IDX_O2II] -
        k[701]*y[IDX_O2HII] - k[702]*y[IDX_OHII] - k[703]*y[IDX_SiHII] -
        k[704]*y[IDX_SiOII] - k[912]*y[IDX_H2II] - k[1027]*y[IDX_H3II] -
        k[1582]*y[IDX_C2H3I] - k[1583]*y[IDX_C2H5I] - k[1584]*y[IDX_C2NI] -
        k[1585]*y[IDX_C3H2I] - k[1586]*y[IDX_CH2I] - k[1587]*y[IDX_CH2I] -
        k[1588]*y[IDX_CH3I] - k[1589]*y[IDX_CHI] - k[1590]*y[IDX_CNI] -
        k[1591]*y[IDX_COI] - k[1592]*y[IDX_CSI] - k[1593]*y[IDX_H2CNI] -
        k[1594]*y[IDX_HCOI] - k[1595]*y[IDX_HSI] - k[1596]*y[IDX_HSI] -
        k[1597]*y[IDX_N2I] - k[1598]*y[IDX_NCCNI] - k[1599]*y[IDX_NH2I] -
        k[1600]*y[IDX_NH2I] - k[1601]*y[IDX_NH2I] - k[1602]*y[IDX_NHI] -
        k[1603]*y[IDX_NHI] - k[1604]*y[IDX_NOI] - k[1605]*y[IDX_NOI] -
        k[1606]*y[IDX_NSI] - k[1607]*y[IDX_NSI] - k[1608]*y[IDX_O2I] -
        k[1609]*y[IDX_OCNI] - k[1610]*y[IDX_OCSI] - k[1611]*y[IDX_OHI] -
        k[1612]*y[IDX_OHI] - k[1613]*y[IDX_S2I] - k[1614]*y[IDX_SO2I] -
        k[1615]*y[IDX_SOI] - k[1616]*y[IDX_SOI] - k[1617]*y[IDX_SiHI] -
        k[1722]*y[IDX_H2I] - k[1788]*y[IDX_HNCOI] - k[1976] - k[2096]*y[IDX_CII]
        - k[2101]*y[IDX_CI] - k[2101]*y[IDX_CI] - k[2101]*y[IDX_CI] -
        k[2101]*y[IDX_CI] - k[2102]*y[IDX_NI] - k[2103]*y[IDX_OII] -
        k[2104]*y[IDX_OI] - k[2105]*y[IDX_SII] - k[2106]*y[IDX_SI] -
        k[2113]*y[IDX_H2I] - k[2123]*y[IDX_HI] - k[2206];
    data[209] = 0.0 + k[14]*y[IDX_CH2I] + k[15]*y[IDX_CHI] +
        k[16]*y[IDX_H2COI] + k[17]*y[IDX_H2SI] + k[18]*y[IDX_HCOI] +
        k[19]*y[IDX_MgI] + k[21]*y[IDX_NH3I] + k[22]*y[IDX_NOI] +
        k[23]*y[IDX_NSI] + k[24]*y[IDX_OCSI] + k[25]*y[IDX_SOI] +
        k[26]*y[IDX_SiI] + k[27]*y[IDX_SiC2I] + k[28]*y[IDX_SiC3I] +
        k[29]*y[IDX_SiCI] + k[30]*y[IDX_SiH2I] + k[31]*y[IDX_SiH3I] +
        k[32]*y[IDX_SiSI] + k[345]*y[IDX_SI] - k[2096]*y[IDX_CI] +
        k[2132]*y[IDX_EM];
    data[210] = 0.0 + k[366] + k[366] + k[647]*y[IDX_C2II] +
        k[658]*y[IDX_SII] + k[1177]*y[IDX_HeII] + k[1347]*y[IDX_NHII] +
        k[1463]*y[IDX_OII] + k[1573]*y[IDX_SI] + k[1736]*y[IDX_HI] +
        k[1790]*y[IDX_NI] + k[1867]*y[IDX_OI] + k[1962] + k[1962];
    data[211] = 0.0 - k[50]*y[IDX_CI] + k[458]*y[IDX_EM] + k[458]*y[IDX_EM]
        + k[647]*y[IDX_C2I] + k[650]*y[IDX_SI] + k[1490]*y[IDX_OI] + k[1960];
    data[212] = 0.0 + k[1187]*y[IDX_HeII];
    data[213] = 0.0 + k[460]*y[IDX_EM] - k[685]*y[IDX_CI] +
        k[1491]*y[IDX_OI];
    data[214] = 0.0 - k[1582]*y[IDX_CI];
    data[215] = 0.0 - k[1583]*y[IDX_CI];
    data[216] = 0.0 + k[376] - k[1584]*y[IDX_CI] + k[1973];
    data[217] = 0.0 + k[469]*y[IDX_EM];
    data[218] = 0.0 + k[473]*y[IDX_EM] + k[2292];
    data[219] = 0.0 - k[1585]*y[IDX_CI];
    data[220] = 0.0 + k[476]*y[IDX_EM];
    data[221] = 0.0 + k[2]*y[IDX_H2I] + k[9]*y[IDX_HI] + k[15]*y[IDX_CII] +
        k[391] + k[841]*y[IDX_COII] + k[856]*y[IDX_NH3II] - k[1589]*y[IDX_CI] +
        k[1683]*y[IDX_NI] + k[1694]*y[IDX_OI] + k[1698]*y[IDX_SI] +
        k[1743]*y[IDX_HI] + k[1999];
    data[222] = 0.0 + k[477]*y[IDX_EM] - k[686]*y[IDX_CI] +
        k[708]*y[IDX_CH3OHI] + k[716]*y[IDX_H2COI] + k[719]*y[IDX_H2OI] +
        k[721]*y[IDX_H2SI] + k[725]*y[IDX_HCNI] + k[727]*y[IDX_HNCI] +
        k[730]*y[IDX_NH3I] + k[737]*y[IDX_OCSI] + k[740]*y[IDX_SI] + k[1977];
    data[223] = 0.0 + k[14]*y[IDX_CII] - k[1586]*y[IDX_CI] -
        k[1587]*y[IDX_CI];
    data[224] = 0.0 + k[478]*y[IDX_EM] + k[479]*y[IDX_EM] -
        k[687]*y[IDX_CI];
    data[225] = 0.0 - k[1588]*y[IDX_CI];
    data[226] = 0.0 - k[688]*y[IDX_CI];
    data[227] = 0.0 + k[708]*y[IDX_CHII];
    data[228] = 0.0 - k[689]*y[IDX_CI];
    data[229] = 0.0 + k[392] + k[1207]*y[IDX_HeII] + k[1469]*y[IDX_OII] -
        k[1590]*y[IDX_CI] + k[1714]*y[IDX_SI] + k[1807]*y[IDX_NI] +
        k[1881]*y[IDX_OI] + k[2001];
    data[230] = 0.0 - k[51]*y[IDX_CI] + k[498]*y[IDX_EM] +
        k[1332]*y[IDX_NI];
    data[231] = 0.0 + k[394] + k[1297]*y[IDX_NII] - k[1591]*y[IDX_CI] +
        k[1745]*y[IDX_HI] + k[1957]*y[IDX_SiI] + k[2004];
    data[232] = 0.0 - k[52]*y[IDX_CI] + k[499]*y[IDX_EM] +
        k[841]*y[IDX_CHI];
    data[233] = 0.0 + k[1211]*y[IDX_HeII];
    data[234] = 0.0 + k[396] + k[1214]*y[IDX_HeII] - k[1592]*y[IDX_CI] +
        k[1884]*y[IDX_OI] + k[2007];
    data[235] = 0.0 + k[500]*y[IDX_EM] + k[2005];
    data[236] = 0.0 + k[458]*y[IDX_C2II] + k[458]*y[IDX_C2II] +
        k[460]*y[IDX_C2HII] + k[469]*y[IDX_C2NII] + k[473]*y[IDX_C3II] +
        k[476]*y[IDX_C4NII] + k[477]*y[IDX_CHII] + k[478]*y[IDX_CH2II] +
        k[479]*y[IDX_CH2II] + k[498]*y[IDX_CNII] + k[499]*y[IDX_COII] +
        k[500]*y[IDX_CSII] + k[579]*y[IDX_OCSII] + k[587]*y[IDX_SiCII] +
        k[589]*y[IDX_SiC2II] + k[590]*y[IDX_SiC3II] + k[2132]*y[IDX_CII];
    data[237] = 0.0 + k[9]*y[IDX_CHI] + k[1736]*y[IDX_C2I] +
        k[1743]*y[IDX_CHI] + k[1745]*y[IDX_COI] - k[2123]*y[IDX_CI];
    data[238] = 0.0 + k[2]*y[IDX_CHI] - k[1722]*y[IDX_CI] -
        k[2113]*y[IDX_CI];
    data[239] = 0.0 - k[912]*y[IDX_CI];
    data[240] = 0.0 - k[1593]*y[IDX_CI];
    data[241] = 0.0 + k[16]*y[IDX_CII] + k[716]*y[IDX_CHII];
    data[242] = 0.0 + k[719]*y[IDX_CHII];
    data[243] = 0.0 - k[690]*y[IDX_CI];
    data[244] = 0.0 + k[17]*y[IDX_CII] + k[721]*y[IDX_CHII];
    data[245] = 0.0 - k[691]*y[IDX_CI];
    data[246] = 0.0 - k[1027]*y[IDX_CI];
    data[247] = 0.0 - k[692]*y[IDX_CI];
    data[248] = 0.0 + k[725]*y[IDX_CHII];
    data[249] = 0.0 - k[693]*y[IDX_CI];
    data[250] = 0.0 + k[18]*y[IDX_CII] - k[1594]*y[IDX_CI];
    data[251] = 0.0 - k[694]*y[IDX_CI];
    data[252] = 0.0 - k[695]*y[IDX_CI];
    data[253] = 0.0 - k[209]*y[IDX_CI] + k[1177]*y[IDX_C2I] +
        k[1187]*y[IDX_C2HI] + k[1207]*y[IDX_CNI] + k[1211]*y[IDX_CO2I] +
        k[1214]*y[IDX_CSI] + k[1244]*y[IDX_HNCI] + k[1275]*y[IDX_SiC3I] +
        k[1276]*y[IDX_SiCI];
    data[254] = 0.0 + k[727]*y[IDX_CHII] + k[1244]*y[IDX_HeII];
    data[255] = 0.0 - k[1788]*y[IDX_CI];
    data[256] = 0.0 - k[696]*y[IDX_CI];
    data[257] = 0.0 - k[1595]*y[IDX_CI] - k[1596]*y[IDX_CI];
    data[258] = 0.0 - k[697]*y[IDX_CI];
    data[259] = 0.0 + k[19]*y[IDX_CII];
    data[260] = 0.0 + k[1332]*y[IDX_CNII] + k[1683]*y[IDX_CHI] +
        k[1790]*y[IDX_C2I] + k[1807]*y[IDX_CNI] - k[2102]*y[IDX_CI];
    data[261] = 0.0 + k[1297]*y[IDX_COI];
    data[262] = 0.0 - k[1597]*y[IDX_CI];
    data[263] = 0.0 - k[53]*y[IDX_CI];
    data[264] = 0.0 - k[698]*y[IDX_CI];
    data[265] = 0.0 - k[1598]*y[IDX_CI];
    data[266] = 0.0 - k[1602]*y[IDX_CI] - k[1603]*y[IDX_CI];
    data[267] = 0.0 - k[699]*y[IDX_CI] + k[1347]*y[IDX_C2I];
    data[268] = 0.0 - k[1599]*y[IDX_CI] - k[1600]*y[IDX_CI] -
        k[1601]*y[IDX_CI];
    data[269] = 0.0 + k[21]*y[IDX_CII] + k[730]*y[IDX_CHII];
    data[270] = 0.0 + k[856]*y[IDX_CHI];
    data[271] = 0.0 + k[22]*y[IDX_CII] - k[1604]*y[IDX_CI] -
        k[1605]*y[IDX_CI];
    data[272] = 0.0 + k[23]*y[IDX_CII] - k[1606]*y[IDX_CI] -
        k[1607]*y[IDX_CI];
    data[273] = 0.0 + k[1490]*y[IDX_C2II] + k[1491]*y[IDX_C2HII] +
        k[1512]*y[IDX_SiCII] + k[1694]*y[IDX_CHI] + k[1867]*y[IDX_C2I] +
        k[1881]*y[IDX_CNI] + k[1884]*y[IDX_CSI] + k[1921]*y[IDX_SiCI] -
        k[2104]*y[IDX_CI];
    data[274] = 0.0 + k[1463]*y[IDX_C2I] + k[1469]*y[IDX_CNI] -
        k[2103]*y[IDX_CI];
    data[275] = 0.0 - k[1608]*y[IDX_CI];
    data[276] = 0.0 - k[54]*y[IDX_CI] - k[700]*y[IDX_CI];
    data[277] = 0.0 - k[701]*y[IDX_CI];
    data[278] = 0.0 - k[1609]*y[IDX_CI];
    data[279] = 0.0 + k[24]*y[IDX_CII] + k[737]*y[IDX_CHII] -
        k[1610]*y[IDX_CI];
    data[280] = 0.0 + k[579]*y[IDX_EM];
    data[281] = 0.0 - k[1611]*y[IDX_CI] - k[1612]*y[IDX_CI];
    data[282] = 0.0 - k[702]*y[IDX_CI];
    data[283] = 0.0 + k[345]*y[IDX_CII] + k[650]*y[IDX_C2II] +
        k[740]*y[IDX_CHII] + k[1573]*y[IDX_C2I] + k[1698]*y[IDX_CHI] +
        k[1714]*y[IDX_CNI] - k[2106]*y[IDX_CI];
    data[284] = 0.0 + k[658]*y[IDX_C2I] - k[2105]*y[IDX_CI];
    data[285] = 0.0 - k[1613]*y[IDX_CI];
    data[286] = 0.0 + k[26]*y[IDX_CII] + k[1957]*y[IDX_COI];
    data[287] = 0.0 + k[29]*y[IDX_CII] + k[451] + k[1276]*y[IDX_HeII] +
        k[1921]*y[IDX_OI] + k[2081];
    data[288] = 0.0 + k[587]*y[IDX_EM] + k[1512]*y[IDX_OI];
    data[289] = 0.0 + k[27]*y[IDX_CII] + k[449];
    data[290] = 0.0 + k[589]*y[IDX_EM];
    data[291] = 0.0 + k[28]*y[IDX_CII] + k[450] + k[1275]*y[IDX_HeII] +
        k[2080];
    data[292] = 0.0 + k[590]*y[IDX_EM];
    data[293] = 0.0 - k[1617]*y[IDX_CI];
    data[294] = 0.0 - k[703]*y[IDX_CI];
    data[295] = 0.0 + k[30]*y[IDX_CII];
    data[296] = 0.0 + k[31]*y[IDX_CII];
    data[297] = 0.0 - k[704]*y[IDX_CI];
    data[298] = 0.0 + k[32]*y[IDX_CII];
    data[299] = 0.0 + k[25]*y[IDX_CII] - k[1615]*y[IDX_CI] -
        k[1616]*y[IDX_CI];
    data[300] = 0.0 - k[1614]*y[IDX_CI];
    data[301] = 0.0 + k[50]*y[IDX_C2II] + k[51]*y[IDX_CNII] +
        k[52]*y[IDX_COII] + k[53]*y[IDX_N2II] + k[54]*y[IDX_O2II] +
        k[209]*y[IDX_HeII] + k[356] + k[379] + k[1976] - k[2096]*y[IDX_CII];
    data[302] = 0.0 - k[14]*y[IDX_CH2I] - k[15]*y[IDX_CHI] -
        k[16]*y[IDX_H2COI] - k[17]*y[IDX_H2SI] - k[18]*y[IDX_HCOI] -
        k[19]*y[IDX_MgI] - k[20]*y[IDX_NCCNI] - k[21]*y[IDX_NH3I] -
        k[22]*y[IDX_NOI] - k[23]*y[IDX_NSI] - k[24]*y[IDX_OCSI] -
        k[25]*y[IDX_SOI] - k[26]*y[IDX_SiI] - k[27]*y[IDX_SiC2I] -
        k[28]*y[IDX_SiC3I] - k[29]*y[IDX_SiCI] - k[30]*y[IDX_SiH2I] -
        k[31]*y[IDX_SiH3I] - k[32]*y[IDX_SiSI] - k[345]*y[IDX_SI] -
        k[606]*y[IDX_C2H5OHI] - k[607]*y[IDX_C2HI] - k[608]*y[IDX_CH2I] -
        k[609]*y[IDX_CH3I] - k[610]*y[IDX_CH3I] - k[611]*y[IDX_CH3CCHI] -
        k[612]*y[IDX_CH3OHI] - k[613]*y[IDX_CH3OHI] - k[614]*y[IDX_CH4I] -
        k[615]*y[IDX_CHI] - k[616]*y[IDX_CO2I] - k[617]*y[IDX_H2COI] -
        k[618]*y[IDX_H2COI] - k[619]*y[IDX_H2CSI] - k[620]*y[IDX_H2OI] -
        k[621]*y[IDX_H2OI] - k[622]*y[IDX_H2SI] - k[623]*y[IDX_HC3NI] -
        k[624]*y[IDX_HC3NI] - k[625]*y[IDX_HC3NI] - k[626]*y[IDX_HCOI] -
        k[627]*y[IDX_HNCI] - k[628]*y[IDX_HSI] - k[629]*y[IDX_NH2I] -
        k[630]*y[IDX_NH3I] - k[631]*y[IDX_NHI] - k[632]*y[IDX_NSI] -
        k[633]*y[IDX_O2I] - k[634]*y[IDX_O2I] - k[635]*y[IDX_OCNI] -
        k[636]*y[IDX_OCSI] - k[637]*y[IDX_OHI] - k[638]*y[IDX_SO2I] -
        k[639]*y[IDX_SOI] - k[640]*y[IDX_SOI] - k[641]*y[IDX_SOI] -
        k[642]*y[IDX_SiCI] - k[643]*y[IDX_SiH2I] - k[644]*y[IDX_SiHI] -
        k[645]*y[IDX_SiOI] - k[646]*y[IDX_SiSI] - k[934]*y[IDX_H2I] -
        k[2096]*y[IDX_CI] - k[2097]*y[IDX_NI] - k[2098]*y[IDX_OI] -
        k[2099]*y[IDX_SI] - k[2112]*y[IDX_H2I] - k[2122]*y[IDX_HI] -
        k[2132]*y[IDX_EM] - k[2221];
    data[303] = 0.0 + k[1177]*y[IDX_HeII];
    data[304] = 0.0 + k[50]*y[IDX_CI] + k[1325]*y[IDX_NI] + k[1960];
    data[305] = 0.0 - k[607]*y[IDX_CII] + k[1188]*y[IDX_HeII];
    data[306] = 0.0 - k[606]*y[IDX_CII];
    data[307] = 0.0 + k[1189]*y[IDX_HeII];
    data[308] = 0.0 - k[15]*y[IDX_CII] - k[615]*y[IDX_CII] +
        k[1206]*y[IDX_HeII];
    data[309] = 0.0 + k[380] + k[1098]*y[IDX_HI];
    data[310] = 0.0 - k[14]*y[IDX_CII] - k[608]*y[IDX_CII] +
        k[1192]*y[IDX_HeII];
    data[311] = 0.0 + k[1978];
    data[312] = 0.0 - k[609]*y[IDX_CII] - k[610]*y[IDX_CII];
    data[313] = 0.0 - k[611]*y[IDX_CII];
    data[314] = 0.0 - k[612]*y[IDX_CII] - k[613]*y[IDX_CII];
    data[315] = 0.0 - k[614]*y[IDX_CII];
    data[316] = 0.0 + k[1208]*y[IDX_HeII];
    data[317] = 0.0 + k[51]*y[IDX_CI];
    data[318] = 0.0 + k[1213]*y[IDX_HeII];
    data[319] = 0.0 + k[52]*y[IDX_CI] + k[2002];
    data[320] = 0.0 - k[616]*y[IDX_CII] + k[1212]*y[IDX_HeII];
    data[321] = 0.0 + k[1215]*y[IDX_HeII];
    data[322] = 0.0 - k[2132]*y[IDX_CII];
    data[323] = 0.0 + k[1098]*y[IDX_CHII] - k[2122]*y[IDX_CII];
    data[324] = 0.0 - k[934]*y[IDX_CII] - k[2112]*y[IDX_CII];
    data[325] = 0.0 - k[16]*y[IDX_CII] - k[617]*y[IDX_CII] -
        k[618]*y[IDX_CII];
    data[326] = 0.0 - k[619]*y[IDX_CII];
    data[327] = 0.0 - k[620]*y[IDX_CII] - k[621]*y[IDX_CII];
    data[328] = 0.0 - k[17]*y[IDX_CII] - k[622]*y[IDX_CII];
    data[329] = 0.0 - k[623]*y[IDX_CII] - k[624]*y[IDX_CII] -
        k[625]*y[IDX_CII];
    data[330] = 0.0 + k[1233]*y[IDX_HeII];
    data[331] = 0.0 - k[18]*y[IDX_CII] - k[626]*y[IDX_CII];
    data[332] = 0.0 + k[209]*y[IDX_CI] + k[1177]*y[IDX_C2I] +
        k[1188]*y[IDX_C2HI] + k[1189]*y[IDX_C2NI] + k[1192]*y[IDX_CH2I] +
        k[1206]*y[IDX_CHI] + k[1208]*y[IDX_CNI] + k[1212]*y[IDX_CO2I] +
        k[1213]*y[IDX_COI] + k[1215]*y[IDX_CSI] + k[1233]*y[IDX_HCNI] +
        k[1243]*y[IDX_HNCI] + k[1277]*y[IDX_SiCI];
    data[333] = 0.0 - k[627]*y[IDX_CII] + k[1243]*y[IDX_HeII];
    data[334] = 0.0 - k[628]*y[IDX_CII];
    data[335] = 0.0 - k[19]*y[IDX_CII];
    data[336] = 0.0 + k[1325]*y[IDX_C2II] - k[2097]*y[IDX_CII];
    data[337] = 0.0 + k[53]*y[IDX_CI];
    data[338] = 0.0 - k[20]*y[IDX_CII];
    data[339] = 0.0 - k[631]*y[IDX_CII];
    data[340] = 0.0 - k[629]*y[IDX_CII];
    data[341] = 0.0 - k[21]*y[IDX_CII] - k[630]*y[IDX_CII];
    data[342] = 0.0 - k[22]*y[IDX_CII];
    data[343] = 0.0 - k[23]*y[IDX_CII] - k[632]*y[IDX_CII];
    data[344] = 0.0 - k[2098]*y[IDX_CII];
    data[345] = 0.0 - k[633]*y[IDX_CII] - k[634]*y[IDX_CII];
    data[346] = 0.0 + k[54]*y[IDX_CI];
    data[347] = 0.0 - k[635]*y[IDX_CII];
    data[348] = 0.0 - k[24]*y[IDX_CII] - k[636]*y[IDX_CII];
    data[349] = 0.0 - k[637]*y[IDX_CII];
    data[350] = 0.0 - k[345]*y[IDX_CII] - k[2099]*y[IDX_CII];
    data[351] = 0.0 - k[26]*y[IDX_CII];
    data[352] = 0.0 - k[29]*y[IDX_CII] - k[642]*y[IDX_CII] +
        k[1277]*y[IDX_HeII];
    data[353] = 0.0 - k[27]*y[IDX_CII];
    data[354] = 0.0 - k[28]*y[IDX_CII];
    data[355] = 0.0 - k[644]*y[IDX_CII];
    data[356] = 0.0 - k[30]*y[IDX_CII] - k[643]*y[IDX_CII];
    data[357] = 0.0 - k[31]*y[IDX_CII];
    data[358] = 0.0 - k[645]*y[IDX_CII];
    data[359] = 0.0 - k[32]*y[IDX_CII] - k[646]*y[IDX_CII];
    data[360] = 0.0 - k[25]*y[IDX_CII] - k[639]*y[IDX_CII] -
        k[640]*y[IDX_CII] - k[641]*y[IDX_CII];
    data[361] = 0.0 - k[638]*y[IDX_CII];
    data[362] = 0.0 + k[2315] + k[2316] + k[2317] + k[2318];
    data[363] = 0.0 + k[50]*y[IDX_C2II] + k[1584]*y[IDX_C2NI] +
        k[1589]*y[IDX_CHI] + k[1590]*y[IDX_CNI] + k[1591]*y[IDX_COI] +
        k[1592]*y[IDX_CSI] + k[2101]*y[IDX_CI] + k[2101]*y[IDX_CI];
    data[364] = 0.0 + k[642]*y[IDX_SiCI];
    data[365] = 0.0 - k[36]*y[IDX_CNII] - k[37]*y[IDX_COII] -
        k[38]*y[IDX_N2II] - k[39]*y[IDX_O2II] - k[110]*y[IDX_HII] -
        k[153]*y[IDX_H2II] - k[175]*y[IDX_H2OII] - k[207]*y[IDX_HeII] -
        k[233]*y[IDX_NII] - k[306]*y[IDX_OII] - k[329]*y[IDX_OHII] - k[366] -
        k[647]*y[IDX_C2II] - k[651]*y[IDX_H2COII] - k[652]*y[IDX_HCNII] -
        k[653]*y[IDX_HCOII] - k[654]*y[IDX_HNOII] - k[655]*y[IDX_N2HII] -
        k[656]*y[IDX_O2II] - k[657]*y[IDX_O2HII] - k[658]*y[IDX_SII] -
        k[659]*y[IDX_SiOII] - k[705]*y[IDX_CHII] - k[823]*y[IDX_CH5II] -
        k[909]*y[IDX_H2II] - k[976]*y[IDX_H2OII] - k[1022]*y[IDX_H3II] -
        k[1080]*y[IDX_H3OII] - k[1177]*y[IDX_HeII] - k[1345]*y[IDX_NHII] -
        k[1346]*y[IDX_NHII] - k[1347]*y[IDX_NHII] - k[1374]*y[IDX_NH2II] -
        k[1412]*y[IDX_NH3II] - k[1463]*y[IDX_OII] - k[1517]*y[IDX_OHII] -
        k[1570]*y[IDX_C2H2I] - k[1571]*y[IDX_HCNI] - k[1572]*y[IDX_O2I] -
        k[1573]*y[IDX_SI] - k[1736]*y[IDX_HI] - k[1790]*y[IDX_NI] -
        k[1867]*y[IDX_OI] - k[1961] - k[1962] - k[2209];
    data[366] = 0.0 + k[33]*y[IDX_HCOI] + k[34]*y[IDX_NOI] + k[35]*y[IDX_SI]
        + k[50]*y[IDX_CI] + k[62]*y[IDX_CH2I] + k[82]*y[IDX_CHI] +
        k[271]*y[IDX_NH2I] + k[339]*y[IDX_OHI] - k[647]*y[IDX_C2I];
    data[367] = 0.0 + k[373] + k[677]*y[IDX_COII] + k[1970];
    data[368] = 0.0 + k[459]*y[IDX_EM] + k[660]*y[IDX_H2COI] +
        k[662]*y[IDX_HCNI] + k[664]*y[IDX_HNCI] + k[754]*y[IDX_CH2I] +
        k[838]*y[IDX_CHI] + k[1395]*y[IDX_NH2I] + k[1418]*y[IDX_NH3I];
    data[369] = 0.0 - k[1570]*y[IDX_C2I];
    data[370] = 0.0 + k[461]*y[IDX_EM];
    data[371] = 0.0 + k[375] + k[1584]*y[IDX_CI] + k[1972];
    data[372] = 0.0 + k[468]*y[IDX_EM];
    data[373] = 0.0 + k[473]*y[IDX_EM];
    data[374] = 0.0 + k[377] + k[1974];
    data[375] = 0.0 + k[378] + k[1191]*y[IDX_HeII] + k[1975];
    data[376] = 0.0 + k[475]*y[IDX_EM];
    data[377] = 0.0 + k[82]*y[IDX_C2II] + k[838]*y[IDX_C2HII] +
        k[1589]*y[IDX_CI];
    data[378] = 0.0 - k[705]*y[IDX_C2I];
    data[379] = 0.0 + k[62]*y[IDX_C2II] + k[754]*y[IDX_C2HII];
    data[380] = 0.0 - k[823]*y[IDX_C2I];
    data[381] = 0.0 + k[1590]*y[IDX_CI] + k[1703]*y[IDX_CNI] +
        k[1703]*y[IDX_CNI];
    data[382] = 0.0 - k[36]*y[IDX_C2I];
    data[383] = 0.0 + k[1591]*y[IDX_CI];
    data[384] = 0.0 - k[37]*y[IDX_C2I] + k[677]*y[IDX_C2HI];
    data[385] = 0.0 + k[1592]*y[IDX_CI];
    data[386] = 0.0 + k[459]*y[IDX_C2HII] + k[461]*y[IDX_C2H2II] +
        k[468]*y[IDX_C2NII] + k[473]*y[IDX_C3II] + k[475]*y[IDX_C4NII] +
        k[588]*y[IDX_SiC2II] + k[591]*y[IDX_SiC3II];
    data[387] = 0.0 - k[1736]*y[IDX_C2I];
    data[388] = 0.0 - k[110]*y[IDX_C2I];
    data[389] = 0.0 - k[153]*y[IDX_C2I] - k[909]*y[IDX_C2I];
    data[390] = 0.0 + k[660]*y[IDX_C2HII];
    data[391] = 0.0 - k[651]*y[IDX_C2I];
    data[392] = 0.0 - k[175]*y[IDX_C2I] - k[976]*y[IDX_C2I];
    data[393] = 0.0 - k[1022]*y[IDX_C2I];
    data[394] = 0.0 - k[1080]*y[IDX_C2I];
    data[395] = 0.0 + k[662]*y[IDX_C2HII] - k[1571]*y[IDX_C2I];
    data[396] = 0.0 - k[652]*y[IDX_C2I];
    data[397] = 0.0 + k[33]*y[IDX_C2II];
    data[398] = 0.0 - k[653]*y[IDX_C2I];
    data[399] = 0.0 - k[207]*y[IDX_C2I] - k[1177]*y[IDX_C2I] +
        k[1191]*y[IDX_C4HI] + k[1274]*y[IDX_SiC2I];
    data[400] = 0.0 + k[664]*y[IDX_C2HII];
    data[401] = 0.0 - k[654]*y[IDX_C2I];
    data[402] = 0.0 - k[1790]*y[IDX_C2I];
    data[403] = 0.0 - k[233]*y[IDX_C2I];
    data[404] = 0.0 - k[38]*y[IDX_C2I];
    data[405] = 0.0 - k[655]*y[IDX_C2I];
    data[406] = 0.0 - k[1345]*y[IDX_C2I] - k[1346]*y[IDX_C2I] -
        k[1347]*y[IDX_C2I];
    data[407] = 0.0 + k[271]*y[IDX_C2II] + k[1395]*y[IDX_C2HII];
    data[408] = 0.0 - k[1374]*y[IDX_C2I];
    data[409] = 0.0 + k[1418]*y[IDX_C2HII];
    data[410] = 0.0 - k[1412]*y[IDX_C2I];
    data[411] = 0.0 + k[34]*y[IDX_C2II];
    data[412] = 0.0 - k[1867]*y[IDX_C2I];
    data[413] = 0.0 - k[306]*y[IDX_C2I] - k[1463]*y[IDX_C2I];
    data[414] = 0.0 - k[1572]*y[IDX_C2I];
    data[415] = 0.0 - k[39]*y[IDX_C2I] - k[656]*y[IDX_C2I];
    data[416] = 0.0 - k[657]*y[IDX_C2I];
    data[417] = 0.0 + k[339]*y[IDX_C2II];
    data[418] = 0.0 - k[329]*y[IDX_C2I] - k[1517]*y[IDX_C2I];
    data[419] = 0.0 + k[35]*y[IDX_C2II] - k[1573]*y[IDX_C2I];
    data[420] = 0.0 - k[658]*y[IDX_C2I];
    data[421] = 0.0 + k[642]*y[IDX_CII];
    data[422] = 0.0 + k[1274]*y[IDX_HeII] + k[2078];
    data[423] = 0.0 + k[588]*y[IDX_EM];
    data[424] = 0.0 + k[2079];
    data[425] = 0.0 + k[591]*y[IDX_EM];
    data[426] = 0.0 - k[659]*y[IDX_C2I];
    data[427] = 0.0 - k[50]*y[IDX_C2II] + k[686]*y[IDX_CHII] +
        k[2096]*y[IDX_CII];
    data[428] = 0.0 + k[615]*y[IDX_CHI] + k[2096]*y[IDX_CI];
    data[429] = 0.0 + k[36]*y[IDX_CNII] + k[37]*y[IDX_COII] +
        k[38]*y[IDX_N2II] + k[39]*y[IDX_O2II] + k[110]*y[IDX_HII] +
        k[153]*y[IDX_H2II] + k[175]*y[IDX_H2OII] + k[207]*y[IDX_HeII] +
        k[233]*y[IDX_NII] + k[306]*y[IDX_OII] + k[329]*y[IDX_OHII] -
        k[647]*y[IDX_C2II] + k[1961];
    data[430] = 0.0 - k[33]*y[IDX_HCOI] - k[34]*y[IDX_NOI] - k[35]*y[IDX_SI]
        - k[50]*y[IDX_CI] - k[62]*y[IDX_CH2I] - k[82]*y[IDX_CHI] -
        k[271]*y[IDX_NH2I] - k[339]*y[IDX_OHI] - k[458]*y[IDX_EM] -
        k[647]*y[IDX_C2I] - k[648]*y[IDX_HCOI] - k[649]*y[IDX_O2I] -
        k[650]*y[IDX_SI] - k[803]*y[IDX_CH4I] - k[804]*y[IDX_CH4I] -
        k[837]*y[IDX_CHI] - k[935]*y[IDX_H2I] - k[990]*y[IDX_H2OI] -
        k[1325]*y[IDX_NI] - k[1394]*y[IDX_NH2I] - k[1444]*y[IDX_NHI] -
        k[1445]*y[IDX_NHI] - k[1490]*y[IDX_OI] - k[1960] - k[2228];
    data[431] = 0.0 + k[884]*y[IDX_HII] + k[1186]*y[IDX_HeII];
    data[432] = 0.0 + k[1963];
    data[433] = 0.0 + k[1178]*y[IDX_HeII];
    data[434] = 0.0 + k[1190]*y[IDX_HeII];
    data[435] = 0.0 - k[82]*y[IDX_C2II] + k[615]*y[IDX_CII] +
        k[712]*y[IDX_CHII] - k[837]*y[IDX_C2II];
    data[436] = 0.0 + k[686]*y[IDX_CI] + k[712]*y[IDX_CHI];
    data[437] = 0.0 - k[62]*y[IDX_C2II];
    data[438] = 0.0 - k[803]*y[IDX_C2II] - k[804]*y[IDX_C2II];
    data[439] = 0.0 + k[36]*y[IDX_C2I];
    data[440] = 0.0 + k[37]*y[IDX_C2I];
    data[441] = 0.0 - k[458]*y[IDX_C2II];
    data[442] = 0.0 + k[110]*y[IDX_C2I] + k[884]*y[IDX_C2HI];
    data[443] = 0.0 - k[935]*y[IDX_C2II];
    data[444] = 0.0 + k[153]*y[IDX_C2I];
    data[445] = 0.0 - k[990]*y[IDX_C2II];
    data[446] = 0.0 + k[175]*y[IDX_C2I];
    data[447] = 0.0 - k[33]*y[IDX_C2II] - k[648]*y[IDX_C2II];
    data[448] = 0.0 + k[207]*y[IDX_C2I] + k[1178]*y[IDX_C2H2I] +
        k[1186]*y[IDX_C2HI] + k[1190]*y[IDX_C3NI];
    data[449] = 0.0 - k[1325]*y[IDX_C2II];
    data[450] = 0.0 + k[233]*y[IDX_C2I];
    data[451] = 0.0 + k[38]*y[IDX_C2I];
    data[452] = 0.0 - k[1444]*y[IDX_C2II] - k[1445]*y[IDX_C2II];
    data[453] = 0.0 - k[271]*y[IDX_C2II] - k[1394]*y[IDX_C2II];
    data[454] = 0.0 - k[34]*y[IDX_C2II];
    data[455] = 0.0 - k[1490]*y[IDX_C2II];
    data[456] = 0.0 + k[306]*y[IDX_C2I];
    data[457] = 0.0 - k[649]*y[IDX_C2II];
    data[458] = 0.0 + k[39]*y[IDX_C2I];
    data[459] = 0.0 - k[339]*y[IDX_C2II];
    data[460] = 0.0 + k[329]*y[IDX_C2I];
    data[461] = 0.0 - k[35]*y[IDX_C2II] - k[650]*y[IDX_C2II];
    data[462] = 0.0 + k[2323] + k[2324] + k[2325] + k[2326];
    data[463] = 0.0 + k[1586]*y[IDX_CH2I];
    data[464] = 0.0 - k[607]*y[IDX_C2HI] + k[623]*y[IDX_HC3NI];
    data[465] = 0.0 - k[47]*y[IDX_CNII] - k[48]*y[IDX_COII] -
        k[49]*y[IDX_N2II] - k[112]*y[IDX_HII] - k[155]*y[IDX_H2II] -
        k[177]*y[IDX_H2OII] - k[234]*y[IDX_NII] - k[308]*y[IDX_OII] -
        k[330]*y[IDX_OHII] - k[373] - k[374] - k[607]*y[IDX_CII] -
        k[677]*y[IDX_COII] - k[678]*y[IDX_H2COII] - k[679]*y[IDX_HCNII] -
        k[680]*y[IDX_HCOII] - k[681]*y[IDX_HNOII] - k[682]*y[IDX_N2HII] -
        k[683]*y[IDX_O2HII] - k[684]*y[IDX_SiII] - k[706]*y[IDX_CHII] -
        k[824]*y[IDX_CH5II] - k[884]*y[IDX_HII] - k[911]*y[IDX_H2II] -
        k[977]*y[IDX_H2OII] - k[1025]*y[IDX_H3II] - k[1186]*y[IDX_HeII] -
        k[1187]*y[IDX_HeII] - k[1188]*y[IDX_HeII] - k[1348]*y[IDX_NHII] -
        k[1375]*y[IDX_NH2II] - k[1465]*y[IDX_OII] - k[1518]*y[IDX_OHII] -
        k[1578]*y[IDX_HCNI] - k[1579]*y[IDX_HNCI] - k[1580]*y[IDX_NCCNI] -
        k[1581]*y[IDX_O2I] - k[1721]*y[IDX_H2I] - k[1795]*y[IDX_NI] -
        k[1875]*y[IDX_OI] - k[1970] - k[1971] - k[2100]*y[IDX_CNI] - k[2224];
    data[466] = 0.0 + k[40]*y[IDX_NOI] + k[41]*y[IDX_SI];
    data[467] = 0.0 + k[368] + k[1737]*y[IDX_HI] + k[1927]*y[IDX_OHI] +
        k[1965];
    data[468] = 0.0 + k[462]*y[IDX_EM] + k[666]*y[IDX_CH3CNI] +
        k[667]*y[IDX_H2SI] + k[668]*y[IDX_HCNI] + k[669]*y[IDX_HNCI] +
        k[991]*y[IDX_H2OI] + k[1396]*y[IDX_NH2I] + k[1419]*y[IDX_NH3I];
    data[469] = 0.0 + k[378] + k[1975];
    data[470] = 0.0 - k[706]*y[IDX_C2HI];
    data[471] = 0.0 + k[1586]*y[IDX_CI];
    data[472] = 0.0 + k[666]*y[IDX_C2H2II];
    data[473] = 0.0 - k[824]*y[IDX_C2HI];
    data[474] = 0.0 - k[2100]*y[IDX_C2HI];
    data[475] = 0.0 - k[47]*y[IDX_C2HI];
    data[476] = 0.0 - k[48]*y[IDX_C2HI] - k[677]*y[IDX_C2HI];
    data[477] = 0.0 + k[462]*y[IDX_C2H2II];
    data[478] = 0.0 + k[1737]*y[IDX_C2H2I];
    data[479] = 0.0 - k[112]*y[IDX_C2HI] - k[884]*y[IDX_C2HI];
    data[480] = 0.0 - k[1721]*y[IDX_C2HI];
    data[481] = 0.0 - k[155]*y[IDX_C2HI] - k[911]*y[IDX_C2HI];
    data[482] = 0.0 - k[678]*y[IDX_C2HI];
    data[483] = 0.0 + k[991]*y[IDX_C2H2II];
    data[484] = 0.0 - k[177]*y[IDX_C2HI] - k[977]*y[IDX_C2HI];
    data[485] = 0.0 + k[667]*y[IDX_C2H2II];
    data[486] = 0.0 - k[1025]*y[IDX_C2HI];
    data[487] = 0.0 + k[407] + k[623]*y[IDX_CII] + k[2027];
    data[488] = 0.0 + k[668]*y[IDX_C2H2II] - k[1578]*y[IDX_C2HI];
    data[489] = 0.0 - k[679]*y[IDX_C2HI];
    data[490] = 0.0 - k[680]*y[IDX_C2HI];
    data[491] = 0.0 - k[1186]*y[IDX_C2HI] - k[1187]*y[IDX_C2HI] -
        k[1188]*y[IDX_C2HI];
    data[492] = 0.0 + k[669]*y[IDX_C2H2II] - k[1579]*y[IDX_C2HI];
    data[493] = 0.0 - k[681]*y[IDX_C2HI];
    data[494] = 0.0 - k[1795]*y[IDX_C2HI];
    data[495] = 0.0 - k[234]*y[IDX_C2HI];
    data[496] = 0.0 - k[49]*y[IDX_C2HI];
    data[497] = 0.0 - k[682]*y[IDX_C2HI];
    data[498] = 0.0 - k[1580]*y[IDX_C2HI];
    data[499] = 0.0 - k[1348]*y[IDX_C2HI];
    data[500] = 0.0 + k[1396]*y[IDX_C2H2II];
    data[501] = 0.0 - k[1375]*y[IDX_C2HI];
    data[502] = 0.0 + k[1419]*y[IDX_C2H2II];
    data[503] = 0.0 + k[40]*y[IDX_C2HII];
    data[504] = 0.0 - k[1875]*y[IDX_C2HI];
    data[505] = 0.0 - k[308]*y[IDX_C2HI] - k[1465]*y[IDX_C2HI];
    data[506] = 0.0 - k[1581]*y[IDX_C2HI];
    data[507] = 0.0 - k[683]*y[IDX_C2HI];
    data[508] = 0.0 + k[1927]*y[IDX_C2H2I];
    data[509] = 0.0 - k[330]*y[IDX_C2HI] - k[1518]*y[IDX_C2HI];
    data[510] = 0.0 + k[41]*y[IDX_C2HII];
    data[511] = 0.0 - k[684]*y[IDX_C2HI];
    data[512] = 0.0 - k[685]*y[IDX_C2HII] + k[687]*y[IDX_CH2II] +
        k[688]*y[IDX_CH3II];
    data[513] = 0.0 + k[608]*y[IDX_CH2I] + k[609]*y[IDX_CH3I];
    data[514] = 0.0 + k[651]*y[IDX_H2COII] + k[652]*y[IDX_HCNII] +
        k[653]*y[IDX_HCOII] + k[654]*y[IDX_HNOII] + k[655]*y[IDX_N2HII] +
        k[657]*y[IDX_O2HII] + k[823]*y[IDX_CH5II] + k[909]*y[IDX_H2II] +
        k[976]*y[IDX_H2OII] + k[1022]*y[IDX_H3II] + k[1080]*y[IDX_H3OII] +
        k[1345]*y[IDX_NHII] + k[1374]*y[IDX_NH2II] + k[1517]*y[IDX_OHII];
    data[515] = 0.0 + k[648]*y[IDX_HCOI] + k[803]*y[IDX_CH4I] +
        k[935]*y[IDX_H2I] + k[990]*y[IDX_H2OI] + k[1444]*y[IDX_NHI];
    data[516] = 0.0 + k[47]*y[IDX_CNII] + k[48]*y[IDX_COII] +
        k[49]*y[IDX_N2II] + k[112]*y[IDX_HII] + k[155]*y[IDX_H2II] +
        k[177]*y[IDX_H2OII] + k[234]*y[IDX_NII] + k[308]*y[IDX_OII] +
        k[330]*y[IDX_OHII] + k[374] + k[1971];
    data[517] = 0.0 - k[40]*y[IDX_NOI] - k[41]*y[IDX_SI] - k[459]*y[IDX_EM]
        - k[460]*y[IDX_EM] - k[660]*y[IDX_H2COI] - k[661]*y[IDX_HCNI] -
        k[662]*y[IDX_HCNI] - k[663]*y[IDX_HCOI] - k[664]*y[IDX_HNCI] -
        k[685]*y[IDX_CI] - k[754]*y[IDX_CH2I] - k[805]*y[IDX_CH4I] -
        k[838]*y[IDX_CHI] - k[936]*y[IDX_H2I] - k[1326]*y[IDX_NI] -
        k[1327]*y[IDX_NI] - k[1395]*y[IDX_NH2I] - k[1418]*y[IDX_NH3I] -
        k[1491]*y[IDX_OI] - k[1963] - k[2242];
    data[518] = 0.0 + k[1179]*y[IDX_HeII];
    data[519] = 0.0 + k[1181]*y[IDX_HeII];
    data[520] = 0.0 + k[1183]*y[IDX_HeII];
    data[521] = 0.0 + k[1191]*y[IDX_HeII];
    data[522] = 0.0 - k[838]*y[IDX_C2HII];
    data[523] = 0.0 + k[707]*y[IDX_CH2I];
    data[524] = 0.0 + k[608]*y[IDX_CII] + k[707]*y[IDX_CHII] -
        k[754]*y[IDX_C2HII];
    data[525] = 0.0 + k[687]*y[IDX_CI];
    data[526] = 0.0 + k[609]*y[IDX_CII];
    data[527] = 0.0 + k[688]*y[IDX_CI];
    data[528] = 0.0 + k[803]*y[IDX_C2II] - k[805]*y[IDX_C2HII];
    data[529] = 0.0 + k[823]*y[IDX_C2I];
    data[530] = 0.0 + k[47]*y[IDX_C2HI];
    data[531] = 0.0 + k[48]*y[IDX_C2HI];
    data[532] = 0.0 - k[459]*y[IDX_C2HII] - k[460]*y[IDX_C2HII];
    data[533] = 0.0 + k[112]*y[IDX_C2HI];
    data[534] = 0.0 + k[935]*y[IDX_C2II] - k[936]*y[IDX_C2HII];
    data[535] = 0.0 + k[155]*y[IDX_C2HI] + k[909]*y[IDX_C2I];
    data[536] = 0.0 - k[660]*y[IDX_C2HII];
    data[537] = 0.0 + k[651]*y[IDX_C2I];
    data[538] = 0.0 + k[990]*y[IDX_C2II];
    data[539] = 0.0 + k[177]*y[IDX_C2HI] + k[976]*y[IDX_C2I];
    data[540] = 0.0 + k[1022]*y[IDX_C2I];
    data[541] = 0.0 + k[1080]*y[IDX_C2I];
    data[542] = 0.0 + k[1230]*y[IDX_HeII];
    data[543] = 0.0 - k[661]*y[IDX_C2HII] - k[662]*y[IDX_C2HII];
    data[544] = 0.0 + k[652]*y[IDX_C2I];
    data[545] = 0.0 + k[648]*y[IDX_C2II] - k[663]*y[IDX_C2HII];
    data[546] = 0.0 + k[653]*y[IDX_C2I];
    data[547] = 0.0 + k[1179]*y[IDX_C2H2I] + k[1181]*y[IDX_C2H3I] +
        k[1183]*y[IDX_C2H4I] + k[1191]*y[IDX_C4HI] + k[1230]*y[IDX_HC3NI];
    data[548] = 0.0 - k[664]*y[IDX_C2HII];
    data[549] = 0.0 + k[654]*y[IDX_C2I];
    data[550] = 0.0 - k[1326]*y[IDX_C2HII] - k[1327]*y[IDX_C2HII];
    data[551] = 0.0 + k[234]*y[IDX_C2HI];
    data[552] = 0.0 + k[49]*y[IDX_C2HI];
    data[553] = 0.0 + k[655]*y[IDX_C2I];
    data[554] = 0.0 + k[1444]*y[IDX_C2II];
    data[555] = 0.0 + k[1345]*y[IDX_C2I];
    data[556] = 0.0 - k[1395]*y[IDX_C2HII];
    data[557] = 0.0 + k[1374]*y[IDX_C2I];
    data[558] = 0.0 - k[1418]*y[IDX_C2HII];
    data[559] = 0.0 - k[40]*y[IDX_C2HII];
    data[560] = 0.0 - k[1491]*y[IDX_C2HII];
    data[561] = 0.0 + k[308]*y[IDX_C2HI];
    data[562] = 0.0 + k[657]*y[IDX_C2I];
    data[563] = 0.0 + k[330]*y[IDX_C2HI] + k[1517]*y[IDX_C2I];
    data[564] = 0.0 - k[41]*y[IDX_C2HII];
    data[565] = 0.0 + k[2327] + k[2328] + k[2329] + k[2330];
    data[566] = 0.0 + k[1588]*y[IDX_CH3I];
    data[567] = 0.0 + k[611]*y[IDX_CH3CCHI];
    data[568] = 0.0 - k[1570]*y[IDX_C2H2I];
    data[569] = 0.0 + k[1721]*y[IDX_H2I];
    data[570] = 0.0 - k[46]*y[IDX_HCNII] - k[75]*y[IDX_CH4II] -
        k[111]*y[IDX_HII] - k[154]*y[IDX_H2II] - k[176]*y[IDX_H2OII] -
        k[208]*y[IDX_HeII] - k[307]*y[IDX_OII] - k[321]*y[IDX_O2II] - k[367] -
        k[368] - k[675]*y[IDX_C2N2II] - k[1178]*y[IDX_HeII] -
        k[1179]*y[IDX_HeII] - k[1180]*y[IDX_HeII] - k[1482]*y[IDX_O2II] -
        k[1556]*y[IDX_SOII] - k[1557]*y[IDX_SOII] - k[1558]*y[IDX_SOII] -
        k[1570]*y[IDX_C2I] - k[1574]*y[IDX_NOI] - k[1575]*y[IDX_SiI] -
        k[1674]*y[IDX_CHI] - k[1701]*y[IDX_CNI] - k[1737]*y[IDX_HI] -
        k[1868]*y[IDX_OI] - k[1927]*y[IDX_OHI] - k[1928]*y[IDX_OHI] -
        k[1929]*y[IDX_OHI] - k[1964] - k[1965] - k[2298];
    data[571] = 0.0 + k[42]*y[IDX_H2COI] + k[43]*y[IDX_H2SI] +
        k[44]*y[IDX_HCOI] + k[45]*y[IDX_NOI] + k[220]*y[IDX_MgI] +
        k[282]*y[IDX_NH3I];
    data[572] = 0.0 + k[369] + k[1577]*y[IDX_O2I] + k[1646]*y[IDX_CH3I] +
        k[1738]*y[IDX_HI] + k[1791]*y[IDX_NI] + k[1930]*y[IDX_OHI] + k[1966];
    data[573] = 0.0 + k[370] + k[1967];
    data[574] = 0.0 - k[675]*y[IDX_C2H2I];
    data[575] = 0.0 - k[1674]*y[IDX_C2H2I];
    data[576] = 0.0 + k[1618]*y[IDX_CH2I] + k[1618]*y[IDX_CH2I] +
        k[1619]*y[IDX_CH2I] + k[1619]*y[IDX_CH2I];
    data[577] = 0.0 + k[1588]*y[IDX_CI] + k[1646]*y[IDX_C2H3I];
    data[578] = 0.0 + k[611]*y[IDX_CII];
    data[579] = 0.0 - k[75]*y[IDX_C2H2I];
    data[580] = 0.0 - k[1701]*y[IDX_C2H2I];
    data[581] = 0.0 - k[1737]*y[IDX_C2H2I] + k[1738]*y[IDX_C2H3I];
    data[582] = 0.0 - k[111]*y[IDX_C2H2I];
    data[583] = 0.0 + k[1721]*y[IDX_C2HI];
    data[584] = 0.0 - k[154]*y[IDX_C2H2I];
    data[585] = 0.0 + k[42]*y[IDX_C2H2II];
    data[586] = 0.0 - k[176]*y[IDX_C2H2I];
    data[587] = 0.0 + k[43]*y[IDX_C2H2II];
    data[588] = 0.0 - k[46]*y[IDX_C2H2I];
    data[589] = 0.0 + k[44]*y[IDX_C2H2II];
    data[590] = 0.0 - k[208]*y[IDX_C2H2I] - k[1178]*y[IDX_C2H2I] -
        k[1179]*y[IDX_C2H2I] - k[1180]*y[IDX_C2H2I];
    data[591] = 0.0 + k[220]*y[IDX_C2H2II];
    data[592] = 0.0 + k[1791]*y[IDX_C2H3I];
    data[593] = 0.0 + k[282]*y[IDX_C2H2II];
    data[594] = 0.0 + k[45]*y[IDX_C2H2II] - k[1574]*y[IDX_C2H2I];
    data[595] = 0.0 - k[1868]*y[IDX_C2H2I];
    data[596] = 0.0 - k[307]*y[IDX_C2H2I];
    data[597] = 0.0 + k[1577]*y[IDX_C2H3I];
    data[598] = 0.0 - k[321]*y[IDX_C2H2I] - k[1482]*y[IDX_C2H2I];
    data[599] = 0.0 - k[1927]*y[IDX_C2H2I] - k[1928]*y[IDX_C2H2I] -
        k[1929]*y[IDX_C2H2I] + k[1930]*y[IDX_C2H3I];
    data[600] = 0.0 - k[1575]*y[IDX_C2H2I];
    data[601] = 0.0 - k[1556]*y[IDX_C2H2I] - k[1557]*y[IDX_C2H2I] -
        k[1558]*y[IDX_C2H2I];
    data[602] = 0.0 + k[610]*y[IDX_CH3I] + k[611]*y[IDX_CH3CCHI] +
        k[614]*y[IDX_CH4I];
    data[603] = 0.0 + k[1412]*y[IDX_NH3II];
    data[604] = 0.0 + k[804]*y[IDX_CH4I];
    data[605] = 0.0 + k[678]*y[IDX_H2COII] + k[679]*y[IDX_HCNII] +
        k[680]*y[IDX_HCOII] + k[681]*y[IDX_HNOII] + k[682]*y[IDX_N2HII] +
        k[683]*y[IDX_O2HII] + k[824]*y[IDX_CH5II] + k[911]*y[IDX_H2II] +
        k[977]*y[IDX_H2OII] + k[1025]*y[IDX_H3II] + k[1348]*y[IDX_NHII] +
        k[1375]*y[IDX_NH2II] + k[1518]*y[IDX_OHII];
    data[606] = 0.0 + k[661]*y[IDX_HCNI] + k[663]*y[IDX_HCOI] +
        k[805]*y[IDX_CH4I] + k[936]*y[IDX_H2I];
    data[607] = 0.0 + k[46]*y[IDX_HCNII] + k[75]*y[IDX_CH4II] +
        k[111]*y[IDX_HII] + k[154]*y[IDX_H2II] + k[176]*y[IDX_H2OII] +
        k[208]*y[IDX_HeII] + k[307]*y[IDX_OII] + k[321]*y[IDX_O2II] + k[367] +
        k[675]*y[IDX_C2N2II] + k[1964];
    data[608] = 0.0 - k[42]*y[IDX_H2COI] - k[43]*y[IDX_H2SI] -
        k[44]*y[IDX_HCOI] - k[45]*y[IDX_NOI] - k[220]*y[IDX_MgI] -
        k[282]*y[IDX_NH3I] - k[461]*y[IDX_EM] - k[462]*y[IDX_EM] -
        k[463]*y[IDX_EM] - k[665]*y[IDX_CH3CNI] - k[666]*y[IDX_CH3CNI] -
        k[667]*y[IDX_H2SI] - k[668]*y[IDX_HCNI] - k[669]*y[IDX_HNCI] -
        k[670]*y[IDX_SiI] - k[671]*y[IDX_SiH4I] - k[672]*y[IDX_SiH4I] -
        k[673]*y[IDX_SiH4I] - k[674]*y[IDX_SiH4I] - k[806]*y[IDX_CH4I] -
        k[991]*y[IDX_H2OI] - k[1328]*y[IDX_NI] - k[1329]*y[IDX_NI] -
        k[1330]*y[IDX_NI] - k[1396]*y[IDX_NH2I] - k[1419]*y[IDX_NH3I] -
        k[1492]*y[IDX_OI] - k[2277];
    data[609] = 0.0 + k[882]*y[IDX_HII] + k[1182]*y[IDX_HeII];
    data[610] = 0.0 + k[883]*y[IDX_HII] + k[910]*y[IDX_H2II] +
        k[1184]*y[IDX_HeII] + k[1464]*y[IDX_OII];
    data[611] = 0.0 + k[675]*y[IDX_C2H2I];
    data[612] = 0.0 + k[839]*y[IDX_CH3II];
    data[613] = 0.0 + k[711]*y[IDX_CH4I];
    data[614] = 0.0 + k[610]*y[IDX_CII];
    data[615] = 0.0 + k[839]*y[IDX_CHI];
    data[616] = 0.0 + k[611]*y[IDX_CII];
    data[617] = 0.0 - k[665]*y[IDX_C2H2II] - k[666]*y[IDX_C2H2II];
    data[618] = 0.0 + k[614]*y[IDX_CII] + k[711]*y[IDX_CHII] +
        k[804]*y[IDX_C2II] + k[805]*y[IDX_C2HII] - k[806]*y[IDX_C2H2II];
    data[619] = 0.0 + k[75]*y[IDX_C2H2I];
    data[620] = 0.0 + k[824]*y[IDX_C2HI];
    data[621] = 0.0 - k[461]*y[IDX_C2H2II] - k[462]*y[IDX_C2H2II] -
        k[463]*y[IDX_C2H2II];
    data[622] = 0.0 + k[111]*y[IDX_C2H2I] + k[882]*y[IDX_C2H3I] +
        k[883]*y[IDX_C2H4I];
    data[623] = 0.0 + k[936]*y[IDX_C2HII];
    data[624] = 0.0 + k[154]*y[IDX_C2H2I] + k[910]*y[IDX_C2H4I] +
        k[911]*y[IDX_C2HI];
    data[625] = 0.0 - k[42]*y[IDX_C2H2II];
    data[626] = 0.0 + k[678]*y[IDX_C2HI];
    data[627] = 0.0 - k[991]*y[IDX_C2H2II];
    data[628] = 0.0 + k[176]*y[IDX_C2H2I] + k[977]*y[IDX_C2HI];
    data[629] = 0.0 - k[43]*y[IDX_C2H2II] - k[667]*y[IDX_C2H2II];
    data[630] = 0.0 + k[1025]*y[IDX_C2HI];
    data[631] = 0.0 + k[661]*y[IDX_C2HII] - k[668]*y[IDX_C2H2II];
    data[632] = 0.0 + k[46]*y[IDX_C2H2I] + k[679]*y[IDX_C2HI];
    data[633] = 0.0 - k[44]*y[IDX_C2H2II] + k[663]*y[IDX_C2HII];
    data[634] = 0.0 + k[680]*y[IDX_C2HI];
    data[635] = 0.0 + k[208]*y[IDX_C2H2I] + k[1182]*y[IDX_C2H3I] +
        k[1184]*y[IDX_C2H4I];
    data[636] = 0.0 - k[669]*y[IDX_C2H2II];
    data[637] = 0.0 + k[681]*y[IDX_C2HI];
    data[638] = 0.0 - k[220]*y[IDX_C2H2II];
    data[639] = 0.0 - k[1328]*y[IDX_C2H2II] - k[1329]*y[IDX_C2H2II] -
        k[1330]*y[IDX_C2H2II];
    data[640] = 0.0 + k[682]*y[IDX_C2HI];
    data[641] = 0.0 + k[1348]*y[IDX_C2HI];
    data[642] = 0.0 - k[1396]*y[IDX_C2H2II];
    data[643] = 0.0 + k[1375]*y[IDX_C2HI];
    data[644] = 0.0 - k[282]*y[IDX_C2H2II] - k[1419]*y[IDX_C2H2II];
    data[645] = 0.0 + k[1412]*y[IDX_C2I];
    data[646] = 0.0 - k[45]*y[IDX_C2H2II];
    data[647] = 0.0 - k[1492]*y[IDX_C2H2II];
    data[648] = 0.0 + k[307]*y[IDX_C2H2I] + k[1464]*y[IDX_C2H4I];
    data[649] = 0.0 + k[321]*y[IDX_C2H2I];
    data[650] = 0.0 + k[683]*y[IDX_C2HI];
    data[651] = 0.0 + k[1518]*y[IDX_C2HI];
    data[652] = 0.0 - k[670]*y[IDX_C2H2II];
    data[653] = 0.0 - k[671]*y[IDX_C2H2II] - k[672]*y[IDX_C2H2II] -
        k[673]*y[IDX_C2H2II] - k[674]*y[IDX_C2H2II];
    data[654] = 0.0 + k[2339] + k[2340] + k[2341] + k[2342];
    data[655] = 0.0 - k[1582]*y[IDX_C2H3I];
    data[656] = 0.0 + k[606]*y[IDX_C2H5OHI];
    data[657] = 0.0 + k[674]*y[IDX_SiH4I];
    data[658] = 0.0 - k[369] - k[882]*y[IDX_HII] - k[1181]*y[IDX_HeII] -
        k[1182]*y[IDX_HeII] - k[1576]*y[IDX_O2I] - k[1577]*y[IDX_O2I] -
        k[1582]*y[IDX_CI] - k[1646]*y[IDX_CH3I] - k[1738]*y[IDX_HI] -
        k[1791]*y[IDX_NI] - k[1869]*y[IDX_OI] - k[1930]*y[IDX_OHI] - k[1966] -
        k[2146];
    data[659] = 0.0 + k[1702]*y[IDX_CNI] + k[1870]*y[IDX_OI];
    data[660] = 0.0 + k[371] + k[1968];
    data[661] = 0.0 + k[606]*y[IDX_CII];
    data[662] = 0.0 + k[1620]*y[IDX_CH2I] + k[1620]*y[IDX_CH2I];
    data[663] = 0.0 - k[1646]*y[IDX_C2H3I];
    data[664] = 0.0 + k[1702]*y[IDX_C2H4I];
    data[665] = 0.0 - k[1738]*y[IDX_C2H3I];
    data[666] = 0.0 - k[882]*y[IDX_C2H3I];
    data[667] = 0.0 - k[1181]*y[IDX_C2H3I] - k[1182]*y[IDX_C2H3I];
    data[668] = 0.0 - k[1791]*y[IDX_C2H3I];
    data[669] = 0.0 - k[1869]*y[IDX_C2H3I] + k[1870]*y[IDX_C2H4I];
    data[670] = 0.0 - k[1576]*y[IDX_C2H3I] - k[1577]*y[IDX_C2H3I];
    data[671] = 0.0 - k[1930]*y[IDX_C2H3I];
    data[672] = 0.0 + k[674]*y[IDX_C2H2II];
    data[673] = 0.0 + k[2351] + k[2352] + k[2353] + k[2354];
    data[674] = 0.0 + k[671]*y[IDX_SiH4I] + k[673]*y[IDX_SiH4I];
    data[675] = 0.0 - k[370] - k[676]*y[IDX_SII] - k[774]*y[IDX_CH3II] -
        k[883]*y[IDX_HII] - k[910]*y[IDX_H2II] - k[1183]*y[IDX_HeII] -
        k[1184]*y[IDX_HeII] - k[1185]*y[IDX_HeII] - k[1464]*y[IDX_OII] -
        k[1559]*y[IDX_SOII] - k[1560]*y[IDX_SOII] - k[1561]*y[IDX_SOII] -
        k[1675]*y[IDX_CHI] - k[1702]*y[IDX_CNI] - k[1792]*y[IDX_NI] -
        k[1870]*y[IDX_OI] - k[1871]*y[IDX_OI] - k[1872]*y[IDX_OI] -
        k[1873]*y[IDX_OI] - k[1967] - k[2121]*y[IDX_H3OII] - k[2300];
    data[676] = 0.0 + k[1793]*y[IDX_NI] + k[1931]*y[IDX_OHI];
    data[677] = 0.0 + k[464]*y[IDX_EM];
    data[678] = 0.0 - k[1675]*y[IDX_C2H4I] + k[1676]*y[IDX_CH4I];
    data[679] = 0.0 + k[1647]*y[IDX_CH3I] + k[1647]*y[IDX_CH3I];
    data[680] = 0.0 - k[774]*y[IDX_C2H4I] + k[775]*y[IDX_CH3CNI];
    data[681] = 0.0 + k[775]*y[IDX_CH3II];
    data[682] = 0.0 + k[1676]*y[IDX_CHI];
    data[683] = 0.0 - k[1702]*y[IDX_C2H4I];
    data[684] = 0.0 + k[464]*y[IDX_C2H5OH2II];
    data[685] = 0.0 - k[883]*y[IDX_C2H4I];
    data[686] = 0.0 - k[910]*y[IDX_C2H4I];
    data[687] = 0.0 - k[2121]*y[IDX_C2H4I];
    data[688] = 0.0 - k[1183]*y[IDX_C2H4I] - k[1184]*y[IDX_C2H4I] -
        k[1185]*y[IDX_C2H4I];
    data[689] = 0.0 - k[1792]*y[IDX_C2H4I] + k[1793]*y[IDX_C2H5I];
    data[690] = 0.0 - k[1870]*y[IDX_C2H4I] - k[1871]*y[IDX_C2H4I] -
        k[1872]*y[IDX_C2H4I] - k[1873]*y[IDX_C2H4I];
    data[691] = 0.0 - k[1464]*y[IDX_C2H4I];
    data[692] = 0.0 + k[1931]*y[IDX_C2H5I];
    data[693] = 0.0 - k[676]*y[IDX_C2H4I];
    data[694] = 0.0 + k[671]*y[IDX_C2H2II] + k[673]*y[IDX_C2H2II];
    data[695] = 0.0 - k[1559]*y[IDX_C2H4I] - k[1560]*y[IDX_C2H4I] -
        k[1561]*y[IDX_C2H4I];
    data[696] = 0.0 + k[2359] + k[2360] + k[2361] + k[2362];
    data[697] = 0.0 - k[1583]*y[IDX_C2H5I];
    data[698] = 0.0 + k[672]*y[IDX_SiH4I];
    data[699] = 0.0 - k[371] - k[1583]*y[IDX_CI] - k[1793]*y[IDX_NI] -
        k[1794]*y[IDX_NI] - k[1874]*y[IDX_OI] - k[1931]*y[IDX_OHI] - k[1968] -
        k[2301];
    data[700] = 0.0 + k[372] + k[1563]*y[IDX_SiII] + k[1969];
    data[701] = 0.0 + k[465]*y[IDX_EM] + k[466]*y[IDX_EM];
    data[702] = 0.0 + k[1648]*y[IDX_CH3I] + k[1648]*y[IDX_CH3I];
    data[703] = 0.0 + k[465]*y[IDX_C2H5OH2II] + k[466]*y[IDX_C2H5OH2II];
    data[704] = 0.0 - k[1793]*y[IDX_C2H5I] - k[1794]*y[IDX_C2H5I];
    data[705] = 0.0 - k[1874]*y[IDX_C2H5I];
    data[706] = 0.0 - k[1931]*y[IDX_C2H5I];
    data[707] = 0.0 + k[1563]*y[IDX_C2H5OHI];
    data[708] = 0.0 + k[672]*y[IDX_C2H2II];
    data[709] = 0.0 + k[2439] + k[2440] + k[2441] + k[2442];
    data[710] = 0.0 - k[606]*y[IDX_C2H5OHI];
    data[711] = 0.0 - k[372] - k[606]*y[IDX_CII] - k[1023]*y[IDX_H3II] -
        k[1024]*y[IDX_H3II] - k[1081]*y[IDX_H3OII] - k[1136]*y[IDX_HCOII] -
        k[1164]*y[IDX_HCSII] - k[1563]*y[IDX_SiII] - k[1969] - k[2150];
    data[712] = 0.0 + k[467]*y[IDX_EM] + k[1420]*y[IDX_NH3I];
    data[713] = 0.0 + k[467]*y[IDX_C2H5OH2II];
    data[714] = 0.0 - k[1023]*y[IDX_C2H5OHI] - k[1024]*y[IDX_C2H5OHI];
    data[715] = 0.0 - k[1081]*y[IDX_C2H5OHI];
    data[716] = 0.0 - k[1136]*y[IDX_C2H5OHI];
    data[717] = 0.0 - k[1164]*y[IDX_C2H5OHI];
    data[718] = 0.0 + k[1420]*y[IDX_C2H5OH2II];
    data[719] = 0.0 - k[1563]*y[IDX_C2H5OHI];
    data[720] = 0.0 + k[2121]*y[IDX_H3OII];
    data[721] = 0.0 + k[1081]*y[IDX_H3OII] + k[1136]*y[IDX_HCOII] +
        k[1164]*y[IDX_HCSII];
    data[722] = 0.0 - k[464]*y[IDX_EM] - k[465]*y[IDX_EM] - k[466]*y[IDX_EM]
        - k[467]*y[IDX_EM] - k[1420]*y[IDX_NH3I] - k[2151];
    data[723] = 0.0 - k[464]*y[IDX_C2H5OH2II] - k[465]*y[IDX_C2H5OH2II] -
        k[466]*y[IDX_C2H5OH2II] - k[467]*y[IDX_C2H5OH2II];
    data[724] = 0.0 + k[1081]*y[IDX_C2H5OHI] + k[2121]*y[IDX_C2H4I];
    data[725] = 0.0 + k[1136]*y[IDX_C2H5OHI];
    data[726] = 0.0 + k[1164]*y[IDX_C2H5OHI];
    data[727] = 0.0 - k[1420]*y[IDX_C2H5OH2II];
    data[728] = 0.0 - k[1584]*y[IDX_C2NI] + k[1593]*y[IDX_H2CNI] +
        k[1598]*y[IDX_NCCNI];
    data[729] = 0.0 + k[1795]*y[IDX_NI];
    data[730] = 0.0 - k[113]*y[IDX_HII] - k[375] - k[376] -
        k[1026]*y[IDX_H3II] - k[1189]*y[IDX_HeII] - k[1584]*y[IDX_CI] -
        k[1796]*y[IDX_NI] - k[1876]*y[IDX_OI] - k[1972] - k[1973] - k[2159];
    data[731] = 0.0 + k[470]*y[IDX_EM];
    data[732] = 0.0 + k[472]*y[IDX_EM];
    data[733] = 0.0 + k[1798]*y[IDX_NI] + k[1877]*y[IDX_OI];
    data[734] = 0.0 + k[475]*y[IDX_EM];
    data[735] = 0.0 + k[470]*y[IDX_C2N2II] + k[472]*y[IDX_C2NHII] +
        k[475]*y[IDX_C4NII];
    data[736] = 0.0 - k[113]*y[IDX_C2NI];
    data[737] = 0.0 + k[1593]*y[IDX_CI];
    data[738] = 0.0 - k[1026]*y[IDX_C2NI];
    data[739] = 0.0 - k[1189]*y[IDX_C2NI];
    data[740] = 0.0 + k[1795]*y[IDX_C2HI] - k[1796]*y[IDX_C2NI] +
        k[1798]*y[IDX_C3NI] + k[1818]*y[IDX_NCCNI];
    data[741] = 0.0 + k[1598]*y[IDX_CI] + k[1818]*y[IDX_NI];
    data[742] = 0.0 - k[1876]*y[IDX_C2NI] + k[1877]*y[IDX_C3NI];
    data[743] = 0.0 + k[20]*y[IDX_NCCNI] + k[623]*y[IDX_HC3NI] +
        k[627]*y[IDX_HNCI];
    data[744] = 0.0 + k[1346]*y[IDX_NHII];
    data[745] = 0.0 + k[1445]*y[IDX_NHI];
    data[746] = 0.0 + k[1326]*y[IDX_NI];
    data[747] = 0.0 + k[1328]*y[IDX_NI];
    data[748] = 0.0 + k[113]*y[IDX_HII];
    data[749] = 0.0 - k[468]*y[IDX_EM] - k[469]*y[IDX_EM] -
        k[992]*y[IDX_H2OI] - k[993]*y[IDX_H2OI] - k[1018]*y[IDX_H2SI] -
        k[1421]*y[IDX_NH3I] - k[2160];
    data[750] = 0.0 + k[713]*y[IDX_CNI] + k[723]*y[IDX_HCNI];
    data[751] = 0.0 + k[713]*y[IDX_CHII];
    data[752] = 0.0 - k[468]*y[IDX_C2NII] - k[469]*y[IDX_C2NII];
    data[753] = 0.0 + k[113]*y[IDX_C2NI];
    data[754] = 0.0 - k[992]*y[IDX_C2NII] - k[993]*y[IDX_C2NII];
    data[755] = 0.0 - k[1018]*y[IDX_C2NII];
    data[756] = 0.0 + k[623]*y[IDX_CII] + k[1229]*y[IDX_HeII];
    data[757] = 0.0 + k[723]*y[IDX_CHII];
    data[758] = 0.0 + k[1229]*y[IDX_HC3NI];
    data[759] = 0.0 + k[627]*y[IDX_CII];
    data[760] = 0.0 + k[1326]*y[IDX_C2HII] + k[1328]*y[IDX_C2H2II];
    data[761] = 0.0 + k[1304]*y[IDX_NCCNI];
    data[762] = 0.0 + k[20]*y[IDX_CII] + k[1304]*y[IDX_NII];
    data[763] = 0.0 + k[1445]*y[IDX_C2II];
    data[764] = 0.0 + k[1346]*y[IDX_C2I];
    data[765] = 0.0 - k[1421]*y[IDX_C2NII];
    data[766] = 0.0 - k[675]*y[IDX_C2N2II];
    data[767] = 0.0 - k[470]*y[IDX_EM] - k[471]*y[IDX_EM] -
        k[675]*y[IDX_C2H2I] - k[1097]*y[IDX_HI] - k[1118]*y[IDX_HCNI] - k[2163];
    data[768] = 0.0 + k[866]*y[IDX_HCNI];
    data[769] = 0.0 - k[470]*y[IDX_C2N2II] - k[471]*y[IDX_C2N2II];
    data[770] = 0.0 - k[1097]*y[IDX_C2N2II];
    data[771] = 0.0 + k[866]*y[IDX_CNII] - k[1118]*y[IDX_C2N2II];
    data[772] = 0.0 + k[1305]*y[IDX_NCCNI];
    data[773] = 0.0 + k[1305]*y[IDX_NII] + k[2046];
    data[774] = 0.0 + k[1394]*y[IDX_NH2I];
    data[775] = 0.0 + k[1329]*y[IDX_NI];
    data[776] = 0.0 + k[1026]*y[IDX_H3II];
    data[777] = 0.0 + k[992]*y[IDX_H2OI];
    data[778] = 0.0 - k[472]*y[IDX_EM] - k[2162];
    data[779] = 0.0 + k[724]*y[IDX_HCNI];
    data[780] = 0.0 - k[472]*y[IDX_C2NHII];
    data[781] = 0.0 + k[992]*y[IDX_C2NII];
    data[782] = 0.0 + k[1026]*y[IDX_C2NI];
    data[783] = 0.0 + k[724]*y[IDX_CHII];
    data[784] = 0.0 + k[1329]*y[IDX_C2H2II];
    data[785] = 0.0 + k[1394]*y[IDX_C2II];
    data[786] = 0.0 + k[685]*y[IDX_C2HII];
    data[787] = 0.0 + k[607]*y[IDX_C2HI] + k[624]*y[IDX_HC3NI];
    data[788] = 0.0 + k[647]*y[IDX_C2II] + k[705]*y[IDX_CHII];
    data[789] = 0.0 + k[647]*y[IDX_C2I] + k[837]*y[IDX_CHI];
    data[790] = 0.0 + k[607]*y[IDX_CII] + k[706]*y[IDX_CHII];
    data[791] = 0.0 + k[685]*y[IDX_CI];
    data[792] = 0.0 - k[473]*y[IDX_EM] - k[1119]*y[IDX_HCNI] - k[2292];
    data[793] = 0.0 + k[837]*y[IDX_C2II];
    data[794] = 0.0 + k[705]*y[IDX_C2I] + k[706]*y[IDX_C2HI];
    data[795] = 0.0 + k[1197]*y[IDX_HeII];
    data[796] = 0.0 - k[473]*y[IDX_C3II];
    data[797] = 0.0 + k[624]*y[IDX_CII];
    data[798] = 0.0 - k[1119]*y[IDX_C3II];
    data[799] = 0.0 + k[1197]*y[IDX_CH3CCHI];
    data[800] = 0.0 + k[2399] + k[2400] + k[2401] + k[2402];
    data[801] = 0.0 + k[1582]*y[IDX_C2H3I] - k[1585]*y[IDX_C3H2I];
    data[802] = 0.0 + k[1674]*y[IDX_CHI];
    data[803] = 0.0 + k[1582]*y[IDX_CI];
    data[804] = 0.0 - k[1585]*y[IDX_CI] - k[1797]*y[IDX_NI] - k[2199];
    data[805] = 0.0 + k[1674]*y[IDX_C2H2I];
    data[806] = 0.0 - k[1797]*y[IDX_C3H2I];
    data[807] = 0.0 + k[665]*y[IDX_CH3CNI] + k[806]*y[IDX_CH4I];
    data[808] = 0.0 + k[774]*y[IDX_CH3II];
    data[809] = 0.0 - k[474]*y[IDX_EM] - k[2148];
    data[810] = 0.0 + k[774]*y[IDX_C2H4I];
    data[811] = 0.0 + k[1082]*y[IDX_H3OII] + k[1137]*y[IDX_HCOII];
    data[812] = 0.0 + k[665]*y[IDX_C2H2II];
    data[813] = 0.0 + k[806]*y[IDX_C2H2II];
    data[814] = 0.0 - k[474]*y[IDX_C3H5II];
    data[815] = 0.0 + k[1082]*y[IDX_CH3CCHI];
    data[816] = 0.0 + k[1137]*y[IDX_CH3CCHI];
    data[817] = 0.0 + k[1571]*y[IDX_HCNI];
    data[818] = 0.0 - k[377] - k[1190]*y[IDX_HeII] - k[1798]*y[IDX_NI] -
        k[1877]*y[IDX_OI] - k[1974] - k[2161];
    data[819] = 0.0 + k[1800]*y[IDX_NI] + k[1878]*y[IDX_OI];
    data[820] = 0.0 + k[476]*y[IDX_EM];
    data[821] = 0.0 + k[476]*y[IDX_C4NII];
    data[822] = 0.0 + k[1571]*y[IDX_C2I];
    data[823] = 0.0 - k[1190]*y[IDX_C3NI];
    data[824] = 0.0 - k[1798]*y[IDX_C3NI] + k[1800]*y[IDX_C4NI];
    data[825] = 0.0 - k[1877]*y[IDX_C3NI] + k[1878]*y[IDX_C4NI];
    data[826] = 0.0 + k[2463] + k[2464] + k[2465] + k[2466];
    data[827] = 0.0 + k[1585]*y[IDX_C3H2I];
    data[828] = 0.0 + k[1570]*y[IDX_C2H2I];
    data[829] = 0.0 + k[1570]*y[IDX_C2I];
    data[830] = 0.0 + k[1585]*y[IDX_CI];
    data[831] = 0.0 - k[378] - k[1191]*y[IDX_HeII] - k[1799]*y[IDX_NI] -
        k[1975] - k[2200];
    data[832] = 0.0 - k[1191]*y[IDX_C4HI];
    data[833] = 0.0 - k[1799]*y[IDX_C4HI];
    data[834] = 0.0 + k[2491] + k[2492] + k[2493] + k[2494];
    data[835] = 0.0 + k[1799]*y[IDX_NI];
    data[836] = 0.0 - k[1800]*y[IDX_NI] - k[1878]*y[IDX_OI] - k[2166];
    data[837] = 0.0 + k[1799]*y[IDX_C4HI] - k[1800]*y[IDX_C4NI];
    data[838] = 0.0 - k[1878]*y[IDX_C4NI];
    data[839] = 0.0 + k[625]*y[IDX_HC3NI];
    data[840] = 0.0 + k[1119]*y[IDX_HCNI];
    data[841] = 0.0 - k[475]*y[IDX_EM] - k[476]*y[IDX_EM] -
        k[994]*y[IDX_H2OI] - k[2165];
    data[842] = 0.0 - k[475]*y[IDX_C4NII] - k[476]*y[IDX_C4NII];
    data[843] = 0.0 - k[994]*y[IDX_C4NII];
    data[844] = 0.0 + k[625]*y[IDX_CII];
    data[845] = 0.0 + k[1119]*y[IDX_C3II];
    data[846] = 0.0 + k[1587]*y[IDX_CH2I] + k[1587]*y[IDX_CH2I] -
        k[1589]*y[IDX_CHI] + k[1594]*y[IDX_HCOI] + k[1596]*y[IDX_HSI] +
        k[1601]*y[IDX_NH2I] + k[1603]*y[IDX_NHI] + k[1612]*y[IDX_OHI] +
        k[1722]*y[IDX_H2I] + k[2123]*y[IDX_HI];
    data[847] = 0.0 - k[15]*y[IDX_CHI] + k[612]*y[IDX_CH3OHI] -
        k[615]*y[IDX_CHI] + k[618]*y[IDX_H2COI];
    data[848] = 0.0 + k[1736]*y[IDX_HI];
    data[849] = 0.0 - k[82]*y[IDX_CHI] - k[837]*y[IDX_CHI];
    data[850] = 0.0 + k[1188]*y[IDX_HeII] + k[1465]*y[IDX_OII] +
        k[1875]*y[IDX_OI];
    data[851] = 0.0 + k[460]*y[IDX_EM] - k[838]*y[IDX_CHI];
    data[852] = 0.0 + k[1180]*y[IDX_HeII] - k[1674]*y[IDX_CHI];
    data[853] = 0.0 + k[463]*y[IDX_EM] + k[463]*y[IDX_EM] +
        k[1492]*y[IDX_OI];
    data[854] = 0.0 - k[1675]*y[IDX_CHI];
    data[855] = 0.0 - k[0]*y[IDX_OI] - k[2]*y[IDX_H2I] - k[9]*y[IDX_HI] -
        k[15]*y[IDX_CII] - k[82]*y[IDX_C2II] - k[83]*y[IDX_CNII] -
        k[84]*y[IDX_COII] - k[85]*y[IDX_H2COII] - k[86]*y[IDX_H2OII] -
        k[87]*y[IDX_NII] - k[88]*y[IDX_N2II] - k[89]*y[IDX_NH2II] -
        k[90]*y[IDX_OII] - k[91]*y[IDX_O2II] - k[92]*y[IDX_OHII] -
        k[117]*y[IDX_HII] - k[158]*y[IDX_H2II] - k[211]*y[IDX_HeII] - k[391] -
        k[615]*y[IDX_CII] - k[712]*y[IDX_CHII] - k[837]*y[IDX_C2II] -
        k[838]*y[IDX_C2HII] - k[839]*y[IDX_CH3II] - k[840]*y[IDX_CH5II] -
        k[841]*y[IDX_COII] - k[842]*y[IDX_H2COII] - k[843]*y[IDX_H2OII] -
        k[844]*y[IDX_H3COII] - k[845]*y[IDX_H3OII] - k[846]*y[IDX_HCNII] -
        k[847]*y[IDX_HCNHII] - k[848]*y[IDX_HCNHII] - k[849]*y[IDX_HCOII] -
        k[850]*y[IDX_HNOII] - k[851]*y[IDX_HSII] - k[852]*y[IDX_NII] -
        k[853]*y[IDX_N2HII] - k[854]*y[IDX_NHII] - k[855]*y[IDX_NH2II] -
        k[856]*y[IDX_NH3II] - k[857]*y[IDX_OII] - k[858]*y[IDX_O2II] -
        k[859]*y[IDX_O2HII] - k[860]*y[IDX_OHII] - k[861]*y[IDX_SII] -
        k[862]*y[IDX_SiII] - k[863]*y[IDX_SiHII] - k[864]*y[IDX_SiOII] -
        k[916]*y[IDX_H2II] - k[1034]*y[IDX_H3II] - k[1206]*y[IDX_HeII] -
        k[1589]*y[IDX_CI] - k[1674]*y[IDX_C2H2I] - k[1675]*y[IDX_C2H4I] -
        k[1676]*y[IDX_CH4I] - k[1677]*y[IDX_CO2I] - k[1678]*y[IDX_H2COI] -
        k[1679]*y[IDX_HCOI] - k[1680]*y[IDX_HNOI] - k[1681]*y[IDX_N2I] -
        k[1682]*y[IDX_NI] - k[1683]*y[IDX_NI] - k[1684]*y[IDX_NOI] -
        k[1685]*y[IDX_NOI] - k[1686]*y[IDX_NOI] - k[1687]*y[IDX_O2I] -
        k[1688]*y[IDX_O2I] - k[1689]*y[IDX_O2I] - k[1690]*y[IDX_O2I] -
        k[1691]*y[IDX_O2HI] - k[1692]*y[IDX_O2HI] - k[1693]*y[IDX_OI] -
        k[1694]*y[IDX_OI] - k[1695]*y[IDX_OCSI] - k[1696]*y[IDX_OHI] -
        k[1697]*y[IDX_SI] - k[1698]*y[IDX_SI] - k[1699]*y[IDX_SOI] -
        k[1700]*y[IDX_SOI] - k[1725]*y[IDX_H2I] - k[1743]*y[IDX_HI] - k[1999] -
        k[2000] - k[2115]*y[IDX_H2I] - k[2210];
    data[856] = 0.0 + k[55]*y[IDX_HCOI] + k[56]*y[IDX_MgI] +
        k[57]*y[IDX_NH3I] + k[58]*y[IDX_NOI] + k[59]*y[IDX_SI] +
        k[60]*y[IDX_SiI] - k[712]*y[IDX_CHI];
    data[857] = 0.0 + k[382] + k[756]*y[IDX_COII] + k[1587]*y[IDX_CI] +
        k[1587]*y[IDX_CI] + k[1621]*y[IDX_CH2I] + k[1621]*y[IDX_CH2I] +
        k[1623]*y[IDX_CNI] + k[1640]*y[IDX_OI] + k[1642]*y[IDX_OHI] +
        k[1739]*y[IDX_HI] + k[1803]*y[IDX_NI] + k[1982];
    data[858] = 0.0 + k[480]*y[IDX_EM] + k[745]*y[IDX_H2SI] +
        k[748]*y[IDX_NH3I] + k[1980];
    data[859] = 0.0 + k[386] + k[1988];
    data[860] = 0.0 + k[482]*y[IDX_EM] + k[483]*y[IDX_EM] -
        k[839]*y[IDX_CHI];
    data[861] = 0.0 + k[612]*y[IDX_CII];
    data[862] = 0.0 - k[1676]*y[IDX_CHI] + k[1998];
    data[863] = 0.0 + k[497]*y[IDX_EM] - k[840]*y[IDX_CHI];
    data[864] = 0.0 + k[1623]*y[IDX_CH2I];
    data[865] = 0.0 - k[83]*y[IDX_CHI];
    data[866] = 0.0 - k[84]*y[IDX_CHI] + k[756]*y[IDX_CH2I] -
        k[841]*y[IDX_CHI];
    data[867] = 0.0 - k[1677]*y[IDX_CHI];
    data[868] = 0.0 + k[460]*y[IDX_C2HII] + k[463]*y[IDX_C2H2II] +
        k[463]*y[IDX_C2H2II] + k[480]*y[IDX_CH2II] + k[482]*y[IDX_CH3II] +
        k[483]*y[IDX_CH3II] + k[497]*y[IDX_CH5II] + k[522]*y[IDX_H3COII] +
        k[546]*y[IDX_HCSII];
    data[869] = 0.0 - k[9]*y[IDX_CHI] + k[1736]*y[IDX_C2I] +
        k[1739]*y[IDX_CH2I] - k[1743]*y[IDX_CHI] + k[2123]*y[IDX_CI];
    data[870] = 0.0 - k[117]*y[IDX_CHI];
    data[871] = 0.0 - k[2]*y[IDX_CHI] + k[1722]*y[IDX_CI] -
        k[1725]*y[IDX_CHI] - k[2115]*y[IDX_CHI];
    data[872] = 0.0 - k[158]*y[IDX_CHI] - k[916]*y[IDX_CHI];
    data[873] = 0.0 + k[618]*y[IDX_CII] - k[1678]*y[IDX_CHI];
    data[874] = 0.0 - k[85]*y[IDX_CHI] - k[842]*y[IDX_CHI];
    data[875] = 0.0 - k[86]*y[IDX_CHI] - k[843]*y[IDX_CHI];
    data[876] = 0.0 + k[745]*y[IDX_CH2II];
    data[877] = 0.0 - k[1034]*y[IDX_CHI];
    data[878] = 0.0 + k[522]*y[IDX_EM] - k[844]*y[IDX_CHI];
    data[879] = 0.0 - k[845]*y[IDX_CHI];
    data[880] = 0.0 + k[1229]*y[IDX_HeII];
    data[881] = 0.0 + k[1232]*y[IDX_HeII] + k[1475]*y[IDX_OII];
    data[882] = 0.0 - k[846]*y[IDX_CHI];
    data[883] = 0.0 - k[847]*y[IDX_CHI] - k[848]*y[IDX_CHI];
    data[884] = 0.0 + k[55]*y[IDX_CHII] + k[1594]*y[IDX_CI] -
        k[1679]*y[IDX_CHI];
    data[885] = 0.0 - k[849]*y[IDX_CHI];
    data[886] = 0.0 + k[546]*y[IDX_EM];
    data[887] = 0.0 - k[211]*y[IDX_CHI] + k[1180]*y[IDX_C2H2I] +
        k[1188]*y[IDX_C2HI] - k[1206]*y[IDX_CHI] + k[1229]*y[IDX_HC3NI] +
        k[1232]*y[IDX_HCNI];
    data[888] = 0.0 - k[1680]*y[IDX_CHI];
    data[889] = 0.0 - k[850]*y[IDX_CHI];
    data[890] = 0.0 + k[1596]*y[IDX_CI];
    data[891] = 0.0 - k[851]*y[IDX_CHI];
    data[892] = 0.0 + k[56]*y[IDX_CHII];
    data[893] = 0.0 - k[1682]*y[IDX_CHI] - k[1683]*y[IDX_CHI] +
        k[1803]*y[IDX_CH2I];
    data[894] = 0.0 - k[87]*y[IDX_CHI] - k[852]*y[IDX_CHI];
    data[895] = 0.0 - k[1681]*y[IDX_CHI];
    data[896] = 0.0 - k[88]*y[IDX_CHI];
    data[897] = 0.0 - k[853]*y[IDX_CHI];
    data[898] = 0.0 + k[1603]*y[IDX_CI];
    data[899] = 0.0 - k[854]*y[IDX_CHI];
    data[900] = 0.0 + k[1601]*y[IDX_CI];
    data[901] = 0.0 - k[89]*y[IDX_CHI] - k[855]*y[IDX_CHI];
    data[902] = 0.0 + k[57]*y[IDX_CHII] + k[748]*y[IDX_CH2II];
    data[903] = 0.0 - k[856]*y[IDX_CHI];
    data[904] = 0.0 + k[58]*y[IDX_CHII] - k[1684]*y[IDX_CHI] -
        k[1685]*y[IDX_CHI] - k[1686]*y[IDX_CHI];
    data[905] = 0.0 - k[0]*y[IDX_CHI] + k[1492]*y[IDX_C2H2II] +
        k[1640]*y[IDX_CH2I] - k[1693]*y[IDX_CHI] - k[1694]*y[IDX_CHI] +
        k[1875]*y[IDX_C2HI];
    data[906] = 0.0 - k[90]*y[IDX_CHI] - k[857]*y[IDX_CHI] +
        k[1465]*y[IDX_C2HI] + k[1475]*y[IDX_HCNI];
    data[907] = 0.0 - k[1687]*y[IDX_CHI] - k[1688]*y[IDX_CHI] -
        k[1689]*y[IDX_CHI] - k[1690]*y[IDX_CHI];
    data[908] = 0.0 - k[91]*y[IDX_CHI] - k[858]*y[IDX_CHI];
    data[909] = 0.0 - k[1691]*y[IDX_CHI] - k[1692]*y[IDX_CHI];
    data[910] = 0.0 - k[859]*y[IDX_CHI];
    data[911] = 0.0 - k[1695]*y[IDX_CHI];
    data[912] = 0.0 + k[1612]*y[IDX_CI] + k[1642]*y[IDX_CH2I] -
        k[1696]*y[IDX_CHI];
    data[913] = 0.0 - k[92]*y[IDX_CHI] - k[860]*y[IDX_CHI];
    data[914] = 0.0 + k[59]*y[IDX_CHII] - k[1697]*y[IDX_CHI] -
        k[1698]*y[IDX_CHI];
    data[915] = 0.0 - k[861]*y[IDX_CHI];
    data[916] = 0.0 + k[60]*y[IDX_CHII];
    data[917] = 0.0 - k[862]*y[IDX_CHI];
    data[918] = 0.0 - k[863]*y[IDX_CHI];
    data[919] = 0.0 - k[864]*y[IDX_CHI];
    data[920] = 0.0 - k[1699]*y[IDX_CHI] - k[1700]*y[IDX_CHI];
    data[921] = 0.0 - k[686]*y[IDX_CHII] + k[689]*y[IDX_CH5II] +
        k[690]*y[IDX_H2OII] + k[693]*y[IDX_HCNII] + k[694]*y[IDX_HCOII] +
        k[695]*y[IDX_HCO2II] + k[696]*y[IDX_HNOII] + k[698]*y[IDX_N2HII] +
        k[699]*y[IDX_NHII] + k[701]*y[IDX_O2HII] + k[702]*y[IDX_OHII] +
        k[912]*y[IDX_H2II] + k[1027]*y[IDX_H3II];
    data[922] = 0.0 + k[15]*y[IDX_CHI] + k[626]*y[IDX_HCOI] +
        k[934]*y[IDX_H2I] + k[2122]*y[IDX_HI];
    data[923] = 0.0 - k[705]*y[IDX_CHII];
    data[924] = 0.0 + k[82]*y[IDX_CHI];
    data[925] = 0.0 - k[706]*y[IDX_CHII] + k[1187]*y[IDX_HeII];
    data[926] = 0.0 + k[1327]*y[IDX_NI];
    data[927] = 0.0 + k[1180]*y[IDX_HeII];
    data[928] = 0.0 + k[1330]*y[IDX_NI];
    data[929] = 0.0 + k[15]*y[IDX_CII] + k[82]*y[IDX_C2II] +
        k[83]*y[IDX_CNII] + k[84]*y[IDX_COII] + k[85]*y[IDX_H2COII] +
        k[86]*y[IDX_H2OII] + k[87]*y[IDX_NII] + k[88]*y[IDX_N2II] +
        k[89]*y[IDX_NH2II] + k[90]*y[IDX_OII] + k[91]*y[IDX_O2II] +
        k[92]*y[IDX_OHII] + k[117]*y[IDX_HII] + k[158]*y[IDX_H2II] +
        k[211]*y[IDX_HeII] - k[712]*y[IDX_CHII] + k[2000];
    data[930] = 0.0 - k[55]*y[IDX_HCOI] - k[56]*y[IDX_MgI] -
        k[57]*y[IDX_NH3I] - k[58]*y[IDX_NOI] - k[59]*y[IDX_SI] -
        k[60]*y[IDX_SiI] - k[380] - k[477]*y[IDX_EM] - k[686]*y[IDX_CI] -
        k[705]*y[IDX_C2I] - k[706]*y[IDX_C2HI] - k[707]*y[IDX_CH2I] -
        k[708]*y[IDX_CH3OHI] - k[709]*y[IDX_CH3OHI] - k[710]*y[IDX_CH3OHI] -
        k[711]*y[IDX_CH4I] - k[712]*y[IDX_CHI] - k[713]*y[IDX_CNI] -
        k[714]*y[IDX_CO2I] - k[715]*y[IDX_H2COI] - k[716]*y[IDX_H2COI] -
        k[717]*y[IDX_H2COI] - k[718]*y[IDX_H2OI] - k[719]*y[IDX_H2OI] -
        k[720]*y[IDX_H2OI] - k[721]*y[IDX_H2SI] - k[722]*y[IDX_H2SI] -
        k[723]*y[IDX_HCNI] - k[724]*y[IDX_HCNI] - k[725]*y[IDX_HCNI] -
        k[726]*y[IDX_HCOI] - k[727]*y[IDX_HNCI] - k[728]*y[IDX_NI] -
        k[729]*y[IDX_NH2I] - k[730]*y[IDX_NH3I] - k[731]*y[IDX_NHI] -
        k[732]*y[IDX_O2I] - k[733]*y[IDX_O2I] - k[734]*y[IDX_O2I] -
        k[735]*y[IDX_OI] - k[736]*y[IDX_OCSI] - k[737]*y[IDX_OCSI] -
        k[738]*y[IDX_OHI] - k[739]*y[IDX_SI] - k[740]*y[IDX_SI] -
        k[937]*y[IDX_H2I] - k[1098]*y[IDX_HI] - k[1977] - k[2231];
    data[931] = 0.0 - k[707]*y[IDX_CHII] + k[885]*y[IDX_HII] +
        k[1193]*y[IDX_HeII];
    data[932] = 0.0 + k[1099]*y[IDX_HI] + k[1979];
    data[933] = 0.0 + k[1196]*y[IDX_HeII];
    data[934] = 0.0 + k[1984];
    data[935] = 0.0 - k[708]*y[IDX_CHII] - k[709]*y[IDX_CHII] -
        k[710]*y[IDX_CHII];
    data[936] = 0.0 - k[711]*y[IDX_CHII] + k[1202]*y[IDX_HeII];
    data[937] = 0.0 + k[689]*y[IDX_CI];
    data[938] = 0.0 - k[713]*y[IDX_CHII];
    data[939] = 0.0 + k[83]*y[IDX_CHI];
    data[940] = 0.0 + k[84]*y[IDX_CHI];
    data[941] = 0.0 - k[714]*y[IDX_CHII];
    data[942] = 0.0 - k[477]*y[IDX_CHII];
    data[943] = 0.0 - k[1098]*y[IDX_CHII] + k[1099]*y[IDX_CH2II] +
        k[2122]*y[IDX_CII];
    data[944] = 0.0 + k[117]*y[IDX_CHI] + k[885]*y[IDX_CH2I];
    data[945] = 0.0 + k[934]*y[IDX_CII] - k[937]*y[IDX_CHII];
    data[946] = 0.0 + k[158]*y[IDX_CHI] + k[912]*y[IDX_CI];
    data[947] = 0.0 - k[715]*y[IDX_CHII] - k[716]*y[IDX_CHII] -
        k[717]*y[IDX_CHII];
    data[948] = 0.0 + k[85]*y[IDX_CHI];
    data[949] = 0.0 - k[718]*y[IDX_CHII] - k[719]*y[IDX_CHII] -
        k[720]*y[IDX_CHII];
    data[950] = 0.0 + k[86]*y[IDX_CHI] + k[690]*y[IDX_CI];
    data[951] = 0.0 - k[721]*y[IDX_CHII] - k[722]*y[IDX_CHII];
    data[952] = 0.0 + k[1027]*y[IDX_CI];
    data[953] = 0.0 - k[723]*y[IDX_CHII] - k[724]*y[IDX_CHII] -
        k[725]*y[IDX_CHII] + k[1234]*y[IDX_HeII];
    data[954] = 0.0 + k[693]*y[IDX_CI];
    data[955] = 0.0 - k[55]*y[IDX_CHII] + k[626]*y[IDX_CII] -
        k[726]*y[IDX_CHII] + k[1237]*y[IDX_HeII];
    data[956] = 0.0 + k[694]*y[IDX_CI];
    data[957] = 0.0 + k[695]*y[IDX_CI];
    data[958] = 0.0 + k[211]*y[IDX_CHI] + k[1180]*y[IDX_C2H2I] +
        k[1187]*y[IDX_C2HI] + k[1193]*y[IDX_CH2I] + k[1196]*y[IDX_CH3I] +
        k[1202]*y[IDX_CH4I] + k[1234]*y[IDX_HCNI] + k[1237]*y[IDX_HCOI];
    data[959] = 0.0 - k[727]*y[IDX_CHII];
    data[960] = 0.0 + k[696]*y[IDX_CI];
    data[961] = 0.0 - k[56]*y[IDX_CHII];
    data[962] = 0.0 - k[728]*y[IDX_CHII] + k[1327]*y[IDX_C2HII] +
        k[1330]*y[IDX_C2H2II];
    data[963] = 0.0 + k[87]*y[IDX_CHI];
    data[964] = 0.0 + k[88]*y[IDX_CHI];
    data[965] = 0.0 + k[698]*y[IDX_CI];
    data[966] = 0.0 - k[731]*y[IDX_CHII];
    data[967] = 0.0 + k[699]*y[IDX_CI];
    data[968] = 0.0 - k[729]*y[IDX_CHII];
    data[969] = 0.0 + k[89]*y[IDX_CHI];
    data[970] = 0.0 - k[57]*y[IDX_CHII] - k[730]*y[IDX_CHII];
    data[971] = 0.0 - k[58]*y[IDX_CHII];
    data[972] = 0.0 - k[735]*y[IDX_CHII];
    data[973] = 0.0 + k[90]*y[IDX_CHI];
    data[974] = 0.0 - k[732]*y[IDX_CHII] - k[733]*y[IDX_CHII] -
        k[734]*y[IDX_CHII];
    data[975] = 0.0 + k[91]*y[IDX_CHI];
    data[976] = 0.0 + k[701]*y[IDX_CI];
    data[977] = 0.0 - k[736]*y[IDX_CHII] - k[737]*y[IDX_CHII];
    data[978] = 0.0 - k[738]*y[IDX_CHII];
    data[979] = 0.0 + k[92]*y[IDX_CHI] + k[702]*y[IDX_CI];
    data[980] = 0.0 - k[59]*y[IDX_CHII] - k[739]*y[IDX_CHII] -
        k[740]*y[IDX_CHII];
    data[981] = 0.0 - k[60]*y[IDX_CHII];
    data[982] = 0.0 - k[1586]*y[IDX_CH2I] - k[1587]*y[IDX_CH2I] +
        k[2113]*y[IDX_H2I];
    data[983] = 0.0 - k[14]*y[IDX_CH2I] - k[608]*y[IDX_CH2I];
    data[984] = 0.0 - k[62]*y[IDX_CH2I] + k[804]*y[IDX_CH4I];
    data[985] = 0.0 - k[754]*y[IDX_CH2I];
    data[986] = 0.0 + k[1868]*y[IDX_OI];
    data[987] = 0.0 + k[1185]*y[IDX_HeII] + k[1872]*y[IDX_OI];
    data[988] = 0.0 + k[1678]*y[IDX_H2COI] + k[1679]*y[IDX_HCOI] +
        k[1680]*y[IDX_HNOI] + k[1692]*y[IDX_O2HI] + k[1725]*y[IDX_H2I];
    data[989] = 0.0 - k[707]*y[IDX_CH2I] + k[710]*y[IDX_CH3OHI] +
        k[717]*y[IDX_H2COI];
    data[990] = 0.0 - k[14]*y[IDX_CII] - k[62]*y[IDX_C2II] -
        k[63]*y[IDX_CNII] - k[64]*y[IDX_COII] - k[65]*y[IDX_H2COII] -
        k[66]*y[IDX_H2OII] - k[67]*y[IDX_N2II] - k[68]*y[IDX_NH2II] -
        k[69]*y[IDX_OII] - k[70]*y[IDX_O2II] - k[71]*y[IDX_OHII] -
        k[114]*y[IDX_HII] - k[156]*y[IDX_H2II] - k[235]*y[IDX_NII] - k[381] -
        k[382] - k[608]*y[IDX_CII] - k[707]*y[IDX_CHII] - k[754]*y[IDX_C2HII] -
        k[755]*y[IDX_CH5II] - k[756]*y[IDX_COII] - k[757]*y[IDX_H2COII] -
        k[758]*y[IDX_H2OII] - k[759]*y[IDX_H3OII] - k[760]*y[IDX_HCNII] -
        k[761]*y[IDX_HCNHII] - k[762]*y[IDX_HCNHII] - k[763]*y[IDX_HCOII] -
        k[764]*y[IDX_HNOII] - k[765]*y[IDX_N2HII] - k[766]*y[IDX_NHII] -
        k[767]*y[IDX_NH2II] - k[768]*y[IDX_NH3II] - k[769]*y[IDX_O2II] -
        k[770]*y[IDX_O2HII] - k[771]*y[IDX_OHII] - k[772]*y[IDX_SII] -
        k[773]*y[IDX_SiOII] - k[885]*y[IDX_HII] - k[913]*y[IDX_H2II] -
        k[1028]*y[IDX_H3II] - k[1192]*y[IDX_HeII] - k[1193]*y[IDX_HeII] -
        k[1586]*y[IDX_CI] - k[1587]*y[IDX_CI] - k[1618]*y[IDX_CH2I] -
        k[1618]*y[IDX_CH2I] - k[1618]*y[IDX_CH2I] - k[1618]*y[IDX_CH2I] -
        k[1619]*y[IDX_CH2I] - k[1619]*y[IDX_CH2I] - k[1619]*y[IDX_CH2I] -
        k[1619]*y[IDX_CH2I] - k[1620]*y[IDX_CH2I] - k[1620]*y[IDX_CH2I] -
        k[1620]*y[IDX_CH2I] - k[1620]*y[IDX_CH2I] - k[1621]*y[IDX_CH2I] -
        k[1621]*y[IDX_CH2I] - k[1621]*y[IDX_CH2I] - k[1621]*y[IDX_CH2I] -
        k[1622]*y[IDX_CH4I] - k[1623]*y[IDX_CNI] - k[1624]*y[IDX_H2COI] -
        k[1625]*y[IDX_HCOI] - k[1626]*y[IDX_HNOI] - k[1627]*y[IDX_N2I] -
        k[1628]*y[IDX_NO2I] - k[1629]*y[IDX_NOI] - k[1630]*y[IDX_NOI] -
        k[1631]*y[IDX_NOI] - k[1632]*y[IDX_O2I] - k[1633]*y[IDX_O2I] -
        k[1634]*y[IDX_O2I] - k[1635]*y[IDX_O2I] - k[1636]*y[IDX_O2I] -
        k[1637]*y[IDX_OI] - k[1638]*y[IDX_OI] - k[1639]*y[IDX_OI] -
        k[1640]*y[IDX_OI] - k[1641]*y[IDX_OHI] - k[1642]*y[IDX_OHI] -
        k[1643]*y[IDX_OHI] - k[1644]*y[IDX_SI] - k[1645]*y[IDX_SI] -
        k[1723]*y[IDX_H2I] - k[1739]*y[IDX_HI] - k[1801]*y[IDX_NI] -
        k[1802]*y[IDX_NI] - k[1803]*y[IDX_NI] - k[1981] - k[1982] - k[2213];
    data[991] = 0.0 + k[61]*y[IDX_NOI];
    data[992] = 0.0 + k[383] + k[1194]*y[IDX_HeII] + k[1983];
    data[993] = 0.0 + k[384] + k[1649]*y[IDX_CH3I] + k[1649]*y[IDX_CH3I] +
        k[1650]*y[IDX_CNI] + k[1662]*y[IDX_O2I] + k[1668]*y[IDX_OHI] +
        k[1741]*y[IDX_HI] + k[1986];
    data[994] = 0.0 + k[481]*y[IDX_EM] + k[781]*y[IDX_NH3I];
    data[995] = 0.0 + k[710]*y[IDX_CHII];
    data[996] = 0.0 + k[486]*y[IDX_EM];
    data[997] = 0.0 + k[390] + k[804]*y[IDX_C2II] + k[820]*y[IDX_OHII] -
        k[1622]*y[IDX_CH2I] + k[1995];
    data[998] = 0.0 + k[491]*y[IDX_EM];
    data[999] = 0.0 + k[493]*y[IDX_EM] - k[755]*y[IDX_CH2I] +
        k[1495]*y[IDX_OI];
    data[1000] = 0.0 - k[1623]*y[IDX_CH2I] + k[1650]*y[IDX_CH3I];
    data[1001] = 0.0 - k[63]*y[IDX_CH2I];
    data[1002] = 0.0 - k[64]*y[IDX_CH2I] - k[756]*y[IDX_CH2I];
    data[1003] = 0.0 + k[481]*y[IDX_CH3II] + k[486]*y[IDX_CH3OH2II] +
        k[491]*y[IDX_CH4II] + k[493]*y[IDX_CH5II] + k[502]*y[IDX_H2COII] +
        k[521]*y[IDX_H3COII];
    data[1004] = 0.0 - k[1739]*y[IDX_CH2I] + k[1741]*y[IDX_CH3I] +
        k[1752]*y[IDX_HCOI];
    data[1005] = 0.0 - k[114]*y[IDX_CH2I] - k[885]*y[IDX_CH2I];
    data[1006] = 0.0 - k[1723]*y[IDX_CH2I] + k[1725]*y[IDX_CHI] +
        k[2113]*y[IDX_CI];
    data[1007] = 0.0 - k[156]*y[IDX_CH2I] - k[913]*y[IDX_CH2I];
    data[1008] = 0.0 + k[717]*y[IDX_CHII] + k[1299]*y[IDX_NII] -
        k[1624]*y[IDX_CH2I] + k[1678]*y[IDX_CHI];
    data[1009] = 0.0 - k[65]*y[IDX_CH2I] + k[502]*y[IDX_EM] -
        k[757]*y[IDX_CH2I];
    data[1010] = 0.0 + k[1220]*y[IDX_HeII];
    data[1011] = 0.0 - k[66]*y[IDX_CH2I] - k[758]*y[IDX_CH2I];
    data[1012] = 0.0 - k[1028]*y[IDX_CH2I];
    data[1013] = 0.0 + k[521]*y[IDX_EM];
    data[1014] = 0.0 - k[759]*y[IDX_CH2I];
    data[1015] = 0.0 - k[760]*y[IDX_CH2I];
    data[1016] = 0.0 - k[761]*y[IDX_CH2I] - k[762]*y[IDX_CH2I];
    data[1017] = 0.0 - k[1625]*y[IDX_CH2I] + k[1679]*y[IDX_CHI] +
        k[1752]*y[IDX_HI];
    data[1018] = 0.0 - k[763]*y[IDX_CH2I];
    data[1019] = 0.0 + k[1185]*y[IDX_C2H4I] - k[1192]*y[IDX_CH2I] -
        k[1193]*y[IDX_CH2I] + k[1194]*y[IDX_CH2COI] + k[1220]*y[IDX_H2CSI];
    data[1020] = 0.0 - k[1626]*y[IDX_CH2I] + k[1680]*y[IDX_CHI];
    data[1021] = 0.0 - k[764]*y[IDX_CH2I];
    data[1022] = 0.0 - k[1801]*y[IDX_CH2I] - k[1802]*y[IDX_CH2I] -
        k[1803]*y[IDX_CH2I];
    data[1023] = 0.0 - k[235]*y[IDX_CH2I] + k[1299]*y[IDX_H2COI];
    data[1024] = 0.0 - k[1627]*y[IDX_CH2I];
    data[1025] = 0.0 - k[67]*y[IDX_CH2I];
    data[1026] = 0.0 - k[765]*y[IDX_CH2I];
    data[1027] = 0.0 - k[766]*y[IDX_CH2I];
    data[1028] = 0.0 - k[68]*y[IDX_CH2I] - k[767]*y[IDX_CH2I];
    data[1029] = 0.0 + k[781]*y[IDX_CH3II];
    data[1030] = 0.0 - k[768]*y[IDX_CH2I];
    data[1031] = 0.0 + k[61]*y[IDX_CH2II] - k[1629]*y[IDX_CH2I] -
        k[1630]*y[IDX_CH2I] - k[1631]*y[IDX_CH2I];
    data[1032] = 0.0 - k[1628]*y[IDX_CH2I];
    data[1033] = 0.0 + k[1495]*y[IDX_CH5II] - k[1637]*y[IDX_CH2I] -
        k[1638]*y[IDX_CH2I] - k[1639]*y[IDX_CH2I] - k[1640]*y[IDX_CH2I] +
        k[1868]*y[IDX_C2H2I] + k[1872]*y[IDX_C2H4I];
    data[1034] = 0.0 - k[69]*y[IDX_CH2I];
    data[1035] = 0.0 - k[1632]*y[IDX_CH2I] - k[1633]*y[IDX_CH2I] -
        k[1634]*y[IDX_CH2I] - k[1635]*y[IDX_CH2I] - k[1636]*y[IDX_CH2I] +
        k[1662]*y[IDX_CH3I];
    data[1036] = 0.0 - k[70]*y[IDX_CH2I] - k[769]*y[IDX_CH2I];
    data[1037] = 0.0 + k[1692]*y[IDX_CHI];
    data[1038] = 0.0 - k[770]*y[IDX_CH2I];
    data[1039] = 0.0 - k[1641]*y[IDX_CH2I] - k[1642]*y[IDX_CH2I] -
        k[1643]*y[IDX_CH2I] + k[1668]*y[IDX_CH3I];
    data[1040] = 0.0 - k[71]*y[IDX_CH2I] - k[771]*y[IDX_CH2I] +
        k[820]*y[IDX_CH4I];
    data[1041] = 0.0 - k[1644]*y[IDX_CH2I] - k[1645]*y[IDX_CH2I];
    data[1042] = 0.0 - k[772]*y[IDX_CH2I];
    data[1043] = 0.0 - k[773]*y[IDX_CH2I];
    data[1044] = 0.0 - k[687]*y[IDX_CH2II];
    data[1045] = 0.0 + k[14]*y[IDX_CH2I] + k[617]*y[IDX_H2COI] +
        k[619]*y[IDX_H2CSI] + k[2112]*y[IDX_H2I];
    data[1046] = 0.0 + k[62]*y[IDX_CH2I];
    data[1047] = 0.0 + k[838]*y[IDX_CHI];
    data[1048] = 0.0 + k[1185]*y[IDX_HeII];
    data[1049] = 0.0 + k[838]*y[IDX_C2HII] + k[840]*y[IDX_CH5II] +
        k[842]*y[IDX_H2COII] + k[843]*y[IDX_H2OII] + k[844]*y[IDX_H3COII] +
        k[845]*y[IDX_H3OII] + k[846]*y[IDX_HCNII] + k[847]*y[IDX_HCNHII] +
        k[848]*y[IDX_HCNHII] + k[849]*y[IDX_HCOII] + k[850]*y[IDX_HNOII] +
        k[851]*y[IDX_HSII] + k[853]*y[IDX_N2HII] + k[854]*y[IDX_NHII] +
        k[855]*y[IDX_NH2II] + k[859]*y[IDX_O2HII] + k[860]*y[IDX_OHII] +
        k[863]*y[IDX_SiHII] + k[916]*y[IDX_H2II] + k[1034]*y[IDX_H3II];
    data[1050] = 0.0 + k[726]*y[IDX_HCOI] + k[937]*y[IDX_H2I];
    data[1051] = 0.0 + k[14]*y[IDX_CII] + k[62]*y[IDX_C2II] +
        k[63]*y[IDX_CNII] + k[64]*y[IDX_COII] + k[65]*y[IDX_H2COII] +
        k[66]*y[IDX_H2OII] + k[67]*y[IDX_N2II] + k[68]*y[IDX_NH2II] +
        k[69]*y[IDX_OII] + k[70]*y[IDX_O2II] + k[71]*y[IDX_OHII] +
        k[114]*y[IDX_HII] + k[156]*y[IDX_H2II] + k[235]*y[IDX_NII] + k[381] +
        k[1981];
    data[1052] = 0.0 - k[61]*y[IDX_NOI] - k[478]*y[IDX_EM] -
        k[479]*y[IDX_EM] - k[480]*y[IDX_EM] - k[687]*y[IDX_CI] -
        k[741]*y[IDX_CO2I] - k[742]*y[IDX_H2COI] - k[743]*y[IDX_H2OI] -
        k[744]*y[IDX_H2SI] - k[745]*y[IDX_H2SI] - k[746]*y[IDX_H2SI] -
        k[747]*y[IDX_HCOI] - k[748]*y[IDX_NH3I] - k[749]*y[IDX_O2I] -
        k[750]*y[IDX_OI] - k[751]*y[IDX_OCSI] - k[752]*y[IDX_OCSI] -
        k[753]*y[IDX_SI] - k[938]*y[IDX_H2I] - k[1099]*y[IDX_HI] -
        k[1331]*y[IDX_NI] - k[1978] - k[1979] - k[1980] - k[2237];
    data[1053] = 0.0 + k[1195]*y[IDX_HeII];
    data[1054] = 0.0 + k[1100]*y[IDX_HI] + k[1985];
    data[1055] = 0.0 + k[815]*y[IDX_N2II] + k[1203]*y[IDX_HeII];
    data[1056] = 0.0 + k[1993];
    data[1057] = 0.0 + k[840]*y[IDX_CHI];
    data[1058] = 0.0 + k[63]*y[IDX_CH2I];
    data[1059] = 0.0 + k[64]*y[IDX_CH2I];
    data[1060] = 0.0 - k[741]*y[IDX_CH2II];
    data[1061] = 0.0 - k[478]*y[IDX_CH2II] - k[479]*y[IDX_CH2II] -
        k[480]*y[IDX_CH2II];
    data[1062] = 0.0 - k[1099]*y[IDX_CH2II] + k[1100]*y[IDX_CH3II];
    data[1063] = 0.0 + k[114]*y[IDX_CH2I];
    data[1064] = 0.0 + k[937]*y[IDX_CHII] - k[938]*y[IDX_CH2II] +
        k[2112]*y[IDX_CII];
    data[1065] = 0.0 + k[156]*y[IDX_CH2I] + k[916]*y[IDX_CHI];
    data[1066] = 0.0 + k[617]*y[IDX_CII] - k[742]*y[IDX_CH2II] +
        k[1218]*y[IDX_HeII];
    data[1067] = 0.0 + k[65]*y[IDX_CH2I] + k[842]*y[IDX_CHI];
    data[1068] = 0.0 + k[619]*y[IDX_CII] + k[1221]*y[IDX_HeII];
    data[1069] = 0.0 - k[743]*y[IDX_CH2II];
    data[1070] = 0.0 + k[66]*y[IDX_CH2I] + k[843]*y[IDX_CHI];
    data[1071] = 0.0 - k[744]*y[IDX_CH2II] - k[745]*y[IDX_CH2II] -
        k[746]*y[IDX_CH2II];
    data[1072] = 0.0 + k[1034]*y[IDX_CHI];
    data[1073] = 0.0 + k[844]*y[IDX_CHI];
    data[1074] = 0.0 + k[845]*y[IDX_CHI];
    data[1075] = 0.0 + k[846]*y[IDX_CHI];
    data[1076] = 0.0 + k[847]*y[IDX_CHI] + k[848]*y[IDX_CHI];
    data[1077] = 0.0 + k[726]*y[IDX_CHII] - k[747]*y[IDX_CH2II];
    data[1078] = 0.0 + k[849]*y[IDX_CHI];
    data[1079] = 0.0 + k[1185]*y[IDX_C2H4I] + k[1195]*y[IDX_CH2COI] +
        k[1203]*y[IDX_CH4I] + k[1218]*y[IDX_H2COI] + k[1221]*y[IDX_H2CSI];
    data[1080] = 0.0 + k[850]*y[IDX_CHI];
    data[1081] = 0.0 + k[851]*y[IDX_CHI];
    data[1082] = 0.0 - k[1331]*y[IDX_CH2II];
    data[1083] = 0.0 + k[235]*y[IDX_CH2I];
    data[1084] = 0.0 + k[67]*y[IDX_CH2I] + k[815]*y[IDX_CH4I];
    data[1085] = 0.0 + k[853]*y[IDX_CHI];
    data[1086] = 0.0 + k[854]*y[IDX_CHI];
    data[1087] = 0.0 + k[68]*y[IDX_CH2I] + k[855]*y[IDX_CHI];
    data[1088] = 0.0 - k[748]*y[IDX_CH2II];
    data[1089] = 0.0 - k[61]*y[IDX_CH2II];
    data[1090] = 0.0 - k[750]*y[IDX_CH2II];
    data[1091] = 0.0 + k[69]*y[IDX_CH2I];
    data[1092] = 0.0 - k[749]*y[IDX_CH2II];
    data[1093] = 0.0 + k[70]*y[IDX_CH2I];
    data[1094] = 0.0 + k[859]*y[IDX_CHI];
    data[1095] = 0.0 - k[751]*y[IDX_CH2II] - k[752]*y[IDX_CH2II];
    data[1096] = 0.0 + k[71]*y[IDX_CH2I] + k[860]*y[IDX_CHI];
    data[1097] = 0.0 - k[753]*y[IDX_CH2II];
    data[1098] = 0.0 + k[863]*y[IDX_CHI];
    data[1099] = 0.0 + k[2415] + k[2416] + k[2417] + k[2418];
    data[1100] = 0.0 + k[1928]*y[IDX_OHI];
    data[1101] = 0.0 + k[1869]*y[IDX_OI];
    data[1102] = 0.0 + k[1871]*y[IDX_OI];
    data[1103] = 0.0 - k[383] - k[1194]*y[IDX_HeII] - k[1195]*y[IDX_HeII] -
        k[1740]*y[IDX_HI] - k[1983] - k[2147];
    data[1104] = 0.0 - k[1740]*y[IDX_CH2COI];
    data[1105] = 0.0 - k[1194]*y[IDX_CH2COI] - k[1195]*y[IDX_CH2COI];
    data[1106] = 0.0 + k[1869]*y[IDX_C2H3I] + k[1871]*y[IDX_C2H4I];
    data[1107] = 0.0 + k[1928]*y[IDX_C2H2I];
    data[1108] = 0.0 - k[1588]*y[IDX_CH3I];
    data[1109] = 0.0 - k[609]*y[IDX_CH3I] - k[610]*y[IDX_CH3I];
    data[1110] = 0.0 + k[803]*y[IDX_CH4I];
    data[1111] = 0.0 + k[805]*y[IDX_CH4I];
    data[1112] = 0.0 + k[1929]*y[IDX_OHI];
    data[1113] = 0.0 - k[1646]*y[IDX_CH3I];
    data[1114] = 0.0 + k[676]*y[IDX_SII] + k[1792]*y[IDX_NI] +
        k[1873]*y[IDX_OI];
    data[1115] = 0.0 + k[1794]*y[IDX_NI] + k[1874]*y[IDX_OI];
    data[1116] = 0.0 + k[2115]*y[IDX_H2I];
    data[1117] = 0.0 + k[1621]*y[IDX_CH2I] + k[1621]*y[IDX_CH2I] +
        k[1622]*y[IDX_CH4I] + k[1622]*y[IDX_CH4I] + k[1624]*y[IDX_H2COI] +
        k[1625]*y[IDX_HCOI] + k[1626]*y[IDX_HNOI] + k[1643]*y[IDX_OHI] +
        k[1723]*y[IDX_H2I];
    data[1118] = 0.0 + k[742]*y[IDX_H2COI];
    data[1119] = 0.0 + k[1740]*y[IDX_HI];
    data[1120] = 0.0 - k[115]*y[IDX_HII] - k[384] - k[385] - k[386] -
        k[609]*y[IDX_CII] - k[610]*y[IDX_CII] - k[790]*y[IDX_SII] -
        k[1029]*y[IDX_H3II] - k[1196]*y[IDX_HeII] - k[1588]*y[IDX_CI] -
        k[1646]*y[IDX_C2H3I] - k[1647]*y[IDX_CH3I] - k[1647]*y[IDX_CH3I] -
        k[1647]*y[IDX_CH3I] - k[1647]*y[IDX_CH3I] - k[1648]*y[IDX_CH3I] -
        k[1648]*y[IDX_CH3I] - k[1648]*y[IDX_CH3I] - k[1648]*y[IDX_CH3I] -
        k[1649]*y[IDX_CH3I] - k[1649]*y[IDX_CH3I] - k[1649]*y[IDX_CH3I] -
        k[1649]*y[IDX_CH3I] - k[1650]*y[IDX_CNI] - k[1651]*y[IDX_H2COI] -
        k[1652]*y[IDX_H2OI] - k[1653]*y[IDX_H2SI] - k[1654]*y[IDX_HCOI] -
        k[1655]*y[IDX_HNOI] - k[1656]*y[IDX_NH2I] - k[1657]*y[IDX_NH3I] -
        k[1658]*y[IDX_NO2I] - k[1659]*y[IDX_NOI] - k[1660]*y[IDX_O2I] -
        k[1661]*y[IDX_O2I] - k[1662]*y[IDX_O2I] - k[1663]*y[IDX_O2HI] -
        k[1664]*y[IDX_OI] - k[1665]*y[IDX_OI] - k[1666]*y[IDX_OHI] -
        k[1667]*y[IDX_OHI] - k[1668]*y[IDX_OHI] - k[1669]*y[IDX_SI] -
        k[1724]*y[IDX_H2I] - k[1741]*y[IDX_HI] - k[1804]*y[IDX_NI] -
        k[1805]*y[IDX_NI] - k[1806]*y[IDX_NI] - k[1986] - k[1987] - k[1988] -
        k[2109]*y[IDX_CNI] - k[2216];
    data[1121] = 0.0 + k[72]*y[IDX_HCOI] + k[73]*y[IDX_MgI] +
        k[74]*y[IDX_NOI] + k[2133]*y[IDX_EM];
    data[1122] = 0.0 + k[387] + k[1198]*y[IDX_HeII] + k[1989];
    data[1123] = 0.0 + k[485]*y[IDX_EM];
    data[1124] = 0.0 + k[389] + k[794]*y[IDX_CH4II] + k[1200]*y[IDX_HeII] +
        k[1291]*y[IDX_NII] + k[1564]*y[IDX_SiII] + k[1992];
    data[1125] = 0.0 + k[487]*y[IDX_EM] + k[488]*y[IDX_EM];
    data[1126] = 0.0 + k[795]*y[IDX_CH4II] + k[803]*y[IDX_C2II] +
        k[805]*y[IDX_C2HII] + k[807]*y[IDX_COII] + k[808]*y[IDX_CSII] +
        k[809]*y[IDX_H2COII] + k[810]*y[IDX_H2OII] + k[811]*y[IDX_HCNII] +
        k[818]*y[IDX_NH3II] + k[1205]*y[IDX_HeII] + k[1622]*y[IDX_CH2I] +
        k[1622]*y[IDX_CH2I] + k[1670]*y[IDX_CNI] + k[1671]*y[IDX_O2I] +
        k[1672]*y[IDX_OHI] + k[1673]*y[IDX_SI] + k[1742]*y[IDX_HI] +
        k[1833]*y[IDX_NH2I] + k[1839]*y[IDX_NHI] + k[1879]*y[IDX_OI] + k[1996];
    data[1127] = 0.0 + k[492]*y[IDX_EM] + k[794]*y[IDX_CH3OHI] +
        k[795]*y[IDX_CH4I] + k[796]*y[IDX_CO2I] + k[797]*y[IDX_COI] +
        k[798]*y[IDX_H2COI] + k[799]*y[IDX_H2OI] + k[800]*y[IDX_H2SI] +
        k[801]*y[IDX_NH3I] + k[802]*y[IDX_OCSI];
    data[1128] = 0.0 + k[494]*y[IDX_EM] + k[495]*y[IDX_EM];
    data[1129] = 0.0 - k[1650]*y[IDX_CH3I] + k[1670]*y[IDX_CH4I] -
        k[2109]*y[IDX_CH3I];
    data[1130] = 0.0 + k[797]*y[IDX_CH4II];
    data[1131] = 0.0 + k[807]*y[IDX_CH4I];
    data[1132] = 0.0 + k[796]*y[IDX_CH4II];
    data[1133] = 0.0 + k[808]*y[IDX_CH4I];
    data[1134] = 0.0 + k[485]*y[IDX_CH3CNHII] + k[487]*y[IDX_CH3OH2II] +
        k[488]*y[IDX_CH3OH2II] + k[492]*y[IDX_CH4II] + k[494]*y[IDX_CH5II] +
        k[495]*y[IDX_CH5II] + k[2133]*y[IDX_CH3II];
    data[1135] = 0.0 + k[1740]*y[IDX_CH2COI] - k[1741]*y[IDX_CH3I] +
        k[1742]*y[IDX_CH4I];
    data[1136] = 0.0 - k[115]*y[IDX_CH3I];
    data[1137] = 0.0 + k[1723]*y[IDX_CH2I] - k[1724]*y[IDX_CH3I] +
        k[2115]*y[IDX_CHI];
    data[1138] = 0.0 + k[742]*y[IDX_CH2II] + k[798]*y[IDX_CH4II] +
        k[1624]*y[IDX_CH2I] - k[1651]*y[IDX_CH3I];
    data[1139] = 0.0 + k[809]*y[IDX_CH4I];
    data[1140] = 0.0 + k[799]*y[IDX_CH4II] - k[1652]*y[IDX_CH3I];
    data[1141] = 0.0 + k[810]*y[IDX_CH4I];
    data[1142] = 0.0 + k[800]*y[IDX_CH4II] - k[1653]*y[IDX_CH3I];
    data[1143] = 0.0 - k[1029]*y[IDX_CH3I];
    data[1144] = 0.0 + k[811]*y[IDX_CH4I];
    data[1145] = 0.0 + k[72]*y[IDX_CH3II] + k[1625]*y[IDX_CH2I] -
        k[1654]*y[IDX_CH3I];
    data[1146] = 0.0 + k[1238]*y[IDX_HeII];
    data[1147] = 0.0 - k[1196]*y[IDX_CH3I] + k[1198]*y[IDX_CH3CNI] +
        k[1200]*y[IDX_CH3OHI] + k[1205]*y[IDX_CH4I] + k[1238]*y[IDX_HCOOCH3I];
    data[1148] = 0.0 + k[1626]*y[IDX_CH2I] - k[1655]*y[IDX_CH3I];
    data[1149] = 0.0 + k[73]*y[IDX_CH3II];
    data[1150] = 0.0 + k[1792]*y[IDX_C2H4I] + k[1794]*y[IDX_C2H5I] -
        k[1804]*y[IDX_CH3I] - k[1805]*y[IDX_CH3I] - k[1806]*y[IDX_CH3I];
    data[1151] = 0.0 + k[1291]*y[IDX_CH3OHI];
    data[1152] = 0.0 + k[1839]*y[IDX_CH4I];
    data[1153] = 0.0 - k[1656]*y[IDX_CH3I] + k[1833]*y[IDX_CH4I];
    data[1154] = 0.0 + k[801]*y[IDX_CH4II] - k[1657]*y[IDX_CH3I];
    data[1155] = 0.0 + k[818]*y[IDX_CH4I];
    data[1156] = 0.0 + k[74]*y[IDX_CH3II] - k[1659]*y[IDX_CH3I];
    data[1157] = 0.0 - k[1658]*y[IDX_CH3I];
    data[1158] = 0.0 - k[1664]*y[IDX_CH3I] - k[1665]*y[IDX_CH3I] +
        k[1873]*y[IDX_C2H4I] + k[1874]*y[IDX_C2H5I] + k[1879]*y[IDX_CH4I];
    data[1159] = 0.0 - k[1660]*y[IDX_CH3I] - k[1661]*y[IDX_CH3I] -
        k[1662]*y[IDX_CH3I] + k[1671]*y[IDX_CH4I];
    data[1160] = 0.0 - k[1663]*y[IDX_CH3I];
    data[1161] = 0.0 + k[802]*y[IDX_CH4II];
    data[1162] = 0.0 + k[1643]*y[IDX_CH2I] - k[1666]*y[IDX_CH3I] -
        k[1667]*y[IDX_CH3I] - k[1668]*y[IDX_CH3I] + k[1672]*y[IDX_CH4I] +
        k[1929]*y[IDX_C2H2I];
    data[1163] = 0.0 - k[1669]*y[IDX_CH3I] + k[1673]*y[IDX_CH4I];
    data[1164] = 0.0 + k[676]*y[IDX_C2H4I] - k[790]*y[IDX_CH3I];
    data[1165] = 0.0 + k[1564]*y[IDX_CH3OHI];
    data[1166] = 0.0 - k[688]*y[IDX_CH3II];
    data[1167] = 0.0 + k[613]*y[IDX_CH3OHI];
    data[1168] = 0.0 + k[754]*y[IDX_CH2I];
    data[1169] = 0.0 - k[774]*y[IDX_CH3II];
    data[1170] = 0.0 + k[1023]*y[IDX_H3II];
    data[1171] = 0.0 - k[839]*y[IDX_CH3II];
    data[1172] = 0.0 + k[709]*y[IDX_CH3OHI] + k[715]*y[IDX_H2COI];
    data[1173] = 0.0 + k[754]*y[IDX_C2HII] + k[755]*y[IDX_CH5II] +
        k[757]*y[IDX_H2COII] + k[758]*y[IDX_H2OII] + k[759]*y[IDX_H3OII] +
        k[760]*y[IDX_HCNII] + k[761]*y[IDX_HCNHII] + k[762]*y[IDX_HCNHII] +
        k[763]*y[IDX_HCOII] + k[764]*y[IDX_HNOII] + k[765]*y[IDX_N2HII] +
        k[766]*y[IDX_NHII] + k[767]*y[IDX_NH2II] + k[768]*y[IDX_NH3II] +
        k[770]*y[IDX_O2HII] + k[771]*y[IDX_OHII] + k[913]*y[IDX_H2II] +
        k[1028]*y[IDX_H3II];
    data[1174] = 0.0 + k[747]*y[IDX_HCOI] + k[938]*y[IDX_H2I];
    data[1175] = 0.0 + k[115]*y[IDX_HII] + k[385] + k[1987];
    data[1176] = 0.0 - k[72]*y[IDX_HCOI] - k[73]*y[IDX_MgI] -
        k[74]*y[IDX_NOI] - k[481]*y[IDX_EM] - k[482]*y[IDX_EM] -
        k[483]*y[IDX_EM] - k[688]*y[IDX_CI] - k[774]*y[IDX_C2H4I] -
        k[775]*y[IDX_CH3CNI] - k[776]*y[IDX_CH3OHI] - k[777]*y[IDX_H2COI] -
        k[778]*y[IDX_H2SI] - k[779]*y[IDX_HCOI] - k[780]*y[IDX_HSI] -
        k[781]*y[IDX_NH3I] - k[782]*y[IDX_O2I] - k[783]*y[IDX_OI] -
        k[784]*y[IDX_OI] - k[785]*y[IDX_OCSI] - k[786]*y[IDX_OHI] -
        k[787]*y[IDX_SI] - k[788]*y[IDX_SOI] - k[789]*y[IDX_SiH4I] -
        k[839]*y[IDX_CHI] - k[1100]*y[IDX_HI] - k[1446]*y[IDX_NHI] - k[1984] -
        k[1985] - k[2107]*y[IDX_H2OI] - k[2108]*y[IDX_HCNI] - k[2114]*y[IDX_H2I]
        - k[2133]*y[IDX_EM] - k[2245];
    data[1177] = 0.0 - k[775]*y[IDX_CH3II] + k[886]*y[IDX_HII] +
        k[1199]*y[IDX_HeII];
    data[1178] = 0.0 + k[613]*y[IDX_CII] + k[709]*y[IDX_CHII] -
        k[776]*y[IDX_CH3II] + k[887]*y[IDX_HII] + k[1031]*y[IDX_H3II] +
        k[1201]*y[IDX_HeII] + k[1292]*y[IDX_NII];
    data[1179] = 0.0 + k[816]*y[IDX_N2II] + k[890]*y[IDX_HII] +
        k[914]*y[IDX_H2II] + k[1204]*y[IDX_HeII] + k[1293]*y[IDX_NII] +
        k[1468]*y[IDX_OII];
    data[1180] = 0.0 + k[1101]*y[IDX_HI] + k[1493]*y[IDX_OI] + k[1994];
    data[1181] = 0.0 + k[755]*y[IDX_CH2I];
    data[1182] = 0.0 - k[481]*y[IDX_CH3II] - k[482]*y[IDX_CH3II] -
        k[483]*y[IDX_CH3II] - k[2133]*y[IDX_CH3II];
    data[1183] = 0.0 - k[1100]*y[IDX_CH3II] + k[1101]*y[IDX_CH4II];
    data[1184] = 0.0 + k[115]*y[IDX_CH3I] + k[886]*y[IDX_CH3CNI] +
        k[887]*y[IDX_CH3OHI] + k[890]*y[IDX_CH4I];
    data[1185] = 0.0 + k[938]*y[IDX_CH2II] - k[2114]*y[IDX_CH3II];
    data[1186] = 0.0 + k[913]*y[IDX_CH2I] + k[914]*y[IDX_CH4I];
    data[1187] = 0.0 + k[715]*y[IDX_CHII] - k[777]*y[IDX_CH3II];
    data[1188] = 0.0 + k[757]*y[IDX_CH2I];
    data[1189] = 0.0 - k[2107]*y[IDX_CH3II];
    data[1190] = 0.0 + k[758]*y[IDX_CH2I];
    data[1191] = 0.0 - k[778]*y[IDX_CH3II];
    data[1192] = 0.0 + k[1023]*y[IDX_C2H5OHI] + k[1028]*y[IDX_CH2I] +
        k[1031]*y[IDX_CH3OHI];
    data[1193] = 0.0 + k[759]*y[IDX_CH2I];
    data[1194] = 0.0 - k[2108]*y[IDX_CH3II];
    data[1195] = 0.0 + k[760]*y[IDX_CH2I];
    data[1196] = 0.0 + k[761]*y[IDX_CH2I] + k[762]*y[IDX_CH2I];
    data[1197] = 0.0 - k[72]*y[IDX_CH3II] + k[747]*y[IDX_CH2II] -
        k[779]*y[IDX_CH3II];
    data[1198] = 0.0 + k[763]*y[IDX_CH2I];
    data[1199] = 0.0 + k[1199]*y[IDX_CH3CNI] + k[1201]*y[IDX_CH3OHI] +
        k[1204]*y[IDX_CH4I];
    data[1200] = 0.0 + k[764]*y[IDX_CH2I];
    data[1201] = 0.0 - k[780]*y[IDX_CH3II];
    data[1202] = 0.0 - k[73]*y[IDX_CH3II];
    data[1203] = 0.0 + k[1292]*y[IDX_CH3OHI] + k[1293]*y[IDX_CH4I];
    data[1204] = 0.0 + k[816]*y[IDX_CH4I];
    data[1205] = 0.0 + k[765]*y[IDX_CH2I];
    data[1206] = 0.0 - k[1446]*y[IDX_CH3II];
    data[1207] = 0.0 + k[766]*y[IDX_CH2I];
    data[1208] = 0.0 + k[767]*y[IDX_CH2I];
    data[1209] = 0.0 - k[781]*y[IDX_CH3II];
    data[1210] = 0.0 + k[768]*y[IDX_CH2I];
    data[1211] = 0.0 - k[74]*y[IDX_CH3II];
    data[1212] = 0.0 - k[783]*y[IDX_CH3II] - k[784]*y[IDX_CH3II] +
        k[1493]*y[IDX_CH4II];
    data[1213] = 0.0 + k[1468]*y[IDX_CH4I];
    data[1214] = 0.0 - k[782]*y[IDX_CH3II];
    data[1215] = 0.0 + k[770]*y[IDX_CH2I];
    data[1216] = 0.0 - k[785]*y[IDX_CH3II];
    data[1217] = 0.0 - k[786]*y[IDX_CH3II];
    data[1218] = 0.0 + k[771]*y[IDX_CH2I];
    data[1219] = 0.0 - k[787]*y[IDX_CH3II];
    data[1220] = 0.0 - k[789]*y[IDX_CH3II];
    data[1221] = 0.0 - k[788]*y[IDX_CH3II];
    data[1222] = 0.0 + k[2403] + k[2404] + k[2405] + k[2406];
    data[1223] = 0.0 + k[1583]*y[IDX_C2H5I];
    data[1224] = 0.0 - k[611]*y[IDX_CH3CCHI];
    data[1225] = 0.0 + k[1675]*y[IDX_CHI];
    data[1226] = 0.0 + k[1583]*y[IDX_CI];
    data[1227] = 0.0 + k[474]*y[IDX_EM];
    data[1228] = 0.0 + k[1675]*y[IDX_C2H4I];
    data[1229] = 0.0 - k[611]*y[IDX_CII] - k[1082]*y[IDX_H3OII] -
        k[1137]*y[IDX_HCOII] - k[1197]*y[IDX_HeII] - k[2149];
    data[1230] = 0.0 + k[474]*y[IDX_C3H5II];
    data[1231] = 0.0 - k[1082]*y[IDX_CH3CCHI];
    data[1232] = 0.0 - k[1137]*y[IDX_CH3CCHI];
    data[1233] = 0.0 - k[1197]*y[IDX_CH3CCHI];
    data[1234] = 0.0 + k[2411] + k[2412] + k[2413] + k[2414];
    data[1235] = 0.0 - k[665]*y[IDX_CH3CNI] - k[666]*y[IDX_CH3CNI];
    data[1236] = 0.0 + k[2109]*y[IDX_CNI];
    data[1237] = 0.0 - k[775]*y[IDX_CH3CNI];
    data[1238] = 0.0 - k[387] - k[665]*y[IDX_C2H2II] - k[666]*y[IDX_C2H2II]
        - k[775]*y[IDX_CH3II] - k[791]*y[IDX_HCO2II] - k[886]*y[IDX_HII] -
        k[1030]*y[IDX_H3II] - k[1083]*y[IDX_H3OII] - k[1130]*y[IDX_HCNHII] -
        k[1131]*y[IDX_HCNHII] - k[1138]*y[IDX_HCOII] - k[1198]*y[IDX_HeII] -
        k[1199]*y[IDX_HeII] - k[1321]*y[IDX_N2HII] - k[1989] - k[2299];
    data[1239] = 0.0 + k[484]*y[IDX_EM];
    data[1240] = 0.0 + k[2109]*y[IDX_CH3I];
    data[1241] = 0.0 + k[484]*y[IDX_CH3CNHII];
    data[1242] = 0.0 - k[886]*y[IDX_CH3CNI];
    data[1243] = 0.0 - k[1030]*y[IDX_CH3CNI];
    data[1244] = 0.0 - k[1083]*y[IDX_CH3CNI];
    data[1245] = 0.0 - k[1130]*y[IDX_CH3CNI] - k[1131]*y[IDX_CH3CNI];
    data[1246] = 0.0 - k[1138]*y[IDX_CH3CNI];
    data[1247] = 0.0 - k[791]*y[IDX_CH3CNI];
    data[1248] = 0.0 - k[1198]*y[IDX_CH3CNI] - k[1199]*y[IDX_CH3CNI];
    data[1249] = 0.0 - k[1321]*y[IDX_CH3CNI];
    data[1250] = 0.0 + k[2419] + k[2420] + k[2421] + k[2422];
    data[1251] = 0.0 - k[2155];
    data[1252] = 0.0 + k[666]*y[IDX_CH3CNI];
    data[1253] = 0.0 + k[2108]*y[IDX_HCNI];
    data[1254] = 0.0 + k[666]*y[IDX_C2H2II] + k[791]*y[IDX_HCO2II] +
        k[1030]*y[IDX_H3II] + k[1083]*y[IDX_H3OII] + k[1130]*y[IDX_HCNHII] +
        k[1131]*y[IDX_HCNHII] + k[1138]*y[IDX_HCOII] + k[1321]*y[IDX_N2HII];
    data[1255] = 0.0 - k[484]*y[IDX_EM] - k[485]*y[IDX_EM] - k[2156];
    data[1256] = 0.0 + k[1120]*y[IDX_HCNI];
    data[1257] = 0.0 - k[484]*y[IDX_CH3CNHII] - k[485]*y[IDX_CH3CNHII];
    data[1258] = 0.0 + k[1030]*y[IDX_CH3CNI];
    data[1259] = 0.0 + k[1083]*y[IDX_CH3CNI];
    data[1260] = 0.0 + k[1120]*y[IDX_CH3OH2II] + k[2108]*y[IDX_CH3II];
    data[1261] = 0.0 + k[1130]*y[IDX_CH3CNI] + k[1131]*y[IDX_CH3CNI];
    data[1262] = 0.0 + k[1138]*y[IDX_CH3CNI];
    data[1263] = 0.0 + k[791]*y[IDX_CH3CNI];
    data[1264] = 0.0 + k[1321]*y[IDX_CH3CNI];
    data[1265] = 0.0 + k[2379] + k[2380] + k[2381] + k[2382];
    data[1266] = 0.0 - k[612]*y[IDX_CH3OHI] - k[613]*y[IDX_CH3OHI];
    data[1267] = 0.0 - k[708]*y[IDX_CH3OHI] - k[709]*y[IDX_CH3OHI] -
        k[710]*y[IDX_CH3OHI];
    data[1268] = 0.0 - k[776]*y[IDX_CH3OHI];
    data[1269] = 0.0 - k[388] - k[389] - k[612]*y[IDX_CII] -
        k[613]*y[IDX_CII] - k[708]*y[IDX_CHII] - k[709]*y[IDX_CHII] -
        k[710]*y[IDX_CHII] - k[776]*y[IDX_CH3II] - k[793]*y[IDX_S2II] -
        k[794]*y[IDX_CH4II] - k[887]*y[IDX_HII] - k[888]*y[IDX_HII] -
        k[889]*y[IDX_HII] - k[965]*y[IDX_H2COII] - k[1031]*y[IDX_H3II] -
        k[1032]*y[IDX_H3II] - k[1078]*y[IDX_H3COII] - k[1084]*y[IDX_H3OII] -
        k[1139]*y[IDX_HCOII] - k[1200]*y[IDX_HeII] - k[1201]*y[IDX_HeII] -
        k[1289]*y[IDX_NII] - k[1290]*y[IDX_NII] - k[1291]*y[IDX_NII] -
        k[1292]*y[IDX_NII] - k[1466]*y[IDX_OII] - k[1467]*y[IDX_OII] -
        k[1483]*y[IDX_O2II] - k[1564]*y[IDX_SiII] - k[1990] - k[1991] - k[1992]
        - k[2154];
    data[1270] = 0.0 + k[489]*y[IDX_EM] + k[792]*y[IDX_NH3I];
    data[1271] = 0.0 - k[794]*y[IDX_CH3OHI];
    data[1272] = 0.0 + k[489]*y[IDX_CH3OH2II] + k[536]*y[IDX_H5C2O2II];
    data[1273] = 0.0 - k[887]*y[IDX_CH3OHI] - k[888]*y[IDX_CH3OHI] -
        k[889]*y[IDX_CH3OHI];
    data[1274] = 0.0 - k[965]*y[IDX_CH3OHI];
    data[1275] = 0.0 - k[1031]*y[IDX_CH3OHI] - k[1032]*y[IDX_CH3OHI];
    data[1276] = 0.0 - k[1078]*y[IDX_CH3OHI];
    data[1277] = 0.0 - k[1084]*y[IDX_CH3OHI];
    data[1278] = 0.0 + k[536]*y[IDX_EM];
    data[1279] = 0.0 - k[1139]*y[IDX_CH3OHI];
    data[1280] = 0.0 - k[1200]*y[IDX_CH3OHI] - k[1201]*y[IDX_CH3OHI];
    data[1281] = 0.0 - k[1289]*y[IDX_CH3OHI] - k[1290]*y[IDX_CH3OHI] -
        k[1291]*y[IDX_CH3OHI] - k[1292]*y[IDX_CH3OHI];
    data[1282] = 0.0 + k[792]*y[IDX_CH3OH2II];
    data[1283] = 0.0 - k[1466]*y[IDX_CH3OHI] - k[1467]*y[IDX_CH3OHI];
    data[1284] = 0.0 - k[1483]*y[IDX_CH3OHI];
    data[1285] = 0.0 - k[793]*y[IDX_CH3OHI];
    data[1286] = 0.0 - k[1564]*y[IDX_CH3OHI];
    data[1287] = 0.0 + k[708]*y[IDX_CH3OHI];
    data[1288] = 0.0 + k[2107]*y[IDX_H2OI];
    data[1289] = 0.0 + k[708]*y[IDX_CHII] + k[794]*y[IDX_CH4II] +
        k[965]*y[IDX_H2COII] + k[1032]*y[IDX_H3II] + k[1078]*y[IDX_H3COII] +
        k[1084]*y[IDX_H3OII] + k[1139]*y[IDX_HCOII];
    data[1290] = 0.0 - k[486]*y[IDX_EM] - k[487]*y[IDX_EM] -
        k[488]*y[IDX_EM] - k[489]*y[IDX_EM] - k[490]*y[IDX_EM] -
        k[792]*y[IDX_NH3I] - k[969]*y[IDX_H2COI] - k[1120]*y[IDX_HCNI] -
        k[2152];
    data[1291] = 0.0 + k[794]*y[IDX_CH3OHI];
    data[1292] = 0.0 - k[486]*y[IDX_CH3OH2II] - k[487]*y[IDX_CH3OH2II] -
        k[488]*y[IDX_CH3OH2II] - k[489]*y[IDX_CH3OH2II] -
        k[490]*y[IDX_CH3OH2II];
    data[1293] = 0.0 - k[969]*y[IDX_CH3OH2II];
    data[1294] = 0.0 + k[965]*y[IDX_CH3OHI];
    data[1295] = 0.0 + k[2107]*y[IDX_CH3II];
    data[1296] = 0.0 + k[1032]*y[IDX_CH3OHI];
    data[1297] = 0.0 + k[1078]*y[IDX_CH3OHI];
    data[1298] = 0.0 + k[1084]*y[IDX_CH3OHI];
    data[1299] = 0.0 - k[1120]*y[IDX_CH3OH2II];
    data[1300] = 0.0 + k[1139]*y[IDX_CH3OHI];
    data[1301] = 0.0 - k[792]*y[IDX_CH3OH2II];
    data[1302] = 0.0 + k[2303] + k[2304] + k[2305] + k[2306];
    data[1303] = 0.0 + k[689]*y[IDX_CH5II];
    data[1304] = 0.0 - k[614]*y[IDX_CH4I];
    data[1305] = 0.0 + k[823]*y[IDX_CH5II];
    data[1306] = 0.0 - k[803]*y[IDX_CH4I] - k[804]*y[IDX_CH4I];
    data[1307] = 0.0 + k[824]*y[IDX_CH5II];
    data[1308] = 0.0 - k[805]*y[IDX_CH4I];
    data[1309] = 0.0 + k[75]*y[IDX_CH4II];
    data[1310] = 0.0 - k[806]*y[IDX_CH4I];
    data[1311] = 0.0 + k[1646]*y[IDX_CH3I];
    data[1312] = 0.0 + k[1023]*y[IDX_H3II] + k[1024]*y[IDX_H3II];
    data[1313] = 0.0 + k[840]*y[IDX_CH5II] - k[1676]*y[IDX_CH4I];
    data[1314] = 0.0 - k[711]*y[IDX_CH4I];
    data[1315] = 0.0 + k[755]*y[IDX_CH5II] - k[1622]*y[IDX_CH4I];
    data[1316] = 0.0 + k[1646]*y[IDX_C2H3I] + k[1649]*y[IDX_CH3I] +
        k[1649]*y[IDX_CH3I] + k[1651]*y[IDX_H2COI] + k[1652]*y[IDX_H2OI] +
        k[1653]*y[IDX_H2SI] + k[1654]*y[IDX_HCOI] + k[1655]*y[IDX_HNOI] +
        k[1656]*y[IDX_NH2I] + k[1657]*y[IDX_NH3I] + k[1663]*y[IDX_O2HI] +
        k[1666]*y[IDX_OHI] + k[1724]*y[IDX_H2I];
    data[1317] = 0.0 + k[776]*y[IDX_CH3OHI] + k[777]*y[IDX_H2COI] +
        k[789]*y[IDX_SiH4I];
    data[1318] = 0.0 + k[776]*y[IDX_CH3II];
    data[1319] = 0.0 - k[81]*y[IDX_COII] - k[116]*y[IDX_HII] -
        k[157]*y[IDX_H2II] - k[210]*y[IDX_HeII] - k[236]*y[IDX_NII] -
        k[309]*y[IDX_OII] - k[390] - k[614]*y[IDX_CII] - k[711]*y[IDX_CHII] -
        k[795]*y[IDX_CH4II] - k[803]*y[IDX_C2II] - k[804]*y[IDX_C2II] -
        k[805]*y[IDX_C2HII] - k[806]*y[IDX_C2H2II] - k[807]*y[IDX_COII] -
        k[808]*y[IDX_CSII] - k[809]*y[IDX_H2COII] - k[810]*y[IDX_H2OII] -
        k[811]*y[IDX_HCNII] - k[812]*y[IDX_HCO2II] - k[813]*y[IDX_HNOII] -
        k[814]*y[IDX_HSII] - k[815]*y[IDX_N2II] - k[816]*y[IDX_N2II] -
        k[817]*y[IDX_N2HII] - k[818]*y[IDX_NH3II] - k[819]*y[IDX_OHII] -
        k[820]*y[IDX_OHII] - k[821]*y[IDX_SII] - k[822]*y[IDX_SII] -
        k[890]*y[IDX_HII] - k[914]*y[IDX_H2II] - k[915]*y[IDX_H2II] -
        k[1033]*y[IDX_H3II] - k[1202]*y[IDX_HeII] - k[1203]*y[IDX_HeII] -
        k[1204]*y[IDX_HeII] - k[1205]*y[IDX_HeII] - k[1293]*y[IDX_NII] -
        k[1294]*y[IDX_NII] - k[1295]*y[IDX_NII] - k[1468]*y[IDX_OII] -
        k[1622]*y[IDX_CH2I] - k[1670]*y[IDX_CNI] - k[1671]*y[IDX_O2I] -
        k[1672]*y[IDX_OHI] - k[1673]*y[IDX_SI] - k[1676]*y[IDX_CHI] -
        k[1742]*y[IDX_HI] - k[1833]*y[IDX_NH2I] - k[1839]*y[IDX_NHI] -
        k[1879]*y[IDX_OI] - k[1995] - k[1996] - k[1997] - k[1998] - k[2217];
    data[1320] = 0.0 + k[75]*y[IDX_C2H2I] + k[76]*y[IDX_H2COI] +
        k[77]*y[IDX_H2SI] + k[78]*y[IDX_NH3I] + k[79]*y[IDX_O2I] +
        k[80]*y[IDX_OCSI] - k[795]*y[IDX_CH4I];
    data[1321] = 0.0 + k[496]*y[IDX_EM] + k[689]*y[IDX_CI] +
        k[755]*y[IDX_CH2I] + k[823]*y[IDX_C2I] + k[824]*y[IDX_C2HI] +
        k[825]*y[IDX_CO2I] + k[826]*y[IDX_COI] + k[827]*y[IDX_H2COI] +
        k[828]*y[IDX_H2OI] + k[829]*y[IDX_H2SI] + k[830]*y[IDX_HCNI] +
        k[831]*y[IDX_HCOI] + k[832]*y[IDX_HClI] + k[833]*y[IDX_HNCI] +
        k[834]*y[IDX_MgI] + k[835]*y[IDX_SI] + k[836]*y[IDX_SiH4I] +
        k[840]*y[IDX_CHI] + k[1397]*y[IDX_NH2I] + k[1422]*y[IDX_NH3I] +
        k[1447]*y[IDX_NHI] + k[1538]*y[IDX_OHI];
    data[1322] = 0.0 - k[1670]*y[IDX_CH4I];
    data[1323] = 0.0 + k[826]*y[IDX_CH5II];
    data[1324] = 0.0 - k[81]*y[IDX_CH4I] - k[807]*y[IDX_CH4I];
    data[1325] = 0.0 + k[825]*y[IDX_CH5II];
    data[1326] = 0.0 - k[808]*y[IDX_CH4I];
    data[1327] = 0.0 + k[496]*y[IDX_CH5II];
    data[1328] = 0.0 - k[1742]*y[IDX_CH4I];
    data[1329] = 0.0 - k[116]*y[IDX_CH4I] - k[890]*y[IDX_CH4I];
    data[1330] = 0.0 + k[1724]*y[IDX_CH3I];
    data[1331] = 0.0 - k[157]*y[IDX_CH4I] - k[914]*y[IDX_CH4I] -
        k[915]*y[IDX_CH4I];
    data[1332] = 0.0 + k[76]*y[IDX_CH4II] + k[777]*y[IDX_CH3II] +
        k[827]*y[IDX_CH5II] + k[1651]*y[IDX_CH3I];
    data[1333] = 0.0 - k[809]*y[IDX_CH4I];
    data[1334] = 0.0 + k[828]*y[IDX_CH5II] + k[1652]*y[IDX_CH3I];
    data[1335] = 0.0 - k[810]*y[IDX_CH4I];
    data[1336] = 0.0 + k[77]*y[IDX_CH4II] + k[829]*y[IDX_CH5II] +
        k[1653]*y[IDX_CH3I];
    data[1337] = 0.0 + k[1023]*y[IDX_C2H5OHI] + k[1024]*y[IDX_C2H5OHI] -
        k[1033]*y[IDX_CH4I];
    data[1338] = 0.0 + k[832]*y[IDX_CH5II];
    data[1339] = 0.0 + k[830]*y[IDX_CH5II];
    data[1340] = 0.0 - k[811]*y[IDX_CH4I];
    data[1341] = 0.0 + k[831]*y[IDX_CH5II] + k[1654]*y[IDX_CH3I];
    data[1342] = 0.0 - k[812]*y[IDX_CH4I];
    data[1343] = 0.0 + k[411];
    data[1344] = 0.0 - k[210]*y[IDX_CH4I] - k[1202]*y[IDX_CH4I] -
        k[1203]*y[IDX_CH4I] - k[1204]*y[IDX_CH4I] - k[1205]*y[IDX_CH4I];
    data[1345] = 0.0 + k[833]*y[IDX_CH5II];
    data[1346] = 0.0 + k[1655]*y[IDX_CH3I];
    data[1347] = 0.0 - k[813]*y[IDX_CH4I];
    data[1348] = 0.0 - k[814]*y[IDX_CH4I];
    data[1349] = 0.0 + k[834]*y[IDX_CH5II];
    data[1350] = 0.0 - k[236]*y[IDX_CH4I] - k[1293]*y[IDX_CH4I] -
        k[1294]*y[IDX_CH4I] - k[1295]*y[IDX_CH4I];
    data[1351] = 0.0 - k[815]*y[IDX_CH4I] - k[816]*y[IDX_CH4I];
    data[1352] = 0.0 - k[817]*y[IDX_CH4I];
    data[1353] = 0.0 + k[1447]*y[IDX_CH5II] - k[1839]*y[IDX_CH4I];
    data[1354] = 0.0 + k[1397]*y[IDX_CH5II] + k[1656]*y[IDX_CH3I] -
        k[1833]*y[IDX_CH4I];
    data[1355] = 0.0 + k[78]*y[IDX_CH4II] + k[1422]*y[IDX_CH5II] +
        k[1657]*y[IDX_CH3I];
    data[1356] = 0.0 - k[818]*y[IDX_CH4I];
    data[1357] = 0.0 - k[1879]*y[IDX_CH4I];
    data[1358] = 0.0 - k[309]*y[IDX_CH4I] - k[1468]*y[IDX_CH4I];
    data[1359] = 0.0 + k[79]*y[IDX_CH4II] - k[1671]*y[IDX_CH4I];
    data[1360] = 0.0 + k[1663]*y[IDX_CH3I];
    data[1361] = 0.0 + k[80]*y[IDX_CH4II];
    data[1362] = 0.0 + k[1538]*y[IDX_CH5II] + k[1666]*y[IDX_CH3I] -
        k[1672]*y[IDX_CH4I];
    data[1363] = 0.0 - k[819]*y[IDX_CH4I] - k[820]*y[IDX_CH4I];
    data[1364] = 0.0 + k[835]*y[IDX_CH5II] - k[1673]*y[IDX_CH4I];
    data[1365] = 0.0 - k[821]*y[IDX_CH4I] - k[822]*y[IDX_CH4I];
    data[1366] = 0.0 + k[789]*y[IDX_CH3II] + k[836]*y[IDX_CH5II];
    data[1367] = 0.0 - k[75]*y[IDX_CH4II];
    data[1368] = 0.0 + k[1029]*y[IDX_H3II];
    data[1369] = 0.0 + k[779]*y[IDX_HCOI];
    data[1370] = 0.0 - k[794]*y[IDX_CH4II];
    data[1371] = 0.0 + k[81]*y[IDX_COII] + k[116]*y[IDX_HII] +
        k[157]*y[IDX_H2II] + k[210]*y[IDX_HeII] + k[236]*y[IDX_NII] +
        k[309]*y[IDX_OII] - k[795]*y[IDX_CH4II] + k[1997];
    data[1372] = 0.0 - k[75]*y[IDX_C2H2I] - k[76]*y[IDX_H2COI] -
        k[77]*y[IDX_H2SI] - k[78]*y[IDX_NH3I] - k[79]*y[IDX_O2I] -
        k[80]*y[IDX_OCSI] - k[491]*y[IDX_EM] - k[492]*y[IDX_EM] -
        k[794]*y[IDX_CH3OHI] - k[795]*y[IDX_CH4I] - k[796]*y[IDX_CO2I] -
        k[797]*y[IDX_COI] - k[798]*y[IDX_H2COI] - k[799]*y[IDX_H2OI] -
        k[800]*y[IDX_H2SI] - k[801]*y[IDX_NH3I] - k[802]*y[IDX_OCSI] -
        k[939]*y[IDX_H2I] - k[1101]*y[IDX_HI] - k[1493]*y[IDX_OI] - k[1993] -
        k[1994] - k[2249];
    data[1373] = 0.0 + k[1102]*y[IDX_HI];
    data[1374] = 0.0 - k[797]*y[IDX_CH4II];
    data[1375] = 0.0 + k[81]*y[IDX_CH4I];
    data[1376] = 0.0 - k[796]*y[IDX_CH4II];
    data[1377] = 0.0 - k[491]*y[IDX_CH4II] - k[492]*y[IDX_CH4II];
    data[1378] = 0.0 - k[1101]*y[IDX_CH4II] + k[1102]*y[IDX_CH5II];
    data[1379] = 0.0 + k[116]*y[IDX_CH4I];
    data[1380] = 0.0 - k[939]*y[IDX_CH4II];
    data[1381] = 0.0 + k[157]*y[IDX_CH4I];
    data[1382] = 0.0 - k[76]*y[IDX_CH4II] - k[798]*y[IDX_CH4II];
    data[1383] = 0.0 - k[799]*y[IDX_CH4II];
    data[1384] = 0.0 - k[77]*y[IDX_CH4II] - k[800]*y[IDX_CH4II];
    data[1385] = 0.0 + k[1029]*y[IDX_CH3I];
    data[1386] = 0.0 + k[779]*y[IDX_CH3II];
    data[1387] = 0.0 + k[210]*y[IDX_CH4I];
    data[1388] = 0.0 + k[236]*y[IDX_CH4I];
    data[1389] = 0.0 - k[78]*y[IDX_CH4II] - k[801]*y[IDX_CH4II];
    data[1390] = 0.0 - k[1493]*y[IDX_CH4II];
    data[1391] = 0.0 + k[309]*y[IDX_CH4I];
    data[1392] = 0.0 - k[79]*y[IDX_CH4II];
    data[1393] = 0.0 - k[80]*y[IDX_CH4II] - k[802]*y[IDX_CH4II];
    data[1394] = 0.0 - k[689]*y[IDX_CH5II];
    data[1395] = 0.0 - k[823]*y[IDX_CH5II];
    data[1396] = 0.0 - k[824]*y[IDX_CH5II];
    data[1397] = 0.0 - k[840]*y[IDX_CH5II];
    data[1398] = 0.0 - k[755]*y[IDX_CH5II];
    data[1399] = 0.0 + k[2114]*y[IDX_H2I];
    data[1400] = 0.0 + k[795]*y[IDX_CH4II] + k[812]*y[IDX_HCO2II] +
        k[813]*y[IDX_HNOII] + k[817]*y[IDX_N2HII] + k[819]*y[IDX_OHII] +
        k[915]*y[IDX_H2II] + k[1033]*y[IDX_H3II];
    data[1401] = 0.0 + k[795]*y[IDX_CH4I] + k[939]*y[IDX_H2I];
    data[1402] = 0.0 - k[493]*y[IDX_EM] - k[494]*y[IDX_EM] -
        k[495]*y[IDX_EM] - k[496]*y[IDX_EM] - k[497]*y[IDX_EM] -
        k[689]*y[IDX_CI] - k[755]*y[IDX_CH2I] - k[823]*y[IDX_C2I] -
        k[824]*y[IDX_C2HI] - k[825]*y[IDX_CO2I] - k[826]*y[IDX_COI] -
        k[827]*y[IDX_H2COI] - k[828]*y[IDX_H2OI] - k[829]*y[IDX_H2SI] -
        k[830]*y[IDX_HCNI] - k[831]*y[IDX_HCOI] - k[832]*y[IDX_HClI] -
        k[833]*y[IDX_HNCI] - k[834]*y[IDX_MgI] - k[835]*y[IDX_SI] -
        k[836]*y[IDX_SiH4I] - k[840]*y[IDX_CHI] - k[1102]*y[IDX_HI] -
        k[1397]*y[IDX_NH2I] - k[1422]*y[IDX_NH3I] - k[1447]*y[IDX_NHI] -
        k[1494]*y[IDX_OI] - k[1495]*y[IDX_OI] - k[1538]*y[IDX_OHI] - k[2248];
    data[1403] = 0.0 - k[826]*y[IDX_CH5II];
    data[1404] = 0.0 - k[825]*y[IDX_CH5II];
    data[1405] = 0.0 - k[493]*y[IDX_CH5II] - k[494]*y[IDX_CH5II] -
        k[495]*y[IDX_CH5II] - k[496]*y[IDX_CH5II] - k[497]*y[IDX_CH5II];
    data[1406] = 0.0 - k[1102]*y[IDX_CH5II];
    data[1407] = 0.0 + k[939]*y[IDX_CH4II] + k[2114]*y[IDX_CH3II];
    data[1408] = 0.0 + k[915]*y[IDX_CH4I];
    data[1409] = 0.0 - k[827]*y[IDX_CH5II];
    data[1410] = 0.0 - k[828]*y[IDX_CH5II];
    data[1411] = 0.0 - k[829]*y[IDX_CH5II];
    data[1412] = 0.0 + k[1033]*y[IDX_CH4I];
    data[1413] = 0.0 - k[832]*y[IDX_CH5II];
    data[1414] = 0.0 - k[830]*y[IDX_CH5II];
    data[1415] = 0.0 - k[831]*y[IDX_CH5II];
    data[1416] = 0.0 + k[812]*y[IDX_CH4I];
    data[1417] = 0.0 - k[833]*y[IDX_CH5II];
    data[1418] = 0.0 + k[813]*y[IDX_CH4I];
    data[1419] = 0.0 - k[834]*y[IDX_CH5II];
    data[1420] = 0.0 + k[817]*y[IDX_CH4I];
    data[1421] = 0.0 - k[1447]*y[IDX_CH5II];
    data[1422] = 0.0 - k[1397]*y[IDX_CH5II];
    data[1423] = 0.0 - k[1422]*y[IDX_CH5II];
    data[1424] = 0.0 - k[1494]*y[IDX_CH5II] - k[1495]*y[IDX_CH5II];
    data[1425] = 0.0 - k[1538]*y[IDX_CH5II];
    data[1426] = 0.0 + k[819]*y[IDX_CH4I];
    data[1427] = 0.0 - k[835]*y[IDX_CH5II];
    data[1428] = 0.0 - k[836]*y[IDX_CH5II];
    data[1429] = 0.0 - k[109]*y[IDX_HII] - k[358] - k[397] -
        k[1040]*y[IDX_H3II] - k[1720]*y[IDX_H2I] - k[2008] - k[2194];
    data[1430] = 0.0 + k[108]*y[IDX_HI] + k[324]*y[IDX_O2I] +
        k[2134]*y[IDX_EM];
    data[1431] = 0.0 + k[508]*y[IDX_H2ClII] + k[548]*y[IDX_HClII] +
        k[2134]*y[IDX_ClII];
    data[1432] = 0.0 + k[108]*y[IDX_ClII] + k[1787]*y[IDX_HClI];
    data[1433] = 0.0 - k[109]*y[IDX_ClI];
    data[1434] = 0.0 - k[1720]*y[IDX_ClI];
    data[1435] = 0.0 + k[508]*y[IDX_EM];
    data[1436] = 0.0 - k[1040]*y[IDX_ClI];
    data[1437] = 0.0 + k[413] + k[1787]*y[IDX_HI] + k[2034];
    data[1438] = 0.0 + k[548]*y[IDX_EM];
    data[1439] = 0.0 + k[324]*y[IDX_ClII];
    data[1440] = 0.0 + k[109]*y[IDX_HII] + k[358] + k[397] + k[2008];
    data[1441] = 0.0 - k[108]*y[IDX_HI] - k[324]*y[IDX_O2I] -
        k[944]*y[IDX_H2I] - k[2134]*y[IDX_EM] - k[2196];
    data[1442] = 0.0 - k[2134]*y[IDX_ClII];
    data[1443] = 0.0 - k[108]*y[IDX_ClII];
    data[1444] = 0.0 + k[109]*y[IDX_ClI];
    data[1445] = 0.0 - k[944]*y[IDX_ClII];
    data[1446] = 0.0 + k[1241]*y[IDX_HeII];
    data[1447] = 0.0 + k[1241]*y[IDX_HClI];
    data[1448] = 0.0 - k[324]*y[IDX_ClII];
    data[1449] = 0.0 + k[51]*y[IDX_CNII] + k[693]*y[IDX_HCNII] +
        k[1584]*y[IDX_C2NI] - k[1590]*y[IDX_CNI] + k[1597]*y[IDX_N2I] +
        k[1598]*y[IDX_NCCNI] + k[1602]*y[IDX_NHI] + k[1604]*y[IDX_NOI] +
        k[1607]*y[IDX_NSI] + k[1609]*y[IDX_OCNI] + k[2102]*y[IDX_NI];
    data[1450] = 0.0 + k[20]*y[IDX_NCCNI] + k[635]*y[IDX_OCNI];
    data[1451] = 0.0 + k[36]*y[IDX_CNII] + k[652]*y[IDX_HCNII] +
        k[1790]*y[IDX_NI];
    data[1452] = 0.0 + k[1325]*y[IDX_NI];
    data[1453] = 0.0 + k[47]*y[IDX_CNII] + k[679]*y[IDX_HCNII] +
        k[1580]*y[IDX_NCCNI] - k[2100]*y[IDX_CNI];
    data[1454] = 0.0 + k[661]*y[IDX_HCNI] + k[1327]*y[IDX_NI];
    data[1455] = 0.0 - k[1701]*y[IDX_CNI];
    data[1456] = 0.0 + k[665]*y[IDX_CH3CNI];
    data[1457] = 0.0 - k[1702]*y[IDX_CNI];
    data[1458] = 0.0 + k[376] + k[1189]*y[IDX_HeII] + k[1584]*y[IDX_CI] +
        k[1796]*y[IDX_NI] + k[1796]*y[IDX_NI] + k[1876]*y[IDX_OI] + k[1973];
    data[1459] = 0.0 + k[469]*y[IDX_EM];
    data[1460] = 0.0 + k[471]*y[IDX_EM] + k[471]*y[IDX_EM] +
        k[1097]*y[IDX_HI];
    data[1461] = 0.0 + k[377] + k[1190]*y[IDX_HeII] + k[1798]*y[IDX_NI] +
        k[1974];
    data[1462] = 0.0 + k[1800]*y[IDX_NI];
    data[1463] = 0.0 + k[83]*y[IDX_CNII] + k[846]*y[IDX_HCNII] +
        k[1682]*y[IDX_NI];
    data[1464] = 0.0 - k[713]*y[IDX_CNI];
    data[1465] = 0.0 + k[63]*y[IDX_CNII] + k[760]*y[IDX_HCNII] -
        k[1623]*y[IDX_CNI];
    data[1466] = 0.0 - k[1650]*y[IDX_CNI] - k[2109]*y[IDX_CNI];
    data[1467] = 0.0 + k[387] + k[665]*y[IDX_C2H2II] + k[1199]*y[IDX_HeII] +
        k[1989];
    data[1468] = 0.0 - k[1670]*y[IDX_CNI];
    data[1469] = 0.0 - k[100]*y[IDX_N2II] - k[159]*y[IDX_H2II] -
        k[237]*y[IDX_NII] - k[392] - k[713]*y[IDX_CHII] - k[869]*y[IDX_HNOII] -
        k[870]*y[IDX_O2HII] - k[917]*y[IDX_H2II] - k[1035]*y[IDX_H3II] -
        k[1207]*y[IDX_HeII] - k[1208]*y[IDX_HeII] - k[1349]*y[IDX_NHII] -
        k[1469]*y[IDX_OII] - k[1519]*y[IDX_OHII] - k[1590]*y[IDX_CI] -
        k[1623]*y[IDX_CH2I] - k[1650]*y[IDX_CH3I] - k[1670]*y[IDX_CH4I] -
        k[1701]*y[IDX_C2H2I] - k[1702]*y[IDX_C2H4I] - k[1703]*y[IDX_CNI] -
        k[1703]*y[IDX_CNI] - k[1703]*y[IDX_CNI] - k[1703]*y[IDX_CNI] -
        k[1704]*y[IDX_H2COI] - k[1705]*y[IDX_HCNI] - k[1706]*y[IDX_HCOI] -
        k[1707]*y[IDX_HNCI] - k[1708]*y[IDX_HNOI] - k[1709]*y[IDX_NO2I] -
        k[1710]*y[IDX_NOI] - k[1711]*y[IDX_NOI] - k[1712]*y[IDX_O2I] -
        k[1713]*y[IDX_O2I] - k[1714]*y[IDX_SI] - k[1715]*y[IDX_SiH4I] -
        k[1726]*y[IDX_H2I] - k[1807]*y[IDX_NI] - k[1838]*y[IDX_NH3I] -
        k[1840]*y[IDX_NHI] - k[1880]*y[IDX_OI] - k[1881]*y[IDX_OI] -
        k[1932]*y[IDX_OHI] - k[1933]*y[IDX_OHI] - k[2001] - k[2100]*y[IDX_C2HI]
        - k[2109]*y[IDX_CH3I] - k[2220];
    data[1470] = 0.0 + k[36]*y[IDX_C2I] + k[47]*y[IDX_C2HI] +
        k[51]*y[IDX_CI] + k[63]*y[IDX_CH2I] + k[83]*y[IDX_CHI] +
        k[93]*y[IDX_COI] + k[94]*y[IDX_H2COI] + k[95]*y[IDX_HCNI] +
        k[96]*y[IDX_HCOI] + k[97]*y[IDX_NOI] + k[98]*y[IDX_O2I] +
        k[99]*y[IDX_SI] + k[191]*y[IDX_HI] + k[272]*y[IDX_NH2I] +
        k[294]*y[IDX_NHI] + k[326]*y[IDX_OI] + k[340]*y[IDX_OHI];
    data[1471] = 0.0 + k[93]*y[IDX_CNII] + k[1111]*y[IDX_HCNII];
    data[1472] = 0.0 + k[1110]*y[IDX_HCNII];
    data[1473] = 0.0 + k[1809]*y[IDX_NI];
    data[1474] = 0.0 + k[469]*y[IDX_C2NII] + k[471]*y[IDX_C2N2II] +
        k[471]*y[IDX_C2N2II] + k[538]*y[IDX_HCNII] + k[539]*y[IDX_HCNHII];
    data[1475] = 0.0 + k[191]*y[IDX_CNII] + k[1097]*y[IDX_C2N2II] +
        k[1750]*y[IDX_HCNI] + k[1759]*y[IDX_NCCNI] + k[1774]*y[IDX_OCNI];
    data[1476] = 0.0 - k[1726]*y[IDX_CNI];
    data[1477] = 0.0 - k[159]*y[IDX_CNI] - k[917]*y[IDX_CNI];
    data[1478] = 0.0 + k[94]*y[IDX_CNII] + k[1112]*y[IDX_HCNII] -
        k[1704]*y[IDX_CNI];
    data[1479] = 0.0 + k[1002]*y[IDX_HCNII];
    data[1480] = 0.0 - k[1035]*y[IDX_CNI];
    data[1481] = 0.0 + k[407] + k[1230]*y[IDX_HeII] + k[2027];
    data[1482] = 0.0 + k[95]*y[IDX_CNII] + k[408] + k[661]*y[IDX_C2HII] +
        k[1113]*y[IDX_HCNII] - k[1705]*y[IDX_CNI] + k[1750]*y[IDX_HI] +
        k[1889]*y[IDX_OI] + k[1939]*y[IDX_OHI] + k[2028];
    data[1483] = 0.0 + k[538]*y[IDX_EM] + k[652]*y[IDX_C2I] +
        k[679]*y[IDX_C2HI] + k[693]*y[IDX_CI] + k[760]*y[IDX_CH2I] +
        k[846]*y[IDX_CHI] + k[1002]*y[IDX_H2OI] + k[1110]*y[IDX_CO2I] +
        k[1111]*y[IDX_COI] + k[1112]*y[IDX_H2COI] + k[1113]*y[IDX_HCNI] +
        k[1114]*y[IDX_HCOI] + k[1116]*y[IDX_HNCI] + k[1117]*y[IDX_SI] +
        k[1403]*y[IDX_NH2I] + k[1451]*y[IDX_NHI] + k[1541]*y[IDX_OHI];
    data[1484] = 0.0 + k[539]*y[IDX_EM];
    data[1485] = 0.0 + k[96]*y[IDX_CNII] + k[1114]*y[IDX_HCNII] -
        k[1706]*y[IDX_CNI];
    data[1486] = 0.0 + k[1189]*y[IDX_C2NI] + k[1190]*y[IDX_C3NI] +
        k[1199]*y[IDX_CH3CNI] - k[1207]*y[IDX_CNI] - k[1208]*y[IDX_CNI] +
        k[1230]*y[IDX_HC3NI] + k[1251]*y[IDX_NCCNI] + k[1263]*y[IDX_OCNI];
    data[1487] = 0.0 + k[414] + k[1116]*y[IDX_HCNII] - k[1707]*y[IDX_CNI] +
        k[2036];
    data[1488] = 0.0 - k[1708]*y[IDX_CNI];
    data[1489] = 0.0 - k[869]*y[IDX_CNI];
    data[1490] = 0.0 + k[1325]*y[IDX_C2II] + k[1327]*y[IDX_C2HII] +
        k[1342]*y[IDX_SiCII] + k[1682]*y[IDX_CHI] + k[1790]*y[IDX_C2I] +
        k[1796]*y[IDX_C2NI] + k[1796]*y[IDX_C2NI] + k[1798]*y[IDX_C3NI] +
        k[1800]*y[IDX_C4NI] - k[1807]*y[IDX_CNI] + k[1809]*y[IDX_CSI] +
        k[1832]*y[IDX_SiCI] + k[2102]*y[IDX_CI];
    data[1491] = 0.0 - k[237]*y[IDX_CNI];
    data[1492] = 0.0 + k[1597]*y[IDX_CI];
    data[1493] = 0.0 - k[100]*y[IDX_CNI];
    data[1494] = 0.0 + k[20]*y[IDX_CII] + k[423] + k[423] +
        k[1251]*y[IDX_HeII] + k[1580]*y[IDX_C2HI] + k[1598]*y[IDX_CI] +
        k[1759]*y[IDX_HI] + k[2047] + k[2047];
    data[1495] = 0.0 + k[294]*y[IDX_CNII] + k[1451]*y[IDX_HCNII] +
        k[1602]*y[IDX_CI] - k[1840]*y[IDX_CNI];
    data[1496] = 0.0 - k[1349]*y[IDX_CNI];
    data[1497] = 0.0 + k[272]*y[IDX_CNII] + k[1403]*y[IDX_HCNII];
    data[1498] = 0.0 - k[1838]*y[IDX_CNI];
    data[1499] = 0.0 + k[97]*y[IDX_CNII] + k[1604]*y[IDX_CI] -
        k[1710]*y[IDX_CNI] - k[1711]*y[IDX_CNI];
    data[1500] = 0.0 - k[1709]*y[IDX_CNI];
    data[1501] = 0.0 + k[1607]*y[IDX_CI];
    data[1502] = 0.0 + k[326]*y[IDX_CNII] + k[1876]*y[IDX_C2NI] -
        k[1880]*y[IDX_CNI] - k[1881]*y[IDX_CNI] + k[1889]*y[IDX_HCNI] +
        k[1911]*y[IDX_OCNI];
    data[1503] = 0.0 - k[1469]*y[IDX_CNI];
    data[1504] = 0.0 + k[98]*y[IDX_CNII] - k[1712]*y[IDX_CNI] -
        k[1713]*y[IDX_CNI];
    data[1505] = 0.0 - k[870]*y[IDX_CNI];
    data[1506] = 0.0 + k[439] + k[635]*y[IDX_CII] + k[1263]*y[IDX_HeII] +
        k[1609]*y[IDX_CI] + k[1774]*y[IDX_HI] + k[1911]*y[IDX_OI] + k[2065];
    data[1507] = 0.0 + k[340]*y[IDX_CNII] + k[1541]*y[IDX_HCNII] -
        k[1932]*y[IDX_CNI] - k[1933]*y[IDX_CNI] + k[1939]*y[IDX_HCNI];
    data[1508] = 0.0 - k[1519]*y[IDX_CNI];
    data[1509] = 0.0 + k[99]*y[IDX_CNII] + k[1117]*y[IDX_HCNII] -
        k[1714]*y[IDX_CNI];
    data[1510] = 0.0 + k[1832]*y[IDX_NI];
    data[1511] = 0.0 + k[1342]*y[IDX_NI];
    data[1512] = 0.0 - k[1715]*y[IDX_CNI];
    data[1513] = 0.0 - k[51]*y[IDX_CNII];
    data[1514] = 0.0 + k[631]*y[IDX_NHI] + k[2097]*y[IDX_NI];
    data[1515] = 0.0 - k[36]*y[IDX_CNII];
    data[1516] = 0.0 - k[47]*y[IDX_CNII];
    data[1517] = 0.0 - k[83]*y[IDX_CNII] + k[852]*y[IDX_NII];
    data[1518] = 0.0 + k[728]*y[IDX_NI] + k[731]*y[IDX_NHI];
    data[1519] = 0.0 - k[63]*y[IDX_CNII];
    data[1520] = 0.0 + k[1198]*y[IDX_HeII];
    data[1521] = 0.0 + k[100]*y[IDX_N2II] + k[159]*y[IDX_H2II] +
        k[237]*y[IDX_NII];
    data[1522] = 0.0 - k[36]*y[IDX_C2I] - k[47]*y[IDX_C2HI] -
        k[51]*y[IDX_CI] - k[63]*y[IDX_CH2I] - k[83]*y[IDX_CHI] -
        k[93]*y[IDX_COI] - k[94]*y[IDX_H2COI] - k[95]*y[IDX_HCNI] -
        k[96]*y[IDX_HCOI] - k[97]*y[IDX_NOI] - k[98]*y[IDX_O2I] -
        k[99]*y[IDX_SI] - k[191]*y[IDX_HI] - k[272]*y[IDX_NH2I] -
        k[294]*y[IDX_NHI] - k[326]*y[IDX_OI] - k[340]*y[IDX_OHI] -
        k[498]*y[IDX_EM] - k[865]*y[IDX_H2COI] - k[866]*y[IDX_HCNI] -
        k[867]*y[IDX_HCOI] - k[868]*y[IDX_O2I] - k[940]*y[IDX_H2I] -
        k[995]*y[IDX_H2OI] - k[996]*y[IDX_H2OI] - k[1332]*y[IDX_NI] - k[2235];
    data[1523] = 0.0 - k[93]*y[IDX_CNII];
    data[1524] = 0.0 - k[498]*y[IDX_CNII];
    data[1525] = 0.0 - k[191]*y[IDX_CNII];
    data[1526] = 0.0 - k[940]*y[IDX_CNII];
    data[1527] = 0.0 + k[159]*y[IDX_CNI];
    data[1528] = 0.0 - k[94]*y[IDX_CNII] - k[865]*y[IDX_CNII];
    data[1529] = 0.0 - k[995]*y[IDX_CNII] - k[996]*y[IDX_CNII];
    data[1530] = 0.0 - k[95]*y[IDX_CNII] - k[866]*y[IDX_CNII] +
        k[1231]*y[IDX_HeII];
    data[1531] = 0.0 - k[96]*y[IDX_CNII] - k[867]*y[IDX_CNII];
    data[1532] = 0.0 + k[1198]*y[IDX_CH3CNI] + k[1231]*y[IDX_HCNI] +
        k[1242]*y[IDX_HNCI] + k[1251]*y[IDX_NCCNI] + k[1262]*y[IDX_OCNI];
    data[1533] = 0.0 + k[1242]*y[IDX_HeII];
    data[1534] = 0.0 + k[728]*y[IDX_CHII] - k[1332]*y[IDX_CNII] +
        k[2097]*y[IDX_CII];
    data[1535] = 0.0 + k[237]*y[IDX_CNI] + k[852]*y[IDX_CHI];
    data[1536] = 0.0 + k[100]*y[IDX_CNI];
    data[1537] = 0.0 + k[1251]*y[IDX_HeII];
    data[1538] = 0.0 - k[294]*y[IDX_CNII] + k[631]*y[IDX_CII] +
        k[731]*y[IDX_CHII];
    data[1539] = 0.0 - k[272]*y[IDX_CNII];
    data[1540] = 0.0 - k[97]*y[IDX_CNII];
    data[1541] = 0.0 - k[326]*y[IDX_CNII];
    data[1542] = 0.0 - k[98]*y[IDX_CNII] - k[868]*y[IDX_CNII];
    data[1543] = 0.0 + k[1262]*y[IDX_HeII];
    data[1544] = 0.0 - k[340]*y[IDX_CNII];
    data[1545] = 0.0 - k[99]*y[IDX_CNII];
    data[1546] = 0.0 + k[2343] + k[2344] + k[2345] + k[2346];
    data[1547] = 0.0 + k[52]*y[IDX_COII] + k[694]*y[IDX_HCOII] +
        k[704]*y[IDX_SiOII] - k[1591]*y[IDX_COI] + k[1594]*y[IDX_HCOI] +
        k[1605]*y[IDX_NOI] + k[1608]*y[IDX_O2I] + k[1609]*y[IDX_OCNI] +
        k[1610]*y[IDX_OCSI] + k[1611]*y[IDX_OHI] + k[1614]*y[IDX_SO2I] +
        k[1616]*y[IDX_SOI] + k[1788]*y[IDX_HNCOI] + k[2104]*y[IDX_OI];
    data[1548] = 0.0 + k[616]*y[IDX_CO2I] + k[617]*y[IDX_H2COI] +
        k[626]*y[IDX_HCOI] + k[634]*y[IDX_O2I] + k[636]*y[IDX_OCSI] +
        k[638]*y[IDX_SO2I] + k[640]*y[IDX_SOI] + k[645]*y[IDX_SiOI];
    data[1549] = 0.0 + k[37]*y[IDX_COII] + k[653]*y[IDX_HCOII] +
        k[656]*y[IDX_O2II] + k[659]*y[IDX_SiOII] + k[1572]*y[IDX_O2I] +
        k[1572]*y[IDX_O2I] + k[1867]*y[IDX_OI];
    data[1550] = 0.0 + k[648]*y[IDX_HCOI] + k[649]*y[IDX_O2I];
    data[1551] = 0.0 + k[48]*y[IDX_COII] + k[680]*y[IDX_HCOII] +
        k[1581]*y[IDX_O2I] + k[1875]*y[IDX_OI];
    data[1552] = 0.0 + k[663]*y[IDX_HCOI];
    data[1553] = 0.0 + k[1482]*y[IDX_O2II] + k[1556]*y[IDX_SOII] +
        k[1574]*y[IDX_NOI] + k[1868]*y[IDX_OI] + k[1929]*y[IDX_OHI];
    data[1554] = 0.0 + k[1136]*y[IDX_HCOII];
    data[1555] = 0.0 + k[1876]*y[IDX_OI];
    data[1556] = 0.0 + k[1877]*y[IDX_OI];
    data[1557] = 0.0 + k[1878]*y[IDX_OI];
    data[1558] = 0.0 + k[84]*y[IDX_COII] + k[849]*y[IDX_HCOII] +
        k[1677]*y[IDX_CO2I] + k[1679]*y[IDX_HCOI] + k[1688]*y[IDX_O2I] +
        k[1689]*y[IDX_O2I] + k[1693]*y[IDX_OI] + k[1695]*y[IDX_OCSI] +
        k[1699]*y[IDX_SOI];
    data[1559] = 0.0 + k[714]*y[IDX_CO2I] + k[715]*y[IDX_H2COI] +
        k[726]*y[IDX_HCOI] + k[736]*y[IDX_OCSI];
    data[1560] = 0.0 + k[64]*y[IDX_COII] + k[763]*y[IDX_HCOII] +
        k[1625]*y[IDX_HCOI] + k[1634]*y[IDX_O2I] + k[1637]*y[IDX_OI] +
        k[1638]*y[IDX_OI];
    data[1561] = 0.0 + k[741]*y[IDX_CO2I] + k[747]*y[IDX_HCOI] +
        k[751]*y[IDX_OCSI];
    data[1562] = 0.0 + k[383] + k[1195]*y[IDX_HeII] + k[1740]*y[IDX_HI] +
        k[1983];
    data[1563] = 0.0 + k[1654]*y[IDX_HCOI] + k[1664]*y[IDX_OI];
    data[1564] = 0.0 + k[779]*y[IDX_HCOI] + k[785]*y[IDX_OCSI];
    data[1565] = 0.0 + k[1137]*y[IDX_HCOII];
    data[1566] = 0.0 + k[1138]*y[IDX_HCOII];
    data[1567] = 0.0 + k[1139]*y[IDX_HCOII];
    data[1568] = 0.0 + k[81]*y[IDX_COII];
    data[1569] = 0.0 - k[797]*y[IDX_COI];
    data[1570] = 0.0 - k[826]*y[IDX_COI];
    data[1571] = 0.0 + k[1706]*y[IDX_HCOI] + k[1710]*y[IDX_NOI] +
        k[1712]*y[IDX_O2I] + k[1880]*y[IDX_OI];
    data[1572] = 0.0 - k[93]*y[IDX_COI] + k[867]*y[IDX_HCOI] +
        k[868]*y[IDX_O2I];
    data[1573] = 0.0 - k[93]*y[IDX_CNII] - k[107]*y[IDX_N2II] -
        k[160]*y[IDX_H2II] - k[238]*y[IDX_NII] - k[310]*y[IDX_OII] - k[357] -
        k[394] - k[797]*y[IDX_CH4II] - k[826]*y[IDX_CH5II] -
        k[874]*y[IDX_H2ClII] - k[875]*y[IDX_HCO2II] - k[876]*y[IDX_HNOII] -
        k[877]*y[IDX_N2HII] - k[878]*y[IDX_O2HII] - k[879]*y[IDX_SO2II] -
        k[880]*y[IDX_SiH4II] - k[881]*y[IDX_SiOII] - k[919]*y[IDX_H2II] -
        k[978]*y[IDX_H2OII] - k[1037]*y[IDX_H3II] - k[1038]*y[IDX_H3II] -
        k[1111]*y[IDX_HCNII] - k[1213]*y[IDX_HeII] - k[1297]*y[IDX_NII] -
        k[1353]*y[IDX_NHII] - k[1521]*y[IDX_OHII] - k[1591]*y[IDX_CI] -
        k[1716]*y[IDX_HNOI] - k[1717]*y[IDX_NO2I] - k[1718]*y[IDX_O2I] -
        k[1719]*y[IDX_O2HI] - k[1745]*y[IDX_HI] - k[1934]*y[IDX_OHI] -
        k[1957]*y[IDX_SiI] - k[2004] - k[2169] - k[2207] - k[2296];
    data[1574] = 0.0 + k[37]*y[IDX_C2I] + k[48]*y[IDX_C2HI] +
        k[52]*y[IDX_CI] + k[64]*y[IDX_CH2I] + k[81]*y[IDX_CH4I] +
        k[84]*y[IDX_CHI] + k[101]*y[IDX_H2COI] + k[102]*y[IDX_H2SI] +
        k[103]*y[IDX_HCOI] + k[104]*y[IDX_NOI] + k[105]*y[IDX_O2I] +
        k[106]*y[IDX_SI] + k[187]*y[IDX_H2OI] + k[192]*y[IDX_HI] +
        k[200]*y[IDX_HCNI] + k[273]*y[IDX_NH2I] + k[283]*y[IDX_NH3I] +
        k[295]*y[IDX_NHI] + k[327]*y[IDX_OI] + k[341]*y[IDX_OHI];
    data[1575] = 0.0 + k[393] + k[616]*y[IDX_CII] + k[714]*y[IDX_CHII] +
        k[741]*y[IDX_CH2II] + k[1210]*y[IDX_HeII] + k[1351]*y[IDX_NHII] +
        k[1470]*y[IDX_OII] + k[1677]*y[IDX_CHI] + k[1744]*y[IDX_HI] +
        k[1808]*y[IDX_NI] + k[1882]*y[IDX_OI] + k[1956]*y[IDX_SiI] + k[2003];
    data[1576] = 0.0 + k[1140]*y[IDX_HCOII] + k[1883]*y[IDX_OI] +
        k[1935]*y[IDX_OHI];
    data[1577] = 0.0 + k[503]*y[IDX_H2COII] + k[504]*y[IDX_H2COII] +
        k[523]*y[IDX_H3COII] + k[542]*y[IDX_HCOII] + k[544]*y[IDX_HCO2II] +
        k[545]*y[IDX_HCO2II] + k[551]*y[IDX_HOCII] + k[581]*y[IDX_OCSII];
    data[1578] = 0.0 + k[192]*y[IDX_COII] + k[1740]*y[IDX_CH2COI] +
        k[1744]*y[IDX_CO2I] - k[1745]*y[IDX_COI] + k[1751]*y[IDX_HCOI] +
        k[1773]*y[IDX_OCNI] + k[1775]*y[IDX_OCSI];
    data[1579] = 0.0 + k[898]*y[IDX_HCOI] + k[900]*y[IDX_HNCOI] +
        k[904]*y[IDX_OCSI];
    data[1580] = 0.0 - k[160]*y[IDX_COI] - k[919]*y[IDX_COI] +
        k[925]*y[IDX_HCOI];
    data[1581] = 0.0 - k[874]*y[IDX_COI];
    data[1582] = 0.0 + k[101]*y[IDX_COII] + k[399] + k[617]*y[IDX_CII] +
        k[715]*y[IDX_CHII] + k[974]*y[IDX_SII] + k[1141]*y[IDX_HCOII] + k[2011]
        + k[2012];
    data[1583] = 0.0 + k[503]*y[IDX_EM] + k[504]*y[IDX_EM] +
        k[1158]*y[IDX_HCOI];
    data[1584] = 0.0 + k[1142]*y[IDX_HCOII];
    data[1585] = 0.0 + k[187]*y[IDX_COII] + k[1003]*y[IDX_HCOII];
    data[1586] = 0.0 - k[978]*y[IDX_COI] + k[984]*y[IDX_HCOI];
    data[1587] = 0.0 + k[102]*y[IDX_COII] + k[1143]*y[IDX_HCOII];
    data[1588] = 0.0 - k[1037]*y[IDX_COI] - k[1038]*y[IDX_COI];
    data[1589] = 0.0 + k[523]*y[IDX_EM];
    data[1590] = 0.0 + k[200]*y[IDX_COII] + k[1124]*y[IDX_HCOII] +
        k[1890]*y[IDX_OI] + k[1940]*y[IDX_OHI];
    data[1591] = 0.0 - k[1111]*y[IDX_COI] + k[1115]*y[IDX_HCOI];
    data[1592] = 0.0 + k[103]*y[IDX_COII] + k[409] + k[626]*y[IDX_CII] +
        k[648]*y[IDX_C2II] + k[663]*y[IDX_C2HII] + k[726]*y[IDX_CHII] +
        k[747]*y[IDX_CH2II] + k[779]*y[IDX_CH3II] + k[867]*y[IDX_CNII] +
        k[898]*y[IDX_HII] + k[925]*y[IDX_H2II] + k[984]*y[IDX_H2OII] +
        k[1115]*y[IDX_HCNII] + k[1144]*y[IDX_HCOII] + k[1158]*y[IDX_H2COII] +
        k[1161]*y[IDX_O2II] + k[1163]*y[IDX_SII] + k[1236]*y[IDX_HeII] +
        k[1303]*y[IDX_NII] + k[1317]*y[IDX_N2II] + k[1416]*y[IDX_NH3II] +
        k[1476]*y[IDX_OII] + k[1526]*y[IDX_OHII] + k[1594]*y[IDX_CI] +
        k[1625]*y[IDX_CH2I] + k[1654]*y[IDX_CH3I] + k[1679]*y[IDX_CHI] +
        k[1706]*y[IDX_CNI] + k[1751]*y[IDX_HI] + k[1780]*y[IDX_HCOI] +
        k[1780]*y[IDX_HCOI] + k[1780]*y[IDX_HCOI] + k[1780]*y[IDX_HCOI] +
        k[1781]*y[IDX_HCOI] + k[1781]*y[IDX_HCOI] + k[1783]*y[IDX_NOI] +
        k[1785]*y[IDX_O2I] + k[1811]*y[IDX_NI] + k[1893]*y[IDX_OI] +
        k[1941]*y[IDX_OHI] + k[1952]*y[IDX_SI] + k[2030];
    data[1593] = 0.0 + k[542]*y[IDX_EM] + k[653]*y[IDX_C2I] +
        k[680]*y[IDX_C2HI] + k[694]*y[IDX_CI] + k[763]*y[IDX_CH2I] +
        k[849]*y[IDX_CHI] + k[1003]*y[IDX_H2OI] + k[1124]*y[IDX_HCNI] +
        k[1136]*y[IDX_C2H5OHI] + k[1137]*y[IDX_CH3CCHI] + k[1138]*y[IDX_CH3CNI]
        + k[1139]*y[IDX_CH3OHI] + k[1140]*y[IDX_CSI] + k[1141]*y[IDX_H2COI] +
        k[1142]*y[IDX_H2CSI] + k[1143]*y[IDX_H2SI] + k[1144]*y[IDX_HCOI] +
        k[1145]*y[IDX_HCOOCH3I] + k[1146]*y[IDX_HS2I] + k[1147]*y[IDX_HSI] +
        k[1148]*y[IDX_NSI] + k[1149]*y[IDX_OCSI] + k[1150]*y[IDX_S2I] +
        k[1151]*y[IDX_SI] + k[1152]*y[IDX_SOI] + k[1153]*y[IDX_SiH2I] +
        k[1154]*y[IDX_SiH4I] + k[1155]*y[IDX_SiHI] + k[1156]*y[IDX_SiOI] +
        k[1157]*y[IDX_SiSI] + k[1168]*y[IDX_HNCI] + k[1406]*y[IDX_NH2I] +
        k[1433]*y[IDX_NH3I] + k[1452]*y[IDX_NHI] + k[1542]*y[IDX_OHI] +
        k[1566]*y[IDX_SiI];
    data[1594] = 0.0 + k[544]*y[IDX_EM] + k[545]*y[IDX_EM] -
        k[875]*y[IDX_COI];
    data[1595] = 0.0 + k[1145]*y[IDX_HCOII];
    data[1596] = 0.0 + k[1894]*y[IDX_OI];
    data[1597] = 0.0 + k[1195]*y[IDX_CH2COI] + k[1210]*y[IDX_CO2I] -
        k[1213]*y[IDX_COI] + k[1236]*y[IDX_HCOI] + k[1266]*y[IDX_OCSI];
    data[1598] = 0.0 + k[1168]*y[IDX_HCOII];
    data[1599] = 0.0 + k[415] + k[900]*y[IDX_HII] + k[1788]*y[IDX_CI] +
        k[2037];
    data[1600] = 0.0 - k[1716]*y[IDX_COI];
    data[1601] = 0.0 - k[876]*y[IDX_COI];
    data[1602] = 0.0 + k[551]*y[IDX_EM];
    data[1603] = 0.0 + k[1147]*y[IDX_HCOII];
    data[1604] = 0.0 + k[1146]*y[IDX_HCOII];
    data[1605] = 0.0 + k[1808]*y[IDX_CO2I] + k[1811]*y[IDX_HCOI];
    data[1606] = 0.0 - k[238]*y[IDX_COI] - k[1297]*y[IDX_COI] +
        k[1303]*y[IDX_HCOI] + k[1313]*y[IDX_OCSI];
    data[1607] = 0.0 - k[107]*y[IDX_COI] + k[1317]*y[IDX_HCOI] +
        k[1318]*y[IDX_OCSI];
    data[1608] = 0.0 - k[877]*y[IDX_COI];
    data[1609] = 0.0 + k[295]*y[IDX_COII] + k[1452]*y[IDX_HCOII];
    data[1610] = 0.0 + k[1351]*y[IDX_CO2I] - k[1353]*y[IDX_COI];
    data[1611] = 0.0 + k[273]*y[IDX_COII] + k[1406]*y[IDX_HCOII];
    data[1612] = 0.0 + k[283]*y[IDX_COII] + k[1433]*y[IDX_HCOII];
    data[1613] = 0.0 + k[1416]*y[IDX_HCOI];
    data[1614] = 0.0 + k[104]*y[IDX_COII] + k[1574]*y[IDX_C2H2I] +
        k[1605]*y[IDX_CI] + k[1710]*y[IDX_CNI] + k[1783]*y[IDX_HCOI];
    data[1615] = 0.0 - k[1717]*y[IDX_COI];
    data[1616] = 0.0 + k[1148]*y[IDX_HCOII];
    data[1617] = 0.0 + k[327]*y[IDX_COII] + k[1637]*y[IDX_CH2I] +
        k[1638]*y[IDX_CH2I] + k[1664]*y[IDX_CH3I] + k[1693]*y[IDX_CHI] +
        k[1867]*y[IDX_C2I] + k[1868]*y[IDX_C2H2I] + k[1875]*y[IDX_C2HI] +
        k[1876]*y[IDX_C2NI] + k[1877]*y[IDX_C3NI] + k[1878]*y[IDX_C4NI] +
        k[1880]*y[IDX_CNI] + k[1882]*y[IDX_CO2I] + k[1883]*y[IDX_CSI] +
        k[1890]*y[IDX_HCNI] + k[1893]*y[IDX_HCOI] + k[1894]*y[IDX_HCSI] +
        k[1910]*y[IDX_OCNI] + k[1913]*y[IDX_OCSI] + k[1918]*y[IDX_SiC2I] +
        k[1919]*y[IDX_SiC3I] + k[1920]*y[IDX_SiCI] + k[2104]*y[IDX_CI];
    data[1618] = 0.0 - k[310]*y[IDX_COI] + k[1470]*y[IDX_CO2I] +
        k[1476]*y[IDX_HCOI];
    data[1619] = 0.0 + k[105]*y[IDX_COII] + k[634]*y[IDX_CII] +
        k[649]*y[IDX_C2II] + k[868]*y[IDX_CNII] + k[1572]*y[IDX_C2I] +
        k[1572]*y[IDX_C2I] + k[1581]*y[IDX_C2HI] + k[1608]*y[IDX_CI] +
        k[1634]*y[IDX_CH2I] + k[1688]*y[IDX_CHI] + k[1689]*y[IDX_CHI] +
        k[1712]*y[IDX_CNI] - k[1718]*y[IDX_COI] + k[1785]*y[IDX_HCOI] +
        k[1864]*y[IDX_OCNI];
    data[1620] = 0.0 + k[656]*y[IDX_C2I] + k[1161]*y[IDX_HCOI] +
        k[1482]*y[IDX_C2H2I];
    data[1621] = 0.0 - k[1719]*y[IDX_COI];
    data[1622] = 0.0 - k[878]*y[IDX_COI];
    data[1623] = 0.0 + k[1609]*y[IDX_CI] + k[1773]*y[IDX_HI] +
        k[1864]*y[IDX_O2I] + k[1910]*y[IDX_OI];
    data[1624] = 0.0 + k[441] + k[636]*y[IDX_CII] + k[736]*y[IDX_CHII] +
        k[751]*y[IDX_CH2II] + k[785]*y[IDX_CH3II] + k[904]*y[IDX_HII] +
        k[1149]*y[IDX_HCOII] + k[1266]*y[IDX_HeII] + k[1313]*y[IDX_NII] +
        k[1318]*y[IDX_N2II] + k[1552]*y[IDX_SII] + k[1565]*y[IDX_SiII] +
        k[1610]*y[IDX_CI] + k[1695]*y[IDX_CHI] + k[1775]*y[IDX_HI] +
        k[1913]*y[IDX_OI] + k[2067];
    data[1625] = 0.0 + k[581]*y[IDX_EM];
    data[1626] = 0.0 + k[341]*y[IDX_COII] + k[1542]*y[IDX_HCOII] +
        k[1611]*y[IDX_CI] + k[1929]*y[IDX_C2H2I] - k[1934]*y[IDX_COI] +
        k[1935]*y[IDX_CSI] + k[1940]*y[IDX_HCNI] + k[1941]*y[IDX_HCOI];
    data[1627] = 0.0 - k[1521]*y[IDX_COI] + k[1526]*y[IDX_HCOI];
    data[1628] = 0.0 + k[106]*y[IDX_COII] + k[1151]*y[IDX_HCOII] +
        k[1952]*y[IDX_HCOI];
    data[1629] = 0.0 + k[974]*y[IDX_H2COI] + k[1163]*y[IDX_HCOI] +
        k[1552]*y[IDX_OCSI];
    data[1630] = 0.0 + k[1150]*y[IDX_HCOII];
    data[1631] = 0.0 + k[1566]*y[IDX_HCOII] + k[1956]*y[IDX_CO2I] -
        k[1957]*y[IDX_COI];
    data[1632] = 0.0 + k[1565]*y[IDX_OCSI];
    data[1633] = 0.0 + k[1920]*y[IDX_OI];
    data[1634] = 0.0 + k[1918]*y[IDX_OI];
    data[1635] = 0.0 + k[1919]*y[IDX_OI];
    data[1636] = 0.0 + k[1155]*y[IDX_HCOII];
    data[1637] = 0.0 + k[1153]*y[IDX_HCOII];
    data[1638] = 0.0 + k[1154]*y[IDX_HCOII];
    data[1639] = 0.0 - k[880]*y[IDX_COI];
    data[1640] = 0.0 + k[645]*y[IDX_CII] + k[1156]*y[IDX_HCOII];
    data[1641] = 0.0 + k[659]*y[IDX_C2I] + k[704]*y[IDX_CI] -
        k[881]*y[IDX_COI];
    data[1642] = 0.0 + k[1157]*y[IDX_HCOII];
    data[1643] = 0.0 + k[640]*y[IDX_CII] + k[1152]*y[IDX_HCOII] +
        k[1616]*y[IDX_CI] + k[1699]*y[IDX_CHI];
    data[1644] = 0.0 + k[1556]*y[IDX_C2H2I];
    data[1645] = 0.0 + k[638]*y[IDX_CII] + k[1614]*y[IDX_CI];
    data[1646] = 0.0 - k[879]*y[IDX_COI];
    data[1647] = 0.0 - k[52]*y[IDX_COII] + k[700]*y[IDX_O2II] +
        k[2103]*y[IDX_OII];
    data[1648] = 0.0 + k[616]*y[IDX_CO2I] + k[633]*y[IDX_O2I] +
        k[635]*y[IDX_OCNI] + k[637]*y[IDX_OHI] + k[641]*y[IDX_SOI] +
        k[2098]*y[IDX_OI];
    data[1649] = 0.0 - k[37]*y[IDX_COII] + k[656]*y[IDX_O2II] +
        k[1463]*y[IDX_OII];
    data[1650] = 0.0 + k[649]*y[IDX_O2I] + k[1490]*y[IDX_OI];
    data[1651] = 0.0 - k[48]*y[IDX_COII] - k[677]*y[IDX_COII] +
        k[1465]*y[IDX_OII];
    data[1652] = 0.0 - k[84]*y[IDX_COII] - k[841]*y[IDX_COII] +
        k[857]*y[IDX_OII];
    data[1653] = 0.0 + k[732]*y[IDX_O2I] + k[735]*y[IDX_OI] +
        k[738]*y[IDX_OHI];
    data[1654] = 0.0 - k[64]*y[IDX_COII] - k[756]*y[IDX_COII];
    data[1655] = 0.0 + k[1194]*y[IDX_HeII];
    data[1656] = 0.0 - k[81]*y[IDX_COII] - k[807]*y[IDX_COII];
    data[1657] = 0.0 + k[93]*y[IDX_COI];
    data[1658] = 0.0 + k[93]*y[IDX_CNII] + k[107]*y[IDX_N2II] +
        k[160]*y[IDX_H2II] + k[238]*y[IDX_NII] + k[310]*y[IDX_OII] + k[357];
    data[1659] = 0.0 - k[37]*y[IDX_C2I] - k[48]*y[IDX_C2HI] -
        k[52]*y[IDX_CI] - k[64]*y[IDX_CH2I] - k[81]*y[IDX_CH4I] -
        k[84]*y[IDX_CHI] - k[101]*y[IDX_H2COI] - k[102]*y[IDX_H2SI] -
        k[103]*y[IDX_HCOI] - k[104]*y[IDX_NOI] - k[105]*y[IDX_O2I] -
        k[106]*y[IDX_SI] - k[187]*y[IDX_H2OI] - k[192]*y[IDX_HI] -
        k[200]*y[IDX_HCNI] - k[273]*y[IDX_NH2I] - k[283]*y[IDX_NH3I] -
        k[295]*y[IDX_NHI] - k[327]*y[IDX_OI] - k[341]*y[IDX_OHI] -
        k[499]*y[IDX_EM] - k[677]*y[IDX_C2HI] - k[756]*y[IDX_CH2I] -
        k[807]*y[IDX_CH4I] - k[841]*y[IDX_CHI] - k[871]*y[IDX_H2COI] -
        k[872]*y[IDX_H2SI] - k[873]*y[IDX_SO2I] - k[941]*y[IDX_H2I] -
        k[942]*y[IDX_H2I] - k[997]*y[IDX_H2OI] - k[1398]*y[IDX_NH2I] -
        k[1423]*y[IDX_NH3I] - k[1448]*y[IDX_NHI] - k[1539]*y[IDX_OHI] - k[2002]
        - k[2234];
    data[1660] = 0.0 + k[616]*y[IDX_CII] + k[1209]*y[IDX_HeII] +
        k[1296]*y[IDX_NII];
    data[1661] = 0.0 + k[1496]*y[IDX_OI];
    data[1662] = 0.0 - k[499]*y[IDX_COII];
    data[1663] = 0.0 - k[192]*y[IDX_COII];
    data[1664] = 0.0 + k[892]*y[IDX_H2COI] + k[897]*y[IDX_HCOI];
    data[1665] = 0.0 - k[941]*y[IDX_COII] - k[942]*y[IDX_COII];
    data[1666] = 0.0 + k[160]*y[IDX_COI];
    data[1667] = 0.0 - k[101]*y[IDX_COII] - k[871]*y[IDX_COII] +
        k[892]*y[IDX_HII] + k[1216]*y[IDX_HeII];
    data[1668] = 0.0 - k[187]*y[IDX_COII] - k[997]*y[IDX_COII];
    data[1669] = 0.0 - k[102]*y[IDX_COII] - k[872]*y[IDX_COII];
    data[1670] = 0.0 - k[200]*y[IDX_COII];
    data[1671] = 0.0 - k[103]*y[IDX_COII] + k[897]*y[IDX_HII] +
        k[1235]*y[IDX_HeII];
    data[1672] = 0.0 + k[2029];
    data[1673] = 0.0 + k[1194]*y[IDX_CH2COI] + k[1209]*y[IDX_CO2I] +
        k[1216]*y[IDX_H2COI] + k[1235]*y[IDX_HCOI] + k[1267]*y[IDX_OCSI];
    data[1674] = 0.0 + k[238]*y[IDX_COI] + k[1296]*y[IDX_CO2I];
    data[1675] = 0.0 + k[107]*y[IDX_COI];
    data[1676] = 0.0 - k[295]*y[IDX_COII] - k[1448]*y[IDX_COII];
    data[1677] = 0.0 - k[273]*y[IDX_COII] - k[1398]*y[IDX_COII];
    data[1678] = 0.0 - k[283]*y[IDX_COII] - k[1423]*y[IDX_COII];
    data[1679] = 0.0 - k[104]*y[IDX_COII];
    data[1680] = 0.0 - k[327]*y[IDX_COII] + k[735]*y[IDX_CHII] +
        k[1490]*y[IDX_C2II] + k[1496]*y[IDX_CSII] + k[2098]*y[IDX_CII];
    data[1681] = 0.0 + k[310]*y[IDX_COI] + k[857]*y[IDX_CHI] +
        k[1463]*y[IDX_C2I] + k[1465]*y[IDX_C2HI] + k[2103]*y[IDX_CI];
    data[1682] = 0.0 - k[105]*y[IDX_COII] + k[633]*y[IDX_CII] +
        k[649]*y[IDX_C2II] + k[732]*y[IDX_CHII];
    data[1683] = 0.0 + k[656]*y[IDX_C2I] + k[700]*y[IDX_CI];
    data[1684] = 0.0 + k[635]*y[IDX_CII];
    data[1685] = 0.0 + k[1267]*y[IDX_HeII];
    data[1686] = 0.0 - k[341]*y[IDX_COII] + k[637]*y[IDX_CII] +
        k[738]*y[IDX_CHII] - k[1539]*y[IDX_COII];
    data[1687] = 0.0 - k[106]*y[IDX_COII];
    data[1688] = 0.0 + k[641]*y[IDX_CII];
    data[1689] = 0.0 - k[873]*y[IDX_COII];
    data[1690] = 0.0 + k[2435] + k[2436] + k[2437] + k[2438];
    data[1691] = 0.0 + k[695]*y[IDX_HCO2II];
    data[1692] = 0.0 - k[616]*y[IDX_CO2I];
    data[1693] = 0.0 - k[1677]*y[IDX_CO2I] + k[1687]*y[IDX_O2I];
    data[1694] = 0.0 - k[714]*y[IDX_CO2I];
    data[1695] = 0.0 + k[1632]*y[IDX_O2I] + k[1633]*y[IDX_O2I];
    data[1696] = 0.0 - k[741]*y[IDX_CO2I];
    data[1697] = 0.0 + k[791]*y[IDX_HCO2II];
    data[1698] = 0.0 + k[812]*y[IDX_HCO2II];
    data[1699] = 0.0 - k[796]*y[IDX_CO2I];
    data[1700] = 0.0 - k[825]*y[IDX_CO2I];
    data[1701] = 0.0 + k[875]*y[IDX_HCO2II] + k[879]*y[IDX_SO2II] +
        k[881]*y[IDX_SiOII] + k[1716]*y[IDX_HNOI] + k[1717]*y[IDX_NO2I] +
        k[1718]*y[IDX_O2I] + k[1719]*y[IDX_O2HI] + k[1934]*y[IDX_OHI];
    data[1702] = 0.0 + k[873]*y[IDX_SO2I];
    data[1703] = 0.0 - k[393] - k[616]*y[IDX_CII] - k[714]*y[IDX_CHII] -
        k[741]*y[IDX_CH2II] - k[796]*y[IDX_CH4II] - k[825]*y[IDX_CH5II] -
        k[891]*y[IDX_HII] - k[918]*y[IDX_H2II] - k[1036]*y[IDX_H3II] -
        k[1110]*y[IDX_HCNII] - k[1173]*y[IDX_HNOII] - k[1209]*y[IDX_HeII] -
        k[1210]*y[IDX_HeII] - k[1211]*y[IDX_HeII] - k[1212]*y[IDX_HeII] -
        k[1296]*y[IDX_NII] - k[1322]*y[IDX_N2HII] - k[1350]*y[IDX_NHII] -
        k[1351]*y[IDX_NHII] - k[1352]*y[IDX_NHII] - k[1470]*y[IDX_OII] -
        k[1489]*y[IDX_O2HII] - k[1520]*y[IDX_OHII] - k[1677]*y[IDX_CHI] -
        k[1744]*y[IDX_HI] - k[1808]*y[IDX_NI] - k[1882]*y[IDX_OI] -
        k[1956]*y[IDX_SiI] - k[2003] - k[2215];
    data[1704] = 0.0 + k[543]*y[IDX_HCO2II];
    data[1705] = 0.0 - k[1744]*y[IDX_CO2I];
    data[1706] = 0.0 - k[891]*y[IDX_CO2I];
    data[1707] = 0.0 - k[918]*y[IDX_CO2I];
    data[1708] = 0.0 + k[1004]*y[IDX_HCO2II];
    data[1709] = 0.0 - k[1036]*y[IDX_CO2I];
    data[1710] = 0.0 - k[1110]*y[IDX_CO2I];
    data[1711] = 0.0 + k[1784]*y[IDX_O2I] + k[1892]*y[IDX_OI];
    data[1712] = 0.0 + k[543]*y[IDX_EM] + k[695]*y[IDX_CI] +
        k[791]*y[IDX_CH3CNI] + k[812]*y[IDX_CH4I] + k[875]*y[IDX_COI] +
        k[1004]*y[IDX_H2OI] + k[1434]*y[IDX_NH3I];
    data[1713] = 0.0 + k[411];
    data[1714] = 0.0 - k[1209]*y[IDX_CO2I] - k[1210]*y[IDX_CO2I] -
        k[1211]*y[IDX_CO2I] - k[1212]*y[IDX_CO2I];
    data[1715] = 0.0 + k[1716]*y[IDX_COI];
    data[1716] = 0.0 - k[1173]*y[IDX_CO2I];
    data[1717] = 0.0 - k[1808]*y[IDX_CO2I];
    data[1718] = 0.0 - k[1296]*y[IDX_CO2I];
    data[1719] = 0.0 - k[1322]*y[IDX_CO2I];
    data[1720] = 0.0 - k[1350]*y[IDX_CO2I] - k[1351]*y[IDX_CO2I] -
        k[1352]*y[IDX_CO2I];
    data[1721] = 0.0 + k[1434]*y[IDX_HCO2II];
    data[1722] = 0.0 + k[1860]*y[IDX_OCNI];
    data[1723] = 0.0 + k[1717]*y[IDX_COI];
    data[1724] = 0.0 - k[1882]*y[IDX_CO2I] + k[1892]*y[IDX_HCOI] +
        k[1912]*y[IDX_OCSI];
    data[1725] = 0.0 - k[1470]*y[IDX_CO2I] + k[1479]*y[IDX_OCSI];
    data[1726] = 0.0 + k[1632]*y[IDX_CH2I] + k[1633]*y[IDX_CH2I] +
        k[1687]*y[IDX_CHI] + k[1718]*y[IDX_COI] + k[1784]*y[IDX_HCOI] +
        k[1863]*y[IDX_OCNI];
    data[1727] = 0.0 + k[1719]*y[IDX_COI];
    data[1728] = 0.0 - k[1489]*y[IDX_CO2I];
    data[1729] = 0.0 + k[1860]*y[IDX_NOI] + k[1863]*y[IDX_O2I];
    data[1730] = 0.0 + k[1479]*y[IDX_OII] + k[1562]*y[IDX_SOII] +
        k[1912]*y[IDX_OI];
    data[1731] = 0.0 + k[1934]*y[IDX_COI];
    data[1732] = 0.0 - k[1520]*y[IDX_CO2I];
    data[1733] = 0.0 - k[1956]*y[IDX_CO2I];
    data[1734] = 0.0 + k[881]*y[IDX_COI];
    data[1735] = 0.0 + k[1562]*y[IDX_OCSI];
    data[1736] = 0.0 + k[873]*y[IDX_COII];
    data[1737] = 0.0 + k[879]*y[IDX_COI];
    data[1738] = 0.0 + k[2431] + k[2432] + k[2433] + k[2434];
    data[1739] = 0.0 - k[1592]*y[IDX_CSI] + k[1595]*y[IDX_HSI] +
        k[1606]*y[IDX_NSI] + k[1610]*y[IDX_OCSI] + k[1613]*y[IDX_S2I] +
        k[1615]*y[IDX_SOI] + k[2106]*y[IDX_SI];
    data[1740] = 0.0 + k[619]*y[IDX_H2CSI];
    data[1741] = 0.0 + k[1573]*y[IDX_SI];
    data[1742] = 0.0 + k[1164]*y[IDX_HCSII];
    data[1743] = 0.0 + k[1695]*y[IDX_OCSI] + k[1697]*y[IDX_SI];
    data[1744] = 0.0 + k[1644]*y[IDX_SI];
    data[1745] = 0.0 - k[118]*y[IDX_HII] - k[395] - k[396] -
        k[1039]*y[IDX_H3II] - k[1085]*y[IDX_H3OII] - k[1140]*y[IDX_HCOII] -
        k[1214]*y[IDX_HeII] - k[1215]*y[IDX_HeII] - k[1592]*y[IDX_CI] -
        k[1809]*y[IDX_NI] - k[1883]*y[IDX_OI] - k[1884]*y[IDX_OI] -
        k[1935]*y[IDX_OHI] - k[1936]*y[IDX_OHI] - k[2006] - k[2007] - k[2255];
    data[1746] = 0.0 + k[221]*y[IDX_MgI] + k[348]*y[IDX_SiI];
    data[1747] = 0.0 + k[506]*y[IDX_H2CSII] + k[526]*y[IDX_H3CSII] +
        k[547]*y[IDX_HCSII] + k[552]*y[IDX_HOCSII] + k[580]*y[IDX_OCSII];
    data[1748] = 0.0 + k[1753]*y[IDX_HCSI];
    data[1749] = 0.0 - k[118]*y[IDX_CSI];
    data[1750] = 0.0 + k[400] + k[619]*y[IDX_CII] + k[2015];
    data[1751] = 0.0 + k[506]*y[IDX_EM];
    data[1752] = 0.0 - k[1039]*y[IDX_CSI];
    data[1753] = 0.0 + k[526]*y[IDX_EM];
    data[1754] = 0.0 - k[1085]*y[IDX_CSI];
    data[1755] = 0.0 - k[1140]*y[IDX_CSI];
    data[1756] = 0.0 + k[1240]*y[IDX_HeII] + k[1753]*y[IDX_HI];
    data[1757] = 0.0 + k[547]*y[IDX_EM] + k[1164]*y[IDX_C2H5OHI] +
        k[1435]*y[IDX_NH3I];
    data[1758] = 0.0 - k[1214]*y[IDX_CSI] - k[1215]*y[IDX_CSI] +
        k[1240]*y[IDX_HCSI] + k[1265]*y[IDX_OCSI];
    data[1759] = 0.0 + k[552]*y[IDX_EM];
    data[1760] = 0.0 + k[1595]*y[IDX_CI];
    data[1761] = 0.0 + k[221]*y[IDX_CSII];
    data[1762] = 0.0 - k[1809]*y[IDX_CSI];
    data[1763] = 0.0 + k[1435]*y[IDX_HCSII];
    data[1764] = 0.0 + k[1606]*y[IDX_CI];
    data[1765] = 0.0 - k[1883]*y[IDX_CSI] - k[1884]*y[IDX_CSI];
    data[1766] = 0.0 + k[1265]*y[IDX_HeII] + k[1610]*y[IDX_CI] +
        k[1695]*y[IDX_CHI];
    data[1767] = 0.0 + k[580]*y[IDX_EM];
    data[1768] = 0.0 - k[1935]*y[IDX_CSI] - k[1936]*y[IDX_CSI];
    data[1769] = 0.0 + k[1573]*y[IDX_C2I] + k[1644]*y[IDX_CH2I] +
        k[1697]*y[IDX_CHI] + k[2106]*y[IDX_CI];
    data[1770] = 0.0 + k[1613]*y[IDX_CI];
    data[1771] = 0.0 + k[348]*y[IDX_CSII];
    data[1772] = 0.0 + k[1615]*y[IDX_CI];
    data[1773] = 0.0 + k[697]*y[IDX_HSII] + k[2105]*y[IDX_SII];
    data[1774] = 0.0 + k[628]*y[IDX_HSI] + k[632]*y[IDX_NSI] +
        k[636]*y[IDX_OCSI] + k[639]*y[IDX_SOI] + k[2099]*y[IDX_SI];
    data[1775] = 0.0 + k[658]*y[IDX_SII];
    data[1776] = 0.0 + k[650]*y[IDX_SI];
    data[1777] = 0.0 + k[861]*y[IDX_SII];
    data[1778] = 0.0 + k[739]*y[IDX_SI];
    data[1779] = 0.0 - k[808]*y[IDX_CSII];
    data[1780] = 0.0 + k[118]*y[IDX_HII] + k[395] + k[2006];
    data[1781] = 0.0 - k[221]*y[IDX_MgI] - k[348]*y[IDX_SiI] -
        k[500]*y[IDX_EM] - k[808]*y[IDX_CH4I] - k[943]*y[IDX_H2I] -
        k[1485]*y[IDX_O2I] - k[1496]*y[IDX_OI] - k[2005] - k[2266];
    data[1782] = 0.0 - k[500]*y[IDX_CSII];
    data[1783] = 0.0 + k[118]*y[IDX_CSI] + k[899]*y[IDX_HCSI];
    data[1784] = 0.0 - k[943]*y[IDX_CSII];
    data[1785] = 0.0 + k[1219]*y[IDX_HeII];
    data[1786] = 0.0 + k[899]*y[IDX_HII] + k[1239]*y[IDX_HeII];
    data[1787] = 0.0 + k[1219]*y[IDX_H2CSI] + k[1239]*y[IDX_HCSI] +
        k[1264]*y[IDX_OCSI];
    data[1788] = 0.0 + k[628]*y[IDX_CII];
    data[1789] = 0.0 + k[697]*y[IDX_CI];
    data[1790] = 0.0 - k[221]*y[IDX_CSII];
    data[1791] = 0.0 + k[1312]*y[IDX_OCSI];
    data[1792] = 0.0 + k[632]*y[IDX_CII];
    data[1793] = 0.0 - k[1496]*y[IDX_CSII];
    data[1794] = 0.0 - k[1485]*y[IDX_CSII];
    data[1795] = 0.0 + k[636]*y[IDX_CII] + k[1264]*y[IDX_HeII] +
        k[1312]*y[IDX_NII];
    data[1796] = 0.0 + k[650]*y[IDX_C2II] + k[739]*y[IDX_CHII] +
        k[2099]*y[IDX_CII];
    data[1797] = 0.0 + k[658]*y[IDX_C2I] + k[861]*y[IDX_CHI] +
        k[2105]*y[IDX_CI];
    data[1798] = 0.0 - k[348]*y[IDX_CSII];
    data[1799] = 0.0 + k[639]*y[IDX_CII];
    data[1800] = 0.0 + k[356] + k[379] + k[1976];
    data[1801] = 0.0 - k[2132]*y[IDX_EM];
    data[1802] = 0.0 + k[1961];
    data[1803] = 0.0 - k[458]*y[IDX_EM];
    data[1804] = 0.0 + k[374] + k[1971];
    data[1805] = 0.0 - k[459]*y[IDX_EM] - k[460]*y[IDX_EM];
    data[1806] = 0.0 + k[367] + k[1964];
    data[1807] = 0.0 - k[461]*y[IDX_EM] - k[462]*y[IDX_EM] -
        k[463]*y[IDX_EM];
    data[1808] = 0.0 - k[464]*y[IDX_EM] - k[465]*y[IDX_EM] -
        k[466]*y[IDX_EM] - k[467]*y[IDX_EM];
    data[1809] = 0.0 - k[468]*y[IDX_EM] - k[469]*y[IDX_EM];
    data[1810] = 0.0 - k[470]*y[IDX_EM] - k[471]*y[IDX_EM];
    data[1811] = 0.0 - k[472]*y[IDX_EM];
    data[1812] = 0.0 - k[473]*y[IDX_EM];
    data[1813] = 0.0 - k[474]*y[IDX_EM];
    data[1814] = 0.0 - k[475]*y[IDX_EM] - k[476]*y[IDX_EM];
    data[1815] = 0.0 + k[0]*y[IDX_OI] + k[2000];
    data[1816] = 0.0 - k[477]*y[IDX_EM];
    data[1817] = 0.0 + k[381] + k[1981];
    data[1818] = 0.0 - k[478]*y[IDX_EM] - k[479]*y[IDX_EM] -
        k[480]*y[IDX_EM];
    data[1819] = 0.0 + k[385] + k[1987];
    data[1820] = 0.0 - k[481]*y[IDX_EM] - k[482]*y[IDX_EM] -
        k[483]*y[IDX_EM] - k[2133]*y[IDX_EM];
    data[1821] = 0.0 - k[484]*y[IDX_EM] - k[485]*y[IDX_EM];
    data[1822] = 0.0 + k[1991];
    data[1823] = 0.0 - k[486]*y[IDX_EM] - k[487]*y[IDX_EM] -
        k[488]*y[IDX_EM] - k[489]*y[IDX_EM] - k[490]*y[IDX_EM];
    data[1824] = 0.0 + k[1997];
    data[1825] = 0.0 - k[491]*y[IDX_EM] - k[492]*y[IDX_EM];
    data[1826] = 0.0 - k[493]*y[IDX_EM] - k[494]*y[IDX_EM] -
        k[495]*y[IDX_EM] - k[496]*y[IDX_EM] - k[497]*y[IDX_EM];
    data[1827] = 0.0 + k[358] + k[397] + k[2008];
    data[1828] = 0.0 - k[2134]*y[IDX_EM];
    data[1829] = 0.0 - k[498]*y[IDX_EM];
    data[1830] = 0.0 + k[357];
    data[1831] = 0.0 - k[499]*y[IDX_EM];
    data[1832] = 0.0 + k[395] + k[2006];
    data[1833] = 0.0 - k[500]*y[IDX_EM];
    data[1834] = 0.0 - k[8]*y[IDX_H2I] + k[8]*y[IDX_H2I] -
        k[458]*y[IDX_C2II] - k[459]*y[IDX_C2HII] - k[460]*y[IDX_C2HII] -
        k[461]*y[IDX_C2H2II] - k[462]*y[IDX_C2H2II] - k[463]*y[IDX_C2H2II] -
        k[464]*y[IDX_C2H5OH2II] - k[465]*y[IDX_C2H5OH2II] -
        k[466]*y[IDX_C2H5OH2II] - k[467]*y[IDX_C2H5OH2II] - k[468]*y[IDX_C2NII]
        - k[469]*y[IDX_C2NII] - k[470]*y[IDX_C2N2II] - k[471]*y[IDX_C2N2II] -
        k[472]*y[IDX_C2NHII] - k[473]*y[IDX_C3II] - k[474]*y[IDX_C3H5II] -
        k[475]*y[IDX_C4NII] - k[476]*y[IDX_C4NII] - k[477]*y[IDX_CHII] -
        k[478]*y[IDX_CH2II] - k[479]*y[IDX_CH2II] - k[480]*y[IDX_CH2II] -
        k[481]*y[IDX_CH3II] - k[482]*y[IDX_CH3II] - k[483]*y[IDX_CH3II] -
        k[484]*y[IDX_CH3CNHII] - k[485]*y[IDX_CH3CNHII] - k[486]*y[IDX_CH3OH2II]
        - k[487]*y[IDX_CH3OH2II] - k[488]*y[IDX_CH3OH2II] -
        k[489]*y[IDX_CH3OH2II] - k[490]*y[IDX_CH3OH2II] - k[491]*y[IDX_CH4II] -
        k[492]*y[IDX_CH4II] - k[493]*y[IDX_CH5II] - k[494]*y[IDX_CH5II] -
        k[495]*y[IDX_CH5II] - k[496]*y[IDX_CH5II] - k[497]*y[IDX_CH5II] -
        k[498]*y[IDX_CNII] - k[499]*y[IDX_COII] - k[500]*y[IDX_CSII] -
        k[501]*y[IDX_H2II] - k[502]*y[IDX_H2COII] - k[503]*y[IDX_H2COII] -
        k[504]*y[IDX_H2COII] - k[505]*y[IDX_H2COII] - k[506]*y[IDX_H2CSII] -
        k[507]*y[IDX_H2CSII] - k[508]*y[IDX_H2ClII] - k[509]*y[IDX_H2ClII] -
        k[510]*y[IDX_H2NOII] - k[511]*y[IDX_H2NOII] - k[512]*y[IDX_H2OII] -
        k[513]*y[IDX_H2OII] - k[514]*y[IDX_H2OII] - k[515]*y[IDX_H2SII] -
        k[516]*y[IDX_H2SII] - k[517]*y[IDX_H2S2II] - k[518]*y[IDX_H2S2II] -
        k[519]*y[IDX_H3II] - k[520]*y[IDX_H3II] - k[521]*y[IDX_H3COII] -
        k[522]*y[IDX_H3COII] - k[523]*y[IDX_H3COII] - k[524]*y[IDX_H3COII] -
        k[525]*y[IDX_H3COII] - k[526]*y[IDX_H3CSII] - k[527]*y[IDX_H3CSII] -
        k[528]*y[IDX_H3OII] - k[529]*y[IDX_H3OII] - k[530]*y[IDX_H3OII] -
        k[531]*y[IDX_H3OII] - k[532]*y[IDX_H3SII] - k[533]*y[IDX_H3SII] -
        k[534]*y[IDX_H3SII] - k[535]*y[IDX_H3SII] - k[536]*y[IDX_H5C2O2II] -
        k[537]*y[IDX_H5C2O2II] - k[538]*y[IDX_HCNII] - k[539]*y[IDX_HCNHII] -
        k[540]*y[IDX_HCNHII] - k[541]*y[IDX_HCNHII] - k[542]*y[IDX_HCOII] -
        k[543]*y[IDX_HCO2II] - k[544]*y[IDX_HCO2II] - k[545]*y[IDX_HCO2II] -
        k[546]*y[IDX_HCSII] - k[547]*y[IDX_HCSII] - k[548]*y[IDX_HClII] -
        k[549]*y[IDX_HNOII] - k[550]*y[IDX_HNSII] - k[551]*y[IDX_HOCII] -
        k[552]*y[IDX_HOCSII] - k[553]*y[IDX_HOCSII] - k[554]*y[IDX_HSII] -
        k[555]*y[IDX_HS2II] - k[556]*y[IDX_HS2II] - k[557]*y[IDX_HSOII] -
        k[558]*y[IDX_HSO2II] - k[559]*y[IDX_HSO2II] - k[560]*y[IDX_HSO2II] -
        k[561]*y[IDX_HSiSII] - k[562]*y[IDX_HSiSII] - k[563]*y[IDX_HeHII] -
        k[564]*y[IDX_N2II] - k[565]*y[IDX_N2HII] - k[566]*y[IDX_N2HII] -
        k[567]*y[IDX_NHII] - k[568]*y[IDX_NH2II] - k[569]*y[IDX_NH2II] -
        k[570]*y[IDX_NH3II] - k[571]*y[IDX_NH3II] - k[572]*y[IDX_NH4II] -
        k[573]*y[IDX_NH4II] - k[574]*y[IDX_NH4II] - k[575]*y[IDX_NOII] -
        k[576]*y[IDX_NSII] - k[577]*y[IDX_O2II] - k[578]*y[IDX_O2HII] -
        k[579]*y[IDX_OCSII] - k[580]*y[IDX_OCSII] - k[581]*y[IDX_OCSII] -
        k[582]*y[IDX_OHII] - k[583]*y[IDX_S2II] - k[584]*y[IDX_SOII] -
        k[585]*y[IDX_SO2II] - k[586]*y[IDX_SO2II] - k[587]*y[IDX_SiCII] -
        k[588]*y[IDX_SiC2II] - k[589]*y[IDX_SiC2II] - k[590]*y[IDX_SiC3II] -
        k[591]*y[IDX_SiC3II] - k[592]*y[IDX_SiHII] - k[593]*y[IDX_SiH2II] -
        k[594]*y[IDX_SiH2II] - k[595]*y[IDX_SiH2II] - k[596]*y[IDX_SiH3II] -
        k[597]*y[IDX_SiH3II] - k[598]*y[IDX_SiH4II] - k[599]*y[IDX_SiH4II] -
        k[600]*y[IDX_SiH5II] - k[601]*y[IDX_SiH5II] - k[602]*y[IDX_SiOII] -
        k[603]*y[IDX_SiOHII] - k[604]*y[IDX_SiOHII] - k[605]*y[IDX_SiSII] -
        k[2132]*y[IDX_CII] - k[2133]*y[IDX_CH3II] - k[2134]*y[IDX_ClII] -
        k[2135]*y[IDX_HII] - k[2136]*y[IDX_H2COII] - k[2137]*y[IDX_H2CSII] -
        k[2138]*y[IDX_H2SII] - k[2139]*y[IDX_HeII] - k[2140]*y[IDX_MgII] -
        k[2141]*y[IDX_NII] - k[2142]*y[IDX_OII] - k[2143]*y[IDX_SII] -
        k[2144]*y[IDX_SiII] - k[2302];
    data[1835] = 0.0 + k[362] + k[406];
    data[1836] = 0.0 - k[2135]*y[IDX_EM];
    data[1837] = 0.0 - k[8]*y[IDX_EM] + k[8]*y[IDX_EM] + k[359] + k[360];
    data[1838] = 0.0 - k[501]*y[IDX_EM];
    data[1839] = 0.0 - k[508]*y[IDX_EM] - k[509]*y[IDX_EM];
    data[1840] = 0.0 + k[2013] + k[2014];
    data[1841] = 0.0 - k[502]*y[IDX_EM] - k[503]*y[IDX_EM] -
        k[504]*y[IDX_EM] - k[505]*y[IDX_EM] - k[2136]*y[IDX_EM];
    data[1842] = 0.0 - k[506]*y[IDX_EM] - k[507]*y[IDX_EM] -
        k[2137]*y[IDX_EM];
    data[1843] = 0.0 - k[510]*y[IDX_EM] - k[511]*y[IDX_EM];
    data[1844] = 0.0 + k[2017];
    data[1845] = 0.0 - k[512]*y[IDX_EM] - k[513]*y[IDX_EM] -
        k[514]*y[IDX_EM];
    data[1846] = 0.0 + k[403] + k[2020];
    data[1847] = 0.0 - k[515]*y[IDX_EM] - k[516]*y[IDX_EM] -
        k[2138]*y[IDX_EM];
    data[1848] = 0.0 - k[517]*y[IDX_EM] - k[518]*y[IDX_EM];
    data[1849] = 0.0 - k[519]*y[IDX_EM] - k[520]*y[IDX_EM];
    data[1850] = 0.0 - k[521]*y[IDX_EM] - k[522]*y[IDX_EM] -
        k[523]*y[IDX_EM] - k[524]*y[IDX_EM] - k[525]*y[IDX_EM];
    data[1851] = 0.0 - k[526]*y[IDX_EM] - k[527]*y[IDX_EM];
    data[1852] = 0.0 - k[528]*y[IDX_EM] - k[529]*y[IDX_EM] -
        k[530]*y[IDX_EM] - k[531]*y[IDX_EM];
    data[1853] = 0.0 - k[532]*y[IDX_EM] - k[533]*y[IDX_EM] -
        k[534]*y[IDX_EM] - k[535]*y[IDX_EM];
    data[1854] = 0.0 - k[536]*y[IDX_EM] - k[537]*y[IDX_EM];
    data[1855] = 0.0 + k[2035];
    data[1856] = 0.0 - k[548]*y[IDX_EM];
    data[1857] = 0.0 - k[538]*y[IDX_EM];
    data[1858] = 0.0 - k[539]*y[IDX_EM] - k[540]*y[IDX_EM] -
        k[541]*y[IDX_EM];
    data[1859] = 0.0 + k[410] + k[2031];
    data[1860] = 0.0 - k[542]*y[IDX_EM];
    data[1861] = 0.0 - k[543]*y[IDX_EM] - k[544]*y[IDX_EM] -
        k[545]*y[IDX_EM];
    data[1862] = 0.0 + k[412] + k[2033];
    data[1863] = 0.0 - k[546]*y[IDX_EM] - k[547]*y[IDX_EM];
    data[1864] = 0.0 + k[363] + k[419];
    data[1865] = 0.0 - k[2139]*y[IDX_EM];
    data[1866] = 0.0 - k[563]*y[IDX_EM];
    data[1867] = 0.0 - k[549]*y[IDX_EM];
    data[1868] = 0.0 - k[550]*y[IDX_EM];
    data[1869] = 0.0 - k[551]*y[IDX_EM];
    data[1870] = 0.0 - k[552]*y[IDX_EM] - k[553]*y[IDX_EM];
    data[1871] = 0.0 - k[554]*y[IDX_EM];
    data[1872] = 0.0 + k[2041];
    data[1873] = 0.0 - k[555]*y[IDX_EM] - k[556]*y[IDX_EM];
    data[1874] = 0.0 - k[561]*y[IDX_EM] - k[562]*y[IDX_EM];
    data[1875] = 0.0 - k[557]*y[IDX_EM];
    data[1876] = 0.0 - k[558]*y[IDX_EM] - k[559]*y[IDX_EM] -
        k[560]*y[IDX_EM];
    data[1877] = 0.0 + k[420] + k[2044];
    data[1878] = 0.0 - k[2140]*y[IDX_EM];
    data[1879] = 0.0 + k[364] + k[422];
    data[1880] = 0.0 - k[2141]*y[IDX_EM];
    data[1881] = 0.0 - k[564]*y[IDX_EM];
    data[1882] = 0.0 - k[565]*y[IDX_EM] - k[566]*y[IDX_EM];
    data[1883] = 0.0 + k[2046];
    data[1884] = 0.0 + k[430] + k[2055];
    data[1885] = 0.0 - k[567]*y[IDX_EM];
    data[1886] = 0.0 + k[424] + k[2049];
    data[1887] = 0.0 - k[568]*y[IDX_EM] - k[569]*y[IDX_EM];
    data[1888] = 0.0 + k[427] + k[2052];
    data[1889] = 0.0 - k[570]*y[IDX_EM] - k[571]*y[IDX_EM];
    data[1890] = 0.0 - k[572]*y[IDX_EM] - k[573]*y[IDX_EM] -
        k[574]*y[IDX_EM];
    data[1891] = 0.0 + k[432] + k[2057];
    data[1892] = 0.0 - k[575]*y[IDX_EM];
    data[1893] = 0.0 - k[576]*y[IDX_EM];
    data[1894] = 0.0 + k[0]*y[IDX_CHI] + k[365] + k[438];
    data[1895] = 0.0 - k[2142]*y[IDX_EM];
    data[1896] = 0.0 + k[435] + k[2061];
    data[1897] = 0.0 - k[577]*y[IDX_EM];
    data[1898] = 0.0 - k[578]*y[IDX_EM];
    data[1899] = 0.0 + k[440] + k[2066];
    data[1900] = 0.0 - k[579]*y[IDX_EM] - k[580]*y[IDX_EM] -
        k[581]*y[IDX_EM];
    data[1901] = 0.0 + k[2070];
    data[1902] = 0.0 - k[582]*y[IDX_EM];
    data[1903] = 0.0 + k[444] + k[2073];
    data[1904] = 0.0 - k[2143]*y[IDX_EM];
    data[1905] = 0.0 + k[2071];
    data[1906] = 0.0 - k[583]*y[IDX_EM];
    data[1907] = 0.0 + k[448] + k[2077];
    data[1908] = 0.0 - k[2144]*y[IDX_EM];
    data[1909] = 0.0 - k[587]*y[IDX_EM];
    data[1910] = 0.0 - k[588]*y[IDX_EM] - k[589]*y[IDX_EM];
    data[1911] = 0.0 - k[590]*y[IDX_EM] - k[591]*y[IDX_EM];
    data[1912] = 0.0 - k[592]*y[IDX_EM];
    data[1913] = 0.0 + k[2083];
    data[1914] = 0.0 - k[593]*y[IDX_EM] - k[594]*y[IDX_EM] -
        k[595]*y[IDX_EM];
    data[1915] = 0.0 + k[2086];
    data[1916] = 0.0 - k[596]*y[IDX_EM] - k[597]*y[IDX_EM];
    data[1917] = 0.0 - k[598]*y[IDX_EM] - k[599]*y[IDX_EM];
    data[1918] = 0.0 - k[600]*y[IDX_EM] - k[601]*y[IDX_EM];
    data[1919] = 0.0 + k[2094];
    data[1920] = 0.0 - k[602]*y[IDX_EM];
    data[1921] = 0.0 - k[603]*y[IDX_EM] - k[604]*y[IDX_EM];
    data[1922] = 0.0 - k[605]*y[IDX_EM];
    data[1923] = 0.0 + k[447] + k[2076];
    data[1924] = 0.0 - k[584]*y[IDX_EM];
    data[1925] = 0.0 - k[585]*y[IDX_EM] - k[586]*y[IDX_EM];
    data[1926] = 0.0 + k[685]*y[IDX_C2HII] + k[686]*y[IDX_CHII] +
        k[687]*y[IDX_CH2II] + k[691]*y[IDX_H2SII] + k[697]*y[IDX_HSII] +
        k[703]*y[IDX_SiHII] + k[912]*y[IDX_H2II] + k[1582]*y[IDX_C2H3I] +
        k[1583]*y[IDX_C2H5I] + k[1585]*y[IDX_C3H2I] + k[1586]*y[IDX_CH2I] +
        k[1588]*y[IDX_CH3I] + k[1589]*y[IDX_CHI] + k[1595]*y[IDX_HSI] +
        k[1599]*y[IDX_NH2I] + k[1600]*y[IDX_NH2I] + k[1602]*y[IDX_NHI] +
        k[1611]*y[IDX_OHI] + k[1617]*y[IDX_SiHI] + k[1722]*y[IDX_H2I] -
        k[2123]*y[IDX_HI];
    data[1927] = 0.0 + k[607]*y[IDX_C2HI] + k[608]*y[IDX_CH2I] +
        k[610]*y[IDX_CH3I] + k[615]*y[IDX_CHI] + k[620]*y[IDX_H2OI] +
        k[621]*y[IDX_H2OI] + k[622]*y[IDX_H2SI] + k[625]*y[IDX_HC3NI] +
        k[627]*y[IDX_HNCI] + k[628]*y[IDX_HSI] + k[629]*y[IDX_NH2I] +
        k[631]*y[IDX_NHI] + k[637]*y[IDX_OHI] + k[644]*y[IDX_SiHI] +
        k[934]*y[IDX_H2I] - k[2122]*y[IDX_HI];
    data[1928] = 0.0 + k[110]*y[IDX_HII] + k[705]*y[IDX_CHII] +
        k[909]*y[IDX_H2II] + k[1346]*y[IDX_NHII] + k[1570]*y[IDX_C2H2I] +
        k[1571]*y[IDX_HCNI] - k[1736]*y[IDX_HI];
    data[1929] = 0.0 + k[837]*y[IDX_CHI] + k[935]*y[IDX_H2I] +
        k[1394]*y[IDX_NH2I] + k[1445]*y[IDX_NHI];
    data[1930] = 0.0 + k[112]*y[IDX_HII] + k[373] + k[607]*y[IDX_CII] +
        k[684]*y[IDX_SiII] + k[911]*y[IDX_H2II] + k[1186]*y[IDX_HeII] +
        k[1578]*y[IDX_HCNI] + k[1579]*y[IDX_HNCI] + k[1721]*y[IDX_H2I] +
        k[1795]*y[IDX_NI] + k[1970];
    data[1931] = 0.0 + k[459]*y[IDX_EM] + k[685]*y[IDX_CI] +
        k[936]*y[IDX_H2I] + k[1326]*y[IDX_NI] + k[1963];
    data[1932] = 0.0 + k[111]*y[IDX_HII] + k[368] + k[1179]*y[IDX_HeII] +
        k[1482]*y[IDX_O2II] + k[1570]*y[IDX_C2I] + k[1574]*y[IDX_NOI] +
        k[1674]*y[IDX_CHI] + k[1701]*y[IDX_CNI] - k[1737]*y[IDX_HI] +
        k[1928]*y[IDX_OHI] + k[1965];
    data[1933] = 0.0 + k[461]*y[IDX_EM] + k[461]*y[IDX_EM] +
        k[462]*y[IDX_EM] + k[806]*y[IDX_CH4I] + k[1329]*y[IDX_NI];
    data[1934] = 0.0 + k[369] + k[1182]*y[IDX_HeII] + k[1582]*y[IDX_CI] -
        k[1738]*y[IDX_HI] + k[1869]*y[IDX_OI] + k[1966];
    data[1935] = 0.0 + k[883]*y[IDX_HII] + k[1183]*y[IDX_HeII] +
        k[1675]*y[IDX_CHI];
    data[1936] = 0.0 + k[1583]*y[IDX_CI];
    data[1937] = 0.0 + k[464]*y[IDX_EM] + k[466]*y[IDX_EM] +
        k[467]*y[IDX_EM] + k[2151];
    data[1938] = 0.0 + k[113]*y[IDX_HII];
    data[1939] = 0.0 - k[1097]*y[IDX_HI];
    data[1940] = 0.0 + k[472]*y[IDX_EM];
    data[1941] = 0.0 + k[1119]*y[IDX_HCNI];
    data[1942] = 0.0 + k[1585]*y[IDX_CI] + k[1797]*y[IDX_NI];
    data[1943] = 0.0 + k[474]*y[IDX_EM] + k[2148];
    data[1944] = 0.0 + k[1799]*y[IDX_NI];
    data[1945] = 0.0 + k[2]*y[IDX_H2I] - k[9]*y[IDX_HI] + k[9]*y[IDX_HI] +
        k[9]*y[IDX_HI] + k[117]*y[IDX_HII] + k[391] + k[615]*y[IDX_CII] +
        k[837]*y[IDX_C2II] + k[852]*y[IDX_NII] + k[857]*y[IDX_OII] +
        k[861]*y[IDX_SII] + k[862]*y[IDX_SiII] + k[916]*y[IDX_H2II] +
        k[1206]*y[IDX_HeII] + k[1589]*y[IDX_CI] + k[1674]*y[IDX_C2H2I] +
        k[1675]*y[IDX_C2H4I] + k[1676]*y[IDX_CH4I] + k[1682]*y[IDX_NI] +
        k[1686]*y[IDX_NOI] + k[1687]*y[IDX_O2I] + k[1688]*y[IDX_O2I] +
        k[1693]*y[IDX_OI] + k[1695]*y[IDX_OCSI] + k[1696]*y[IDX_OHI] +
        k[1697]*y[IDX_SI] + k[1700]*y[IDX_SOI] + k[1725]*y[IDX_H2I] -
        k[1743]*y[IDX_HI] + k[1999];
    data[1946] = 0.0 + k[380] + k[477]*y[IDX_EM] + k[686]*y[IDX_CI] +
        k[705]*y[IDX_C2I] + k[711]*y[IDX_CH4I] + k[713]*y[IDX_CNI] +
        k[718]*y[IDX_H2OI] + k[724]*y[IDX_HCNI] + k[728]*y[IDX_NI] +
        k[735]*y[IDX_OI] + k[739]*y[IDX_SI] + k[937]*y[IDX_H2I] -
        k[1098]*y[IDX_HI];
    data[1947] = 0.0 + k[114]*y[IDX_HII] + k[382] + k[608]*y[IDX_CII] +
        k[772]*y[IDX_SII] + k[913]*y[IDX_H2II] + k[1193]*y[IDX_HeII] +
        k[1586]*y[IDX_CI] + k[1619]*y[IDX_CH2I] + k[1619]*y[IDX_CH2I] +
        k[1619]*y[IDX_CH2I] + k[1619]*y[IDX_CH2I] + k[1620]*y[IDX_CH2I] +
        k[1620]*y[IDX_CH2I] + k[1631]*y[IDX_NOI] + k[1633]*y[IDX_O2I] +
        k[1633]*y[IDX_O2I] + k[1638]*y[IDX_OI] + k[1638]*y[IDX_OI] +
        k[1639]*y[IDX_OI] + k[1641]*y[IDX_OHI] + k[1645]*y[IDX_SI] +
        k[1723]*y[IDX_H2I] - k[1739]*y[IDX_HI] + k[1801]*y[IDX_NI] +
        k[1802]*y[IDX_NI] + k[1982];
    data[1948] = 0.0 + k[479]*y[IDX_EM] + k[479]*y[IDX_EM] +
        k[480]*y[IDX_EM] + k[687]*y[IDX_CI] + k[743]*y[IDX_H2OI] +
        k[744]*y[IDX_H2SI] + k[746]*y[IDX_H2SI] + k[750]*y[IDX_OI] +
        k[753]*y[IDX_SI] + k[938]*y[IDX_H2I] - k[1099]*y[IDX_HI] +
        k[1331]*y[IDX_NI] + k[1979];
    data[1949] = 0.0 - k[1740]*y[IDX_HI];
    data[1950] = 0.0 + k[115]*y[IDX_HII] + k[384] + k[610]*y[IDX_CII] +
        k[790]*y[IDX_SII] + k[1588]*y[IDX_CI] + k[1648]*y[IDX_CH3I] +
        k[1648]*y[IDX_CH3I] + k[1664]*y[IDX_OI] + k[1665]*y[IDX_OI] +
        k[1669]*y[IDX_SI] + k[1724]*y[IDX_H2I] - k[1741]*y[IDX_HI] +
        k[1804]*y[IDX_NI] + k[1806]*y[IDX_NI] + k[1806]*y[IDX_NI] + k[1986];
    data[1951] = 0.0 + k[481]*y[IDX_EM] + k[483]*y[IDX_EM] +
        k[483]*y[IDX_EM] + k[783]*y[IDX_OI] - k[1100]*y[IDX_HI] + k[1985];
    data[1952] = 0.0 + k[484]*y[IDX_EM];
    data[1953] = 0.0 + k[1289]*y[IDX_NII] + k[1291]*y[IDX_NII] +
        k[1292]*y[IDX_NII] + k[1483]*y[IDX_O2II] + k[1991];
    data[1954] = 0.0 + k[486]*y[IDX_EM] + k[488]*y[IDX_EM] +
        k[489]*y[IDX_EM] + k[490]*y[IDX_EM] + k[2152];
    data[1955] = 0.0 + k[116]*y[IDX_HII] + k[711]*y[IDX_CHII] +
        k[806]*y[IDX_C2H2II] + k[816]*y[IDX_N2II] + k[821]*y[IDX_SII] +
        k[822]*y[IDX_SII] + k[914]*y[IDX_H2II] + k[915]*y[IDX_H2II] +
        k[1202]*y[IDX_HeII] + k[1204]*y[IDX_HeII] + k[1293]*y[IDX_NII] +
        k[1294]*y[IDX_NII] + k[1295]*y[IDX_NII] + k[1295]*y[IDX_NII] +
        k[1676]*y[IDX_CHI] - k[1742]*y[IDX_HI] + k[1996] + k[1998];
    data[1956] = 0.0 + k[491]*y[IDX_EM] + k[491]*y[IDX_EM] +
        k[492]*y[IDX_EM] + k[939]*y[IDX_H2I] - k[1101]*y[IDX_HI] + k[1994];
    data[1957] = 0.0 + k[493]*y[IDX_EM] + k[495]*y[IDX_EM] +
        k[495]*y[IDX_EM] + k[496]*y[IDX_EM] + k[834]*y[IDX_MgI] -
        k[1102]*y[IDX_HI] + k[2248];
    data[1958] = 0.0 + k[109]*y[IDX_HII] + k[1720]*y[IDX_H2I];
    data[1959] = 0.0 - k[108]*y[IDX_HI] + k[944]*y[IDX_H2I];
    data[1960] = 0.0 + k[713]*y[IDX_CHII] + k[917]*y[IDX_H2II] +
        k[1701]*y[IDX_C2H2I] + k[1705]*y[IDX_HCNI] + k[1707]*y[IDX_HNCI] +
        k[1726]*y[IDX_H2I] + k[1933]*y[IDX_OHI];
    data[1961] = 0.0 - k[191]*y[IDX_HI] + k[866]*y[IDX_HCNI] +
        k[940]*y[IDX_H2I];
    data[1962] = 0.0 + k[919]*y[IDX_H2II] - k[1745]*y[IDX_HI] +
        k[1934]*y[IDX_OHI];
    data[1963] = 0.0 - k[192]*y[IDX_HI] + k[941]*y[IDX_H2I] +
        k[942]*y[IDX_H2I];
    data[1964] = 0.0 + k[918]*y[IDX_H2II] - k[1744]*y[IDX_HI];
    data[1965] = 0.0 + k[118]*y[IDX_HII] + k[1936]*y[IDX_OHI];
    data[1966] = 0.0 + k[943]*y[IDX_H2I];
    data[1967] = 0.0 + k[8]*y[IDX_H2I] + k[8]*y[IDX_H2I] +
        k[459]*y[IDX_C2HII] + k[461]*y[IDX_C2H2II] + k[461]*y[IDX_C2H2II] +
        k[462]*y[IDX_C2H2II] + k[464]*y[IDX_C2H5OH2II] + k[466]*y[IDX_C2H5OH2II]
        + k[467]*y[IDX_C2H5OH2II] + k[472]*y[IDX_C2NHII] + k[474]*y[IDX_C3H5II]
        + k[477]*y[IDX_CHII] + k[479]*y[IDX_CH2II] + k[479]*y[IDX_CH2II] +
        k[480]*y[IDX_CH2II] + k[481]*y[IDX_CH3II] + k[483]*y[IDX_CH3II] +
        k[483]*y[IDX_CH3II] + k[484]*y[IDX_CH3CNHII] + k[486]*y[IDX_CH3OH2II] +
        k[488]*y[IDX_CH3OH2II] + k[489]*y[IDX_CH3OH2II] + k[490]*y[IDX_CH3OH2II]
        + k[491]*y[IDX_CH4II] + k[491]*y[IDX_CH4II] + k[492]*y[IDX_CH4II] +
        k[493]*y[IDX_CH5II] + k[495]*y[IDX_CH5II] + k[495]*y[IDX_CH5II] +
        k[496]*y[IDX_CH5II] + k[501]*y[IDX_H2II] + k[501]*y[IDX_H2II] +
        k[504]*y[IDX_H2COII] + k[504]*y[IDX_H2COII] + k[505]*y[IDX_H2COII] +
        k[506]*y[IDX_H2CSII] + k[506]*y[IDX_H2CSII] + k[507]*y[IDX_H2CSII] +
        k[508]*y[IDX_H2ClII] + k[508]*y[IDX_H2ClII] + k[509]*y[IDX_H2ClII] +
        k[510]*y[IDX_H2NOII] + k[513]*y[IDX_H2OII] + k[513]*y[IDX_H2OII] +
        k[514]*y[IDX_H2OII] + k[515]*y[IDX_H2SII] + k[516]*y[IDX_H2SII] +
        k[516]*y[IDX_H2SII] + k[517]*y[IDX_H2S2II] + k[519]*y[IDX_H3II] +
        k[520]*y[IDX_H3II] + k[520]*y[IDX_H3II] + k[520]*y[IDX_H3II] +
        k[523]*y[IDX_H3COII] + k[524]*y[IDX_H3COII] + k[525]*y[IDX_H3COII] +
        k[525]*y[IDX_H3COII] + k[526]*y[IDX_H3CSII] + k[527]*y[IDX_H3CSII] +
        k[528]*y[IDX_H3OII] + k[529]*y[IDX_H3OII] + k[531]*y[IDX_H3OII] +
        k[531]*y[IDX_H3OII] + k[532]*y[IDX_H3SII] + k[534]*y[IDX_H3SII] +
        k[534]*y[IDX_H3SII] + k[535]*y[IDX_H3SII] + k[537]*y[IDX_H5C2O2II] +
        k[538]*y[IDX_HCNII] + k[539]*y[IDX_HCNHII] + k[539]*y[IDX_HCNHII] +
        k[540]*y[IDX_HCNHII] + k[541]*y[IDX_HCNHII] + k[542]*y[IDX_HCOII] +
        k[543]*y[IDX_HCO2II] + k[544]*y[IDX_HCO2II] + k[547]*y[IDX_HCSII] +
        k[548]*y[IDX_HClII] + k[549]*y[IDX_HNOII] + k[550]*y[IDX_HNSII] +
        k[551]*y[IDX_HOCII] + k[553]*y[IDX_HOCSII] + k[554]*y[IDX_HSII] +
        k[556]*y[IDX_HS2II] + k[557]*y[IDX_HSOII] + k[558]*y[IDX_HSO2II] +
        k[559]*y[IDX_HSO2II] + k[562]*y[IDX_HSiSII] + k[563]*y[IDX_HeHII] +
        k[565]*y[IDX_N2HII] + k[567]*y[IDX_NHII] + k[568]*y[IDX_NH2II] +
        k[568]*y[IDX_NH2II] + k[569]*y[IDX_NH2II] + k[570]*y[IDX_NH3II] +
        k[571]*y[IDX_NH3II] + k[571]*y[IDX_NH3II] + k[573]*y[IDX_NH4II] +
        k[573]*y[IDX_NH4II] + k[574]*y[IDX_NH4II] + k[578]*y[IDX_O2HII] +
        k[582]*y[IDX_OHII] + k[592]*y[IDX_SiHII] + k[594]*y[IDX_SiH2II] +
        k[594]*y[IDX_SiH2II] + k[595]*y[IDX_SiH2II] + k[596]*y[IDX_SiH3II] +
        k[599]*y[IDX_SiH4II] + k[601]*y[IDX_SiH5II] + k[604]*y[IDX_SiOHII] +
        k[2135]*y[IDX_HII];
    data[1968] = 0.0 - k[9]*y[IDX_CHI] + k[9]*y[IDX_CHI] + k[9]*y[IDX_CHI] -
        k[10]*y[IDX_H2I] + k[10]*y[IDX_H2I] + k[10]*y[IDX_H2I] +
        k[10]*y[IDX_H2I] - k[11]*y[IDX_H2OI] + k[11]*y[IDX_H2OI] +
        k[11]*y[IDX_H2OI] - k[12]*y[IDX_O2I] + k[12]*y[IDX_O2I] -
        k[13]*y[IDX_OHI] + k[13]*y[IDX_OHI] + k[13]*y[IDX_OHI] -
        k[108]*y[IDX_ClII] - k[191]*y[IDX_CNII] - k[192]*y[IDX_COII] -
        k[193]*y[IDX_H2II] - k[194]*y[IDX_HCNII] - k[195]*y[IDX_HeII] -
        k[196]*y[IDX_OII] - k[362] - k[406] - k[1097]*y[IDX_C2N2II] -
        k[1098]*y[IDX_CHII] - k[1099]*y[IDX_CH2II] - k[1100]*y[IDX_CH3II] -
        k[1101]*y[IDX_CH4II] - k[1102]*y[IDX_CH5II] - k[1103]*y[IDX_H2SII] -
        k[1104]*y[IDX_H3SII] - k[1105]*y[IDX_HSII] - k[1106]*y[IDX_HeHII] -
        k[1107]*y[IDX_SO2II] - k[1108]*y[IDX_SiHII] - k[1109]*y[IDX_SiSII] -
        k[1736]*y[IDX_C2I] - k[1737]*y[IDX_C2H2I] - k[1738]*y[IDX_C2H3I] -
        k[1739]*y[IDX_CH2I] - k[1740]*y[IDX_CH2COI] - k[1741]*y[IDX_CH3I] -
        k[1742]*y[IDX_CH4I] - k[1743]*y[IDX_CHI] - k[1744]*y[IDX_CO2I] -
        k[1745]*y[IDX_COI] - k[1746]*y[IDX_H2CNI] - k[1747]*y[IDX_H2COI] -
        k[1748]*y[IDX_H2OI] - k[1749]*y[IDX_H2SI] - k[1750]*y[IDX_HCNI] -
        k[1751]*y[IDX_HCOI] - k[1752]*y[IDX_HCOI] - k[1753]*y[IDX_HCSI] -
        k[1754]*y[IDX_HNCI] + k[1754]*y[IDX_HNCI] - k[1755]*y[IDX_HNOI] -
        k[1756]*y[IDX_HNOI] - k[1757]*y[IDX_HNOI] - k[1758]*y[IDX_HSI] -
        k[1759]*y[IDX_NCCNI] - k[1760]*y[IDX_NH2I] - k[1761]*y[IDX_NH3I] -
        k[1762]*y[IDX_NHI] - k[1763]*y[IDX_NO2I] - k[1764]*y[IDX_NOI] -
        k[1765]*y[IDX_NOI] - k[1766]*y[IDX_NSI] - k[1767]*y[IDX_NSI] -
        k[1768]*y[IDX_O2I] - k[1769]*y[IDX_O2HI] - k[1770]*y[IDX_O2HI] -
        k[1771]*y[IDX_O2HI] - k[1772]*y[IDX_OCNI] - k[1773]*y[IDX_OCNI] -
        k[1774]*y[IDX_OCNI] - k[1775]*y[IDX_OCSI] - k[1776]*y[IDX_OHI] -
        k[1777]*y[IDX_S2I] - k[1778]*y[IDX_SOI] - k[1779]*y[IDX_SOI] -
        k[1787]*y[IDX_HClI] - k[2110]*y[IDX_HII] - k[2122]*y[IDX_CII] -
        k[2123]*y[IDX_CI] - k[2124]*y[IDX_OI] - k[2125]*y[IDX_OHI] -
        k[2126]*y[IDX_SiII];
    data[1969] = 0.0 + k[109]*y[IDX_ClI] + k[110]*y[IDX_C2I] +
        k[111]*y[IDX_C2H2I] + k[112]*y[IDX_C2HI] + k[113]*y[IDX_C2NI] +
        k[114]*y[IDX_CH2I] + k[115]*y[IDX_CH3I] + k[116]*y[IDX_CH4I] +
        k[117]*y[IDX_CHI] + k[118]*y[IDX_CSI] + k[119]*y[IDX_H2COI] +
        k[120]*y[IDX_H2CSI] + k[121]*y[IDX_H2OI] + k[122]*y[IDX_H2S2I] +
        k[123]*y[IDX_H2SI] + k[124]*y[IDX_HCNI] + k[125]*y[IDX_HCOI] +
        k[126]*y[IDX_HClI] + k[127]*y[IDX_HS2I] + k[128]*y[IDX_HSI] +
        k[129]*y[IDX_MgI] + k[130]*y[IDX_NH2I] + k[131]*y[IDX_NH3I] +
        k[132]*y[IDX_NHI] + k[133]*y[IDX_NOI] + k[134]*y[IDX_NSI] +
        k[135]*y[IDX_O2I] + k[136]*y[IDX_OI] + k[137]*y[IDX_OCSI] +
        k[138]*y[IDX_OHI] + k[139]*y[IDX_S2I] + k[140]*y[IDX_SI] +
        k[141]*y[IDX_SO2I] + k[142]*y[IDX_SOI] + k[143]*y[IDX_SiI] +
        k[144]*y[IDX_SiC2I] + k[145]*y[IDX_SiC3I] + k[146]*y[IDX_SiCI] +
        k[147]*y[IDX_SiH2I] + k[148]*y[IDX_SiH3I] + k[149]*y[IDX_SiH4I] +
        k[150]*y[IDX_SiHI] + k[151]*y[IDX_SiOI] + k[152]*y[IDX_SiSI] +
        k[883]*y[IDX_C2H4I] + k[892]*y[IDX_H2COI] + k[895]*y[IDX_H2SI] -
        k[2110]*y[IDX_HI] + k[2135]*y[IDX_EM];
    data[1970] = 0.0 + k[2]*y[IDX_CHI] + k[3]*y[IDX_H2I] + k[3]*y[IDX_H2I] +
        k[3]*y[IDX_H2I] + k[3]*y[IDX_H2I] + k[4]*y[IDX_H2OI] + k[7]*y[IDX_OHI] +
        k[8]*y[IDX_EM] + k[8]*y[IDX_EM] - k[10]*y[IDX_HI] + k[10]*y[IDX_HI] +
        k[10]*y[IDX_HI] + k[10]*y[IDX_HI] + k[359] + k[361] + k[361] +
        k[920]*y[IDX_H2II] + k[934]*y[IDX_CII] + k[935]*y[IDX_C2II] +
        k[936]*y[IDX_C2HII] + k[937]*y[IDX_CHII] + k[938]*y[IDX_CH2II] +
        k[939]*y[IDX_CH4II] + k[940]*y[IDX_CNII] + k[941]*y[IDX_COII] +
        k[942]*y[IDX_COII] + k[943]*y[IDX_CSII] + k[944]*y[IDX_ClII] +
        k[945]*y[IDX_H2OII] + k[946]*y[IDX_H2SII] + k[947]*y[IDX_HCNII] +
        k[948]*y[IDX_HClII] + k[949]*y[IDX_HSII] + k[950]*y[IDX_HeII] +
        k[952]*y[IDX_NII] + k[953]*y[IDX_N2II] + k[955]*y[IDX_NHII] +
        k[956]*y[IDX_NH2II] + k[957]*y[IDX_NH3II] + k[958]*y[IDX_OII] +
        k[960]*y[IDX_OHII] + k[961]*y[IDX_SII] + k[962]*y[IDX_SO2II] +
        k[963]*y[IDX_SiH4II] + k[964]*y[IDX_SiOII] + k[1720]*y[IDX_ClI] +
        k[1721]*y[IDX_C2HI] + k[1722]*y[IDX_CI] + k[1723]*y[IDX_CH2I] +
        k[1724]*y[IDX_CH3I] + k[1725]*y[IDX_CHI] + k[1726]*y[IDX_CNI] +
        k[1727]*y[IDX_HSI] + k[1728]*y[IDX_NI] + k[1729]*y[IDX_NH2I] +
        k[1730]*y[IDX_NHI] + k[1731]*y[IDX_O2I] + k[1733]*y[IDX_OI] +
        k[1734]*y[IDX_OHI] + k[1735]*y[IDX_SI];
    data[1971] = 0.0 - k[193]*y[IDX_HI] + k[501]*y[IDX_EM] +
        k[501]*y[IDX_EM] + k[909]*y[IDX_C2I] + k[911]*y[IDX_C2HI] +
        k[912]*y[IDX_CI] + k[913]*y[IDX_CH2I] + k[914]*y[IDX_CH4I] +
        k[915]*y[IDX_CH4I] + k[916]*y[IDX_CHI] + k[917]*y[IDX_CNI] +
        k[918]*y[IDX_CO2I] + k[919]*y[IDX_COI] + k[920]*y[IDX_H2I] +
        k[921]*y[IDX_H2COI] + k[922]*y[IDX_H2OI] + k[923]*y[IDX_H2SI] +
        k[926]*y[IDX_HeI] + k[927]*y[IDX_N2I] + k[928]*y[IDX_NI] +
        k[929]*y[IDX_NHI] + k[930]*y[IDX_NOI] + k[931]*y[IDX_O2I] +
        k[932]*y[IDX_OI] + k[933]*y[IDX_OHI] + k[2009];
    data[1972] = 0.0 + k[508]*y[IDX_EM] + k[508]*y[IDX_EM] +
        k[509]*y[IDX_EM] + k[2198];
    data[1973] = 0.0 + k[398] - k[1746]*y[IDX_HI] + k[2010];
    data[1974] = 0.0 + k[119]*y[IDX_HII] + k[892]*y[IDX_HII] +
        k[921]*y[IDX_H2II] + k[972]*y[IDX_O2II] + k[1217]*y[IDX_HeII] +
        k[1314]*y[IDX_N2II] - k[1747]*y[IDX_HI] + k[2012] + k[2012] + k[2014];
    data[1975] = 0.0 + k[504]*y[IDX_EM] + k[504]*y[IDX_EM] +
        k[505]*y[IDX_EM];
    data[1976] = 0.0 + k[120]*y[IDX_HII];
    data[1977] = 0.0 + k[506]*y[IDX_EM] + k[506]*y[IDX_EM] +
        k[507]*y[IDX_EM];
    data[1978] = 0.0 + k[510]*y[IDX_EM] + k[2273];
    data[1979] = 0.0 + k[4]*y[IDX_H2I] - k[11]*y[IDX_HI] + k[11]*y[IDX_HI] +
        k[11]*y[IDX_HI] + k[121]*y[IDX_HII] + k[401] + k[620]*y[IDX_CII] +
        k[621]*y[IDX_CII] + k[718]*y[IDX_CHII] + k[743]*y[IDX_CH2II] +
        k[922]*y[IDX_H2II] + k[1013]*y[IDX_SiII] + k[1222]*y[IDX_HeII] -
        k[1748]*y[IDX_HI] + k[2018];
    data[1980] = 0.0 + k[513]*y[IDX_EM] + k[513]*y[IDX_EM] +
        k[514]*y[IDX_EM] + k[945]*y[IDX_H2I] + k[988]*y[IDX_SI] +
        k[1333]*y[IDX_NI] + k[2016];
    data[1981] = 0.0 + k[123]*y[IDX_HII] + k[622]*y[IDX_CII] +
        k[744]*y[IDX_CH2II] + k[746]*y[IDX_CH2II] + k[895]*y[IDX_HII] +
        k[923]*y[IDX_H2II] + k[1226]*y[IDX_HeII] + k[1302]*y[IDX_NII] +
        k[1315]*y[IDX_N2II] + k[1550]*y[IDX_SII] - k[1749]*y[IDX_HI] + k[2021];
    data[1982] = 0.0 + k[515]*y[IDX_EM] + k[516]*y[IDX_EM] +
        k[516]*y[IDX_EM] + k[691]*y[IDX_CI] + k[946]*y[IDX_H2I] -
        k[1103]*y[IDX_HI];
    data[1983] = 0.0 + k[122]*y[IDX_HII] + k[1225]*y[IDX_HeII];
    data[1984] = 0.0 + k[517]*y[IDX_EM];
    data[1985] = 0.0 + k[1228]*y[IDX_HeII] + k[2024] + k[2024];
    data[1986] = 0.0 + k[519]*y[IDX_EM] + k[520]*y[IDX_EM] +
        k[520]*y[IDX_EM] + k[520]*y[IDX_EM] + k[1054]*y[IDX_MgI] +
        k[1063]*y[IDX_OI] + k[2025];
    data[1987] = 0.0 + k[523]*y[IDX_EM] + k[524]*y[IDX_EM] +
        k[525]*y[IDX_EM] + k[525]*y[IDX_EM];
    data[1988] = 0.0 + k[526]*y[IDX_EM] + k[527]*y[IDX_EM] + k[2282];
    data[1989] = 0.0 + k[528]*y[IDX_EM] + k[529]*y[IDX_EM] +
        k[531]*y[IDX_EM] + k[531]*y[IDX_EM] + k[2246];
    data[1990] = 0.0 + k[532]*y[IDX_EM] + k[534]*y[IDX_EM] +
        k[534]*y[IDX_EM] + k[535]*y[IDX_EM] - k[1104]*y[IDX_HI] +
        k[1553]*y[IDX_SI] + k[2278];
    data[1991] = 0.0 + k[537]*y[IDX_EM] + k[2167];
    data[1992] = 0.0 + k[625]*y[IDX_CII];
    data[1993] = 0.0 + k[126]*y[IDX_HII] + k[413] + k[1241]*y[IDX_HeII] -
        k[1787]*y[IDX_HI] + k[2034];
    data[1994] = 0.0 + k[548]*y[IDX_EM] + k[948]*y[IDX_H2I];
    data[1995] = 0.0 + k[124]*y[IDX_HII] + k[408] + k[724]*y[IDX_CHII] +
        k[866]*y[IDX_CNII] + k[1119]*y[IDX_C3II] + k[1231]*y[IDX_HeII] +
        k[1233]*y[IDX_HeII] + k[1571]*y[IDX_C2I] + k[1578]*y[IDX_C2HI] +
        k[1705]*y[IDX_CNI] - k[1750]*y[IDX_HI] + k[1891]*y[IDX_OI] + k[2028];
    data[1996] = 0.0 - k[194]*y[IDX_HI] + k[538]*y[IDX_EM] +
        k[947]*y[IDX_H2I];
    data[1997] = 0.0 + k[539]*y[IDX_EM] + k[539]*y[IDX_EM] +
        k[540]*y[IDX_EM] + k[541]*y[IDX_EM] + k[2289];
    data[1998] = 0.0 + k[125]*y[IDX_HII] + k[409] + k[1235]*y[IDX_HeII] -
        k[1751]*y[IDX_HI] - k[1752]*y[IDX_HI] + k[1813]*y[IDX_NI] +
        k[1892]*y[IDX_OI] + k[1951]*y[IDX_SI] + k[2030];
    data[1999] = 0.0 + k[542]*y[IDX_EM] + k[1543]*y[IDX_OHI] + k[2029];
    data[2000] = 0.0 + k[543]*y[IDX_EM] + k[544]*y[IDX_EM] + k[2247];
    data[2001] = 0.0 + k[1239]*y[IDX_HeII] - k[1753]*y[IDX_HI] +
        k[1895]*y[IDX_OI];
    data[2002] = 0.0 + k[547]*y[IDX_EM] + k[1501]*y[IDX_OI];
    data[2003] = 0.0 + k[926]*y[IDX_H2II];
    data[2004] = 0.0 - k[195]*y[IDX_HI] + k[950]*y[IDX_H2I] +
        k[1179]*y[IDX_C2H2I] + k[1182]*y[IDX_C2H3I] + k[1183]*y[IDX_C2H4I] +
        k[1186]*y[IDX_C2HI] + k[1193]*y[IDX_CH2I] + k[1202]*y[IDX_CH4I] +
        k[1204]*y[IDX_CH4I] + k[1206]*y[IDX_CHI] + k[1217]*y[IDX_H2COI] +
        k[1222]*y[IDX_H2OI] + k[1225]*y[IDX_H2S2I] + k[1226]*y[IDX_H2SI] +
        k[1228]*y[IDX_H2SiOI] + k[1231]*y[IDX_HCNI] + k[1233]*y[IDX_HCNI] +
        k[1235]*y[IDX_HCOI] + k[1239]*y[IDX_HCSI] + k[1241]*y[IDX_HClI] +
        k[1242]*y[IDX_HNCI] + k[1243]*y[IDX_HNCI] + k[1245]*y[IDX_HNOI] +
        k[1248]*y[IDX_HS2I] + k[1249]*y[IDX_HSI] + k[1253]*y[IDX_NH2I] +
        k[1255]*y[IDX_NH3I] + k[1256]*y[IDX_NHI] + k[1268]*y[IDX_OHI] +
        k[1279]*y[IDX_SiH2I] + k[1281]*y[IDX_SiH3I] + k[1283]*y[IDX_SiH4I] +
        k[1284]*y[IDX_SiHI];
    data[2005] = 0.0 + k[563]*y[IDX_EM] - k[1106]*y[IDX_HI];
    data[2006] = 0.0 + k[414] + k[627]*y[IDX_CII] + k[1242]*y[IDX_HeII] +
        k[1243]*y[IDX_HeII] + k[1579]*y[IDX_C2HI] + k[1707]*y[IDX_CNI] -
        k[1754]*y[IDX_HI] + k[1754]*y[IDX_HI] + k[2036];
    data[2007] = 0.0 + k[416] + k[1245]*y[IDX_HeII] - k[1755]*y[IDX_HI] -
        k[1756]*y[IDX_HI] - k[1757]*y[IDX_HI] + k[1896]*y[IDX_OI] + k[2038];
    data[2008] = 0.0 + k[549]*y[IDX_EM];
    data[2009] = 0.0 + k[550]*y[IDX_EM] + k[2291];
    data[2010] = 0.0 + k[551]*y[IDX_EM];
    data[2011] = 0.0 + k[553]*y[IDX_EM] + k[2285];
    data[2012] = 0.0 + k[128]*y[IDX_HII] + k[418] + k[628]*y[IDX_CII] +
        k[1249]*y[IDX_HeII] + k[1595]*y[IDX_CI] + k[1727]*y[IDX_H2I] -
        k[1758]*y[IDX_HI] + k[1816]*y[IDX_NI] + k[1900]*y[IDX_OI] +
        k[1953]*y[IDX_SI] + k[2043];
    data[2013] = 0.0 + k[554]*y[IDX_EM] + k[697]*y[IDX_CI] +
        k[949]*y[IDX_H2I] - k[1105]*y[IDX_HI] + k[1336]*y[IDX_NI] +
        k[1504]*y[IDX_OI] + k[2039];
    data[2014] = 0.0 + k[127]*y[IDX_HII] + k[1248]*y[IDX_HeII];
    data[2015] = 0.0 + k[556]*y[IDX_EM];
    data[2016] = 0.0 + k[562]*y[IDX_EM] + k[2191];
    data[2017] = 0.0 + k[557]*y[IDX_EM] + k[2283];
    data[2018] = 0.0 + k[558]*y[IDX_EM] + k[559]*y[IDX_EM] + k[2295];
    data[2019] = 0.0 + k[129]*y[IDX_HII] + k[834]*y[IDX_CH5II] +
        k[1054]*y[IDX_H3II];
    data[2020] = 0.0 + k[728]*y[IDX_CHII] + k[928]*y[IDX_H2II] +
        k[1326]*y[IDX_C2HII] + k[1329]*y[IDX_C2H2II] + k[1331]*y[IDX_CH2II] +
        k[1333]*y[IDX_H2OII] + k[1336]*y[IDX_HSII] + k[1337]*y[IDX_NHII] +
        k[1338]*y[IDX_NH2II] + k[1340]*y[IDX_OHII] + k[1682]*y[IDX_CHI] +
        k[1728]*y[IDX_H2I] + k[1795]*y[IDX_C2HI] + k[1797]*y[IDX_C3H2I] +
        k[1799]*y[IDX_C4HI] + k[1801]*y[IDX_CH2I] + k[1802]*y[IDX_CH2I] +
        k[1804]*y[IDX_CH3I] + k[1806]*y[IDX_CH3I] + k[1806]*y[IDX_CH3I] +
        k[1813]*y[IDX_HCOI] + k[1816]*y[IDX_HSI] + k[1819]*y[IDX_NHI] +
        k[1827]*y[IDX_OHI];
    data[2021] = 0.0 + k[852]*y[IDX_CHI] + k[952]*y[IDX_H2I] +
        k[1289]*y[IDX_CH3OHI] + k[1291]*y[IDX_CH3OHI] + k[1292]*y[IDX_CH3OHI] +
        k[1293]*y[IDX_CH4I] + k[1294]*y[IDX_CH4I] + k[1295]*y[IDX_CH4I] +
        k[1295]*y[IDX_CH4I] + k[1302]*y[IDX_H2SI] + k[1308]*y[IDX_NHI];
    data[2022] = 0.0 + k[927]*y[IDX_H2II];
    data[2023] = 0.0 + k[816]*y[IDX_CH4I] + k[953]*y[IDX_H2I] +
        k[1314]*y[IDX_H2COI] + k[1315]*y[IDX_H2SI];
    data[2024] = 0.0 + k[565]*y[IDX_EM] + k[2290];
    data[2025] = 0.0 - k[1759]*y[IDX_HI];
    data[2026] = 0.0 + k[132]*y[IDX_HII] + k[429] + k[631]*y[IDX_CII] +
        k[929]*y[IDX_H2II] + k[1256]*y[IDX_HeII] + k[1308]*y[IDX_NII] +
        k[1445]*y[IDX_C2II] + k[1457]*y[IDX_OII] + k[1461]*y[IDX_SII] +
        k[1602]*y[IDX_CI] + k[1730]*y[IDX_H2I] - k[1762]*y[IDX_HI] +
        k[1819]*y[IDX_NI] + k[1844]*y[IDX_NHI] + k[1844]*y[IDX_NHI] +
        k[1844]*y[IDX_NHI] + k[1844]*y[IDX_NHI] + k[1847]*y[IDX_NOI] +
        k[1851]*y[IDX_OI] + k[1854]*y[IDX_OHI] + k[1857]*y[IDX_SI] + k[2054];
    data[2027] = 0.0 + k[567]*y[IDX_EM] + k[955]*y[IDX_H2I] +
        k[1337]*y[IDX_NI] + k[1346]*y[IDX_C2I] + k[1373]*y[IDX_SI];
    data[2028] = 0.0 + k[130]*y[IDX_HII] + k[425] + k[629]*y[IDX_CII] +
        k[1253]*y[IDX_HeII] + k[1394]*y[IDX_C2II] + k[1599]*y[IDX_CI] +
        k[1600]*y[IDX_CI] + k[1729]*y[IDX_H2I] - k[1760]*y[IDX_HI] +
        k[1835]*y[IDX_NOI] + k[1902]*y[IDX_OI] + k[2050];
    data[2029] = 0.0 + k[568]*y[IDX_EM] + k[568]*y[IDX_EM] +
        k[569]*y[IDX_EM] + k[956]*y[IDX_H2I] + k[1338]*y[IDX_NI] +
        k[1392]*y[IDX_SI] + k[1507]*y[IDX_OI];
    data[2030] = 0.0 + k[131]*y[IDX_HII] + k[426] + k[1255]*y[IDX_HeII] -
        k[1761]*y[IDX_HI] + k[2051];
    data[2031] = 0.0 + k[570]*y[IDX_EM] + k[571]*y[IDX_EM] +
        k[571]*y[IDX_EM] + k[957]*y[IDX_H2I];
    data[2032] = 0.0 + k[573]*y[IDX_EM] + k[573]*y[IDX_EM] +
        k[574]*y[IDX_EM] + k[2288];
    data[2033] = 0.0 + k[133]*y[IDX_HII] + k[930]*y[IDX_H2II] +
        k[1574]*y[IDX_C2H2I] + k[1631]*y[IDX_CH2I] + k[1686]*y[IDX_CHI] -
        k[1764]*y[IDX_HI] - k[1765]*y[IDX_HI] + k[1835]*y[IDX_NH2I] +
        k[1847]*y[IDX_NHI] + k[1945]*y[IDX_OHI];
    data[2034] = 0.0 - k[1763]*y[IDX_HI];
    data[2035] = 0.0 + k[134]*y[IDX_HII] - k[1766]*y[IDX_HI] -
        k[1767]*y[IDX_HI];
    data[2036] = 0.0 + k[136]*y[IDX_HII] + k[735]*y[IDX_CHII] +
        k[750]*y[IDX_CH2II] + k[783]*y[IDX_CH3II] + k[932]*y[IDX_H2II] +
        k[1063]*y[IDX_H3II] + k[1501]*y[IDX_HCSII] + k[1504]*y[IDX_HSII] +
        k[1507]*y[IDX_NH2II] + k[1511]*y[IDX_OHII] + k[1513]*y[IDX_SiHII] +
        k[1514]*y[IDX_SiH2II] + k[1638]*y[IDX_CH2I] + k[1638]*y[IDX_CH2I] +
        k[1639]*y[IDX_CH2I] + k[1664]*y[IDX_CH3I] + k[1665]*y[IDX_CH3I] +
        k[1693]*y[IDX_CHI] + k[1733]*y[IDX_H2I] + k[1851]*y[IDX_NHI] +
        k[1869]*y[IDX_C2H3I] + k[1891]*y[IDX_HCNI] + k[1892]*y[IDX_HCOI] +
        k[1895]*y[IDX_HCSI] + k[1896]*y[IDX_HNOI] + k[1900]*y[IDX_HSI] +
        k[1902]*y[IDX_NH2I] + k[1914]*y[IDX_OHI] + k[1923]*y[IDX_SiH2I] +
        k[1923]*y[IDX_SiH2I] + k[1924]*y[IDX_SiH3I] + k[1926]*y[IDX_SiHI] -
        k[2124]*y[IDX_HI];
    data[2037] = 0.0 - k[196]*y[IDX_HI] + k[857]*y[IDX_CHI] +
        k[958]*y[IDX_H2I] + k[1457]*y[IDX_NHI] + k[1480]*y[IDX_OHI];
    data[2038] = 0.0 - k[12]*y[IDX_HI] + k[12]*y[IDX_HI] + k[135]*y[IDX_HII]
        + k[931]*y[IDX_H2II] + k[1633]*y[IDX_CH2I] + k[1633]*y[IDX_CH2I] +
        k[1687]*y[IDX_CHI] + k[1688]*y[IDX_CHI] + k[1731]*y[IDX_H2I] -
        k[1768]*y[IDX_HI];
    data[2039] = 0.0 + k[972]*y[IDX_H2COI] + k[1482]*y[IDX_C2H2I] +
        k[1483]*y[IDX_CH3OHI];
    data[2040] = 0.0 + k[437] - k[1769]*y[IDX_HI] - k[1770]*y[IDX_HI] -
        k[1771]*y[IDX_HI] + k[2063];
    data[2041] = 0.0 + k[578]*y[IDX_EM];
    data[2042] = 0.0 - k[1772]*y[IDX_HI] - k[1773]*y[IDX_HI] -
        k[1774]*y[IDX_HI];
    data[2043] = 0.0 + k[137]*y[IDX_HII] + k[1695]*y[IDX_CHI] -
        k[1775]*y[IDX_HI];
    data[2044] = 0.0 + k[7]*y[IDX_H2I] - k[13]*y[IDX_HI] + k[13]*y[IDX_HI] +
        k[13]*y[IDX_HI] + k[138]*y[IDX_HII] + k[442] + k[637]*y[IDX_CII] +
        k[933]*y[IDX_H2II] + k[1268]*y[IDX_HeII] + k[1480]*y[IDX_OII] +
        k[1543]*y[IDX_HCOII] + k[1548]*y[IDX_SII] + k[1549]*y[IDX_SiII] +
        k[1611]*y[IDX_CI] + k[1641]*y[IDX_CH2I] + k[1696]*y[IDX_CHI] +
        k[1734]*y[IDX_H2I] - k[1776]*y[IDX_HI] + k[1827]*y[IDX_NI] +
        k[1854]*y[IDX_NHI] + k[1914]*y[IDX_OI] + k[1928]*y[IDX_C2H2I] +
        k[1933]*y[IDX_CNI] + k[1934]*y[IDX_COI] + k[1936]*y[IDX_CSI] +
        k[1945]*y[IDX_NOI] + k[1948]*y[IDX_SI] + k[1949]*y[IDX_SOI] +
        k[1950]*y[IDX_SiI] + k[2069] - k[2125]*y[IDX_HI];
    data[2045] = 0.0 + k[582]*y[IDX_EM] + k[960]*y[IDX_H2I] +
        k[1340]*y[IDX_NI] + k[1511]*y[IDX_OI] + k[1534]*y[IDX_SI] + k[2068];
    data[2046] = 0.0 + k[140]*y[IDX_HII] + k[739]*y[IDX_CHII] +
        k[753]*y[IDX_CH2II] + k[988]*y[IDX_H2OII] + k[1373]*y[IDX_NHII] +
        k[1392]*y[IDX_NH2II] + k[1534]*y[IDX_OHII] + k[1553]*y[IDX_H3SII] +
        k[1568]*y[IDX_SiH2II] + k[1645]*y[IDX_CH2I] + k[1669]*y[IDX_CH3I] +
        k[1697]*y[IDX_CHI] + k[1735]*y[IDX_H2I] + k[1857]*y[IDX_NHI] +
        k[1948]*y[IDX_OHI] + k[1951]*y[IDX_HCOI] + k[1953]*y[IDX_HSI];
    data[2047] = 0.0 + k[772]*y[IDX_CH2I] + k[790]*y[IDX_CH3I] +
        k[821]*y[IDX_CH4I] + k[822]*y[IDX_CH4I] + k[861]*y[IDX_CHI] +
        k[961]*y[IDX_H2I] + k[1461]*y[IDX_NHI] + k[1548]*y[IDX_OHI] +
        k[1550]*y[IDX_H2SI] + k[1569]*y[IDX_SiHI];
    data[2048] = 0.0 + k[139]*y[IDX_HII] - k[1777]*y[IDX_HI];
    data[2049] = 0.0 + k[143]*y[IDX_HII] + k[1950]*y[IDX_OHI];
    data[2050] = 0.0 + k[684]*y[IDX_C2HI] + k[862]*y[IDX_CHI] +
        k[1013]*y[IDX_H2OI] + k[1549]*y[IDX_OHI] - k[2126]*y[IDX_HI];
    data[2051] = 0.0 + k[146]*y[IDX_HII];
    data[2052] = 0.0 + k[144]*y[IDX_HII];
    data[2053] = 0.0 + k[145]*y[IDX_HII];
    data[2054] = 0.0 + k[150]*y[IDX_HII] + k[455] + k[644]*y[IDX_CII] +
        k[1284]*y[IDX_HeII] + k[1569]*y[IDX_SII] + k[1617]*y[IDX_CI] +
        k[1926]*y[IDX_OI] + k[2091];
    data[2055] = 0.0 + k[592]*y[IDX_EM] + k[703]*y[IDX_CI] -
        k[1108]*y[IDX_HI] + k[1513]*y[IDX_OI] + k[2082];
    data[2056] = 0.0 + k[147]*y[IDX_HII] + k[452] + k[1279]*y[IDX_HeII] +
        k[1923]*y[IDX_OI] + k[1923]*y[IDX_OI] + k[2084];
    data[2057] = 0.0 + k[594]*y[IDX_EM] + k[594]*y[IDX_EM] +
        k[595]*y[IDX_EM] + k[1514]*y[IDX_OI] + k[1568]*y[IDX_SI];
    data[2058] = 0.0 + k[148]*y[IDX_HII] + k[453] + k[1281]*y[IDX_HeII] +
        k[1924]*y[IDX_OI] + k[2085];
    data[2059] = 0.0 + k[596]*y[IDX_EM];
    data[2060] = 0.0 + k[149]*y[IDX_HII] + k[1283]*y[IDX_HeII] + k[2089] +
        k[2090];
    data[2061] = 0.0 + k[599]*y[IDX_EM] + k[963]*y[IDX_H2I];
    data[2062] = 0.0 + k[601]*y[IDX_EM] + k[2193];
    data[2063] = 0.0 + k[151]*y[IDX_HII];
    data[2064] = 0.0 + k[964]*y[IDX_H2I];
    data[2065] = 0.0 + k[604]*y[IDX_EM];
    data[2066] = 0.0 + k[152]*y[IDX_HII];
    data[2067] = 0.0 - k[1109]*y[IDX_HI];
    data[2068] = 0.0 + k[142]*y[IDX_HII] + k[1700]*y[IDX_CHI] -
        k[1778]*y[IDX_HI] - k[1779]*y[IDX_HI] + k[1949]*y[IDX_OHI];
    data[2069] = 0.0 + k[141]*y[IDX_HII];
    data[2070] = 0.0 + k[962]*y[IDX_H2I] - k[1107]*y[IDX_HI];
    data[2071] = 0.0 - k[110]*y[IDX_HII];
    data[2072] = 0.0 - k[112]*y[IDX_HII] - k[884]*y[IDX_HII];
    data[2073] = 0.0 - k[111]*y[IDX_HII];
    data[2074] = 0.0 - k[882]*y[IDX_HII];
    data[2075] = 0.0 - k[883]*y[IDX_HII];
    data[2076] = 0.0 - k[113]*y[IDX_HII];
    data[2077] = 0.0 - k[117]*y[IDX_HII];
    data[2078] = 0.0 + k[1977];
    data[2079] = 0.0 - k[114]*y[IDX_HII] - k[885]*y[IDX_HII];
    data[2080] = 0.0 + k[1980];
    data[2081] = 0.0 - k[115]*y[IDX_HII];
    data[2082] = 0.0 - k[886]*y[IDX_HII];
    data[2083] = 0.0 - k[887]*y[IDX_HII] - k[888]*y[IDX_HII] -
        k[889]*y[IDX_HII];
    data[2084] = 0.0 - k[116]*y[IDX_HII] - k[890]*y[IDX_HII] +
        k[1205]*y[IDX_HeII];
    data[2085] = 0.0 - k[109]*y[IDX_HII];
    data[2086] = 0.0 + k[108]*y[IDX_HI];
    data[2087] = 0.0 + k[191]*y[IDX_HI];
    data[2088] = 0.0 + k[192]*y[IDX_HI];
    data[2089] = 0.0 - k[891]*y[IDX_HII];
    data[2090] = 0.0 - k[118]*y[IDX_HII];
    data[2091] = 0.0 - k[2135]*y[IDX_HII];
    data[2092] = 0.0 + k[108]*y[IDX_ClII] + k[191]*y[IDX_CNII] +
        k[192]*y[IDX_COII] + k[193]*y[IDX_H2II] + k[194]*y[IDX_HCNII] +
        k[195]*y[IDX_HeII] + k[196]*y[IDX_OII] + k[362] + k[406] -
        k[2110]*y[IDX_HII];
    data[2093] = 0.0 - k[1]*y[IDX_HNCI] + k[1]*y[IDX_HNCI] -
        k[109]*y[IDX_ClI] - k[110]*y[IDX_C2I] - k[111]*y[IDX_C2H2I] -
        k[112]*y[IDX_C2HI] - k[113]*y[IDX_C2NI] - k[114]*y[IDX_CH2I] -
        k[115]*y[IDX_CH3I] - k[116]*y[IDX_CH4I] - k[117]*y[IDX_CHI] -
        k[118]*y[IDX_CSI] - k[119]*y[IDX_H2COI] - k[120]*y[IDX_H2CSI] -
        k[121]*y[IDX_H2OI] - k[122]*y[IDX_H2S2I] - k[123]*y[IDX_H2SI] -
        k[124]*y[IDX_HCNI] - k[125]*y[IDX_HCOI] - k[126]*y[IDX_HClI] -
        k[127]*y[IDX_HS2I] - k[128]*y[IDX_HSI] - k[129]*y[IDX_MgI] -
        k[130]*y[IDX_NH2I] - k[131]*y[IDX_NH3I] - k[132]*y[IDX_NHI] -
        k[133]*y[IDX_NOI] - k[134]*y[IDX_NSI] - k[135]*y[IDX_O2I] -
        k[136]*y[IDX_OI] - k[137]*y[IDX_OCSI] - k[138]*y[IDX_OHI] -
        k[139]*y[IDX_S2I] - k[140]*y[IDX_SI] - k[141]*y[IDX_SO2I] -
        k[142]*y[IDX_SOI] - k[143]*y[IDX_SiI] - k[144]*y[IDX_SiC2I] -
        k[145]*y[IDX_SiC3I] - k[146]*y[IDX_SiCI] - k[147]*y[IDX_SiH2I] -
        k[148]*y[IDX_SiH3I] - k[149]*y[IDX_SiH4I] - k[150]*y[IDX_SiHI] -
        k[151]*y[IDX_SiOI] - k[152]*y[IDX_SiSI] - k[882]*y[IDX_C2H3I] -
        k[883]*y[IDX_C2H4I] - k[884]*y[IDX_C2HI] - k[885]*y[IDX_CH2I] -
        k[886]*y[IDX_CH3CNI] - k[887]*y[IDX_CH3OHI] - k[888]*y[IDX_CH3OHI] -
        k[889]*y[IDX_CH3OHI] - k[890]*y[IDX_CH4I] - k[891]*y[IDX_CO2I] -
        k[892]*y[IDX_H2COI] - k[893]*y[IDX_H2COI] - k[894]*y[IDX_H2SI] -
        k[895]*y[IDX_H2SI] - k[896]*y[IDX_H2SiOI] - k[897]*y[IDX_HCOI] -
        k[898]*y[IDX_HCOI] - k[899]*y[IDX_HCSI] - k[900]*y[IDX_HNCOI] -
        k[901]*y[IDX_HNOI] - k[902]*y[IDX_HSI] - k[903]*y[IDX_NO2I] -
        k[904]*y[IDX_OCSI] - k[905]*y[IDX_SiH2I] - k[906]*y[IDX_SiH3I] -
        k[907]*y[IDX_SiH4I] - k[908]*y[IDX_SiHI] - k[2110]*y[IDX_HI] -
        k[2111]*y[IDX_HeI] - k[2135]*y[IDX_EM];
    data[2094] = 0.0 + k[359] + k[950]*y[IDX_HeII];
    data[2095] = 0.0 + k[193]*y[IDX_HI] + k[2009];
    data[2096] = 0.0 - k[119]*y[IDX_HII] - k[892]*y[IDX_HII] -
        k[893]*y[IDX_HII];
    data[2097] = 0.0 - k[120]*y[IDX_HII];
    data[2098] = 0.0 - k[121]*y[IDX_HII] + k[1223]*y[IDX_HeII];
    data[2099] = 0.0 - k[123]*y[IDX_HII] - k[894]*y[IDX_HII] -
        k[895]*y[IDX_HII];
    data[2100] = 0.0 - k[122]*y[IDX_HII];
    data[2101] = 0.0 - k[896]*y[IDX_HII];
    data[2102] = 0.0 + k[2026];
    data[2103] = 0.0 - k[126]*y[IDX_HII];
    data[2104] = 0.0 - k[124]*y[IDX_HII];
    data[2105] = 0.0 + k[194]*y[IDX_HI];
    data[2106] = 0.0 - k[125]*y[IDX_HII] - k[897]*y[IDX_HII] -
        k[898]*y[IDX_HII];
    data[2107] = 0.0 - k[899]*y[IDX_HII] + k[1240]*y[IDX_HeII];
    data[2108] = 0.0 - k[2111]*y[IDX_HII];
    data[2109] = 0.0 + k[195]*y[IDX_HI] + k[950]*y[IDX_H2I] +
        k[1205]*y[IDX_CH4I] + k[1223]*y[IDX_H2OI] + k[1240]*y[IDX_HCSI] +
        k[1246]*y[IDX_HNOI];
    data[2110] = 0.0 - k[1]*y[IDX_HII] + k[1]*y[IDX_HII];
    data[2111] = 0.0 - k[900]*y[IDX_HII];
    data[2112] = 0.0 - k[901]*y[IDX_HII] + k[1246]*y[IDX_HeII];
    data[2113] = 0.0 - k[128]*y[IDX_HII] - k[902]*y[IDX_HII];
    data[2114] = 0.0 + k[2040];
    data[2115] = 0.0 - k[127]*y[IDX_HII];
    data[2116] = 0.0 - k[129]*y[IDX_HII];
    data[2117] = 0.0 - k[132]*y[IDX_HII];
    data[2118] = 0.0 + k[2048];
    data[2119] = 0.0 - k[130]*y[IDX_HII];
    data[2120] = 0.0 - k[131]*y[IDX_HII];
    data[2121] = 0.0 - k[133]*y[IDX_HII];
    data[2122] = 0.0 - k[903]*y[IDX_HII];
    data[2123] = 0.0 - k[134]*y[IDX_HII];
    data[2124] = 0.0 - k[136]*y[IDX_HII];
    data[2125] = 0.0 + k[196]*y[IDX_HI];
    data[2126] = 0.0 - k[135]*y[IDX_HII];
    data[2127] = 0.0 - k[137]*y[IDX_HII] - k[904]*y[IDX_HII];
    data[2128] = 0.0 - k[138]*y[IDX_HII];
    data[2129] = 0.0 - k[140]*y[IDX_HII];
    data[2130] = 0.0 - k[139]*y[IDX_HII];
    data[2131] = 0.0 - k[143]*y[IDX_HII];
    data[2132] = 0.0 - k[146]*y[IDX_HII];
    data[2133] = 0.0 - k[144]*y[IDX_HII];
    data[2134] = 0.0 - k[145]*y[IDX_HII];
    data[2135] = 0.0 - k[150]*y[IDX_HII] - k[908]*y[IDX_HII];
    data[2136] = 0.0 - k[147]*y[IDX_HII] - k[905]*y[IDX_HII];
    data[2137] = 0.0 - k[148]*y[IDX_HII] - k[906]*y[IDX_HII];
    data[2138] = 0.0 - k[149]*y[IDX_HII] - k[907]*y[IDX_HII];
    data[2139] = 0.0 - k[151]*y[IDX_HII];
    data[2140] = 0.0 - k[152]*y[IDX_HII];
    data[2141] = 0.0 - k[142]*y[IDX_HII];
    data[2142] = 0.0 - k[141]*y[IDX_HII];
    data[2143] = 0.0 + k[688]*y[IDX_CH3II] + k[692]*y[IDX_H3OII] +
        k[1027]*y[IDX_H3II] + k[1593]*y[IDX_H2CNI] - k[1722]*y[IDX_H2I] -
        k[2113]*y[IDX_H2I];
    data[2144] = 0.0 + k[609]*y[IDX_CH3I] + k[614]*y[IDX_CH4I] +
        k[630]*y[IDX_NH3I] + k[643]*y[IDX_SiH2I] - k[934]*y[IDX_H2I] -
        k[2112]*y[IDX_H2I];
    data[2145] = 0.0 + k[153]*y[IDX_H2II] + k[1022]*y[IDX_H3II];
    data[2146] = 0.0 - k[935]*y[IDX_H2I];
    data[2147] = 0.0 + k[155]*y[IDX_H2II] + k[706]*y[IDX_CHII] +
        k[884]*y[IDX_HII] + k[1025]*y[IDX_H3II] - k[1721]*y[IDX_H2I];
    data[2148] = 0.0 - k[936]*y[IDX_H2I];
    data[2149] = 0.0 + k[154]*y[IDX_H2II] + k[1178]*y[IDX_HeII] +
        k[1575]*y[IDX_SiI] + k[1737]*y[IDX_HI];
    data[2150] = 0.0 + k[670]*y[IDX_SiI] + k[671]*y[IDX_SiH4I] +
        k[1328]*y[IDX_NI];
    data[2151] = 0.0 + k[882]*y[IDX_HII] + k[1181]*y[IDX_HeII] +
        k[1738]*y[IDX_HI];
    data[2152] = 0.0 + k[370] + k[774]*y[IDX_CH3II] + k[883]*y[IDX_HII] +
        k[910]*y[IDX_H2II] + k[910]*y[IDX_H2II] + k[1183]*y[IDX_HeII] +
        k[1184]*y[IDX_HeII] + k[1871]*y[IDX_OI] + k[1967];
    data[2153] = 0.0 + k[371] + k[1968];
    data[2154] = 0.0 + k[1024]*y[IDX_H3II];
    data[2155] = 0.0 + k[1026]*y[IDX_H3II];
    data[2156] = 0.0 - k[2]*y[IDX_H2I] + k[2]*y[IDX_H2I] +
        k[158]*y[IDX_H2II] + k[712]*y[IDX_CHII] + k[839]*y[IDX_CH3II] +
        k[1034]*y[IDX_H3II] - k[1725]*y[IDX_H2I] + k[1743]*y[IDX_HI] -
        k[2115]*y[IDX_H2I];
    data[2157] = 0.0 + k[706]*y[IDX_C2HI] + k[707]*y[IDX_CH2I] +
        k[711]*y[IDX_CH4I] + k[712]*y[IDX_CHI] + k[720]*y[IDX_H2OI] +
        k[722]*y[IDX_H2SI] + k[723]*y[IDX_HCNI] + k[729]*y[IDX_NH2I] +
        k[731]*y[IDX_NHI] + k[738]*y[IDX_OHI] - k[937]*y[IDX_H2I] +
        k[1098]*y[IDX_HI];
    data[2158] = 0.0 + k[156]*y[IDX_H2II] + k[707]*y[IDX_CHII] +
        k[885]*y[IDX_HII] + k[1028]*y[IDX_H3II] + k[1192]*y[IDX_HeII] +
        k[1618]*y[IDX_CH2I] + k[1618]*y[IDX_CH2I] + k[1632]*y[IDX_O2I] +
        k[1637]*y[IDX_OI] + k[1644]*y[IDX_SI] - k[1723]*y[IDX_H2I] +
        k[1739]*y[IDX_HI];
    data[2159] = 0.0 + k[478]*y[IDX_EM] + k[746]*y[IDX_H2SI] -
        k[938]*y[IDX_H2I] + k[1099]*y[IDX_HI] + k[1978];
    data[2160] = 0.0 + k[386] + k[609]*y[IDX_CII] + k[1029]*y[IDX_H3II] +
        k[1196]*y[IDX_HeII] + k[1647]*y[IDX_CH3I] + k[1647]*y[IDX_CH3I] +
        k[1664]*y[IDX_OI] + k[1667]*y[IDX_OHI] - k[1724]*y[IDX_H2I] +
        k[1741]*y[IDX_HI] + k[1805]*y[IDX_NI] + k[1988];
    data[2161] = 0.0 + k[482]*y[IDX_EM] + k[688]*y[IDX_CI] +
        k[774]*y[IDX_C2H4I] + k[778]*y[IDX_H2SI] + k[780]*y[IDX_HSI] +
        k[784]*y[IDX_OI] + k[786]*y[IDX_OHI] + k[787]*y[IDX_SI] +
        k[788]*y[IDX_SOI] + k[839]*y[IDX_CHI] + k[1100]*y[IDX_HI] +
        k[1446]*y[IDX_NHI] + k[1984] - k[2114]*y[IDX_H2I];
    data[2162] = 0.0 + k[1197]*y[IDX_HeII] + k[1197]*y[IDX_HeII];
    data[2163] = 0.0 + k[1030]*y[IDX_H3II];
    data[2164] = 0.0 + k[388] + k[888]*y[IDX_HII] + k[889]*y[IDX_HII] +
        k[889]*y[IDX_HII] + k[1031]*y[IDX_H3II] + k[1032]*y[IDX_H3II] + k[1990];
    data[2165] = 0.0 + k[490]*y[IDX_EM] + k[969]*y[IDX_H2COI];
    data[2166] = 0.0 + k[157]*y[IDX_H2II] + k[390] + k[614]*y[IDX_CII] +
        k[711]*y[IDX_CHII] + k[814]*y[IDX_HSII] + k[815]*y[IDX_N2II] +
        k[822]*y[IDX_SII] + k[890]*y[IDX_HII] + k[914]*y[IDX_H2II] +
        k[1033]*y[IDX_H3II] + k[1202]*y[IDX_HeII] + k[1203]*y[IDX_HeII] +
        k[1294]*y[IDX_NII] + k[1742]*y[IDX_HI] + k[1995] + k[1998];
    data[2167] = 0.0 - k[939]*y[IDX_H2I] + k[1101]*y[IDX_HI] + k[1993];
    data[2168] = 0.0 + k[493]*y[IDX_EM] + k[494]*y[IDX_EM] +
        k[497]*y[IDX_EM] + k[497]*y[IDX_EM] + k[836]*y[IDX_SiH4I] +
        k[1102]*y[IDX_HI] + k[1494]*y[IDX_OI];
    data[2169] = 0.0 + k[1040]*y[IDX_H3II] - k[1720]*y[IDX_H2I];
    data[2170] = 0.0 - k[944]*y[IDX_H2I];
    data[2171] = 0.0 + k[159]*y[IDX_H2II] + k[1035]*y[IDX_H3II] -
        k[1726]*y[IDX_H2I];
    data[2172] = 0.0 - k[940]*y[IDX_H2I];
    data[2173] = 0.0 + k[160]*y[IDX_H2II] + k[1037]*y[IDX_H3II] +
        k[1038]*y[IDX_H3II];
    data[2174] = 0.0 - k[941]*y[IDX_H2I] - k[942]*y[IDX_H2I];
    data[2175] = 0.0 + k[1036]*y[IDX_H3II];
    data[2176] = 0.0 + k[1039]*y[IDX_H3II];
    data[2177] = 0.0 - k[943]*y[IDX_H2I];
    data[2178] = 0.0 - k[8]*y[IDX_H2I] + k[478]*y[IDX_CH2II] +
        k[482]*y[IDX_CH3II] + k[490]*y[IDX_CH3OH2II] + k[493]*y[IDX_CH5II] +
        k[494]*y[IDX_CH5II] + k[497]*y[IDX_CH5II] + k[497]*y[IDX_CH5II] +
        k[503]*y[IDX_H2COII] + k[511]*y[IDX_H2NOII] + k[512]*y[IDX_H2OII] +
        k[519]*y[IDX_H3II] + k[523]*y[IDX_H3COII] + k[526]*y[IDX_H3CSII] +
        k[529]*y[IDX_H3OII] + k[530]*y[IDX_H3OII] + k[533]*y[IDX_H3SII] +
        k[535]*y[IDX_H3SII] + k[572]*y[IDX_NH4II] + k[593]*y[IDX_SiH2II] +
        k[597]*y[IDX_SiH3II] + k[598]*y[IDX_SiH4II] + k[600]*y[IDX_SiH5II];
    data[2179] = 0.0 - k[10]*y[IDX_H2I] + k[193]*y[IDX_H2II] +
        k[1098]*y[IDX_CHII] + k[1099]*y[IDX_CH2II] + k[1100]*y[IDX_CH3II] +
        k[1101]*y[IDX_CH4II] + k[1102]*y[IDX_CH5II] + k[1103]*y[IDX_H2SII] +
        k[1104]*y[IDX_H3SII] + k[1105]*y[IDX_HSII] + k[1108]*y[IDX_SiHII] +
        k[1737]*y[IDX_C2H2I] + k[1738]*y[IDX_C2H3I] + k[1739]*y[IDX_CH2I] +
        k[1741]*y[IDX_CH3I] + k[1742]*y[IDX_CH4I] + k[1743]*y[IDX_CHI] +
        k[1746]*y[IDX_H2CNI] + k[1747]*y[IDX_H2COI] + k[1748]*y[IDX_H2OI] +
        k[1749]*y[IDX_H2SI] + k[1750]*y[IDX_HCNI] + k[1751]*y[IDX_HCOI] +
        k[1753]*y[IDX_HCSI] + k[1756]*y[IDX_HNOI] + k[1758]*y[IDX_HSI] +
        k[1760]*y[IDX_NH2I] + k[1761]*y[IDX_NH3I] + k[1762]*y[IDX_NHI] +
        k[1770]*y[IDX_O2HI] + k[1776]*y[IDX_OHI] + k[1787]*y[IDX_HClI];
    data[2180] = 0.0 + k[882]*y[IDX_C2H3I] + k[883]*y[IDX_C2H4I] +
        k[884]*y[IDX_C2HI] + k[885]*y[IDX_CH2I] + k[888]*y[IDX_CH3OHI] +
        k[889]*y[IDX_CH3OHI] + k[889]*y[IDX_CH3OHI] + k[890]*y[IDX_CH4I] +
        k[892]*y[IDX_H2COI] + k[893]*y[IDX_H2COI] + k[894]*y[IDX_H2SI] +
        k[895]*y[IDX_H2SI] + k[896]*y[IDX_H2SiOI] + k[897]*y[IDX_HCOI] +
        k[899]*y[IDX_HCSI] + k[901]*y[IDX_HNOI] + k[902]*y[IDX_HSI] +
        k[905]*y[IDX_SiH2I] + k[906]*y[IDX_SiH3I] + k[907]*y[IDX_SiH4I] +
        k[908]*y[IDX_SiHI];
    data[2181] = 0.0 - k[2]*y[IDX_CHI] + k[2]*y[IDX_CHI] - k[3]*y[IDX_H2I] -
        k[3]*y[IDX_H2I] - k[3]*y[IDX_H2I] - k[3]*y[IDX_H2I] + k[3]*y[IDX_H2I] +
        k[3]*y[IDX_H2I] - k[4]*y[IDX_H2OI] + k[4]*y[IDX_H2OI] -
        k[5]*y[IDX_HOCII] + k[5]*y[IDX_HOCII] - k[6]*y[IDX_O2I] +
        k[6]*y[IDX_O2I] - k[7]*y[IDX_OHI] + k[7]*y[IDX_OHI] - k[8]*y[IDX_EM] -
        k[10]*y[IDX_HI] - k[172]*y[IDX_HeII] - k[359] - k[360] - k[361] -
        k[920]*y[IDX_H2II] - k[934]*y[IDX_CII] - k[935]*y[IDX_C2II] -
        k[936]*y[IDX_C2HII] - k[937]*y[IDX_CHII] - k[938]*y[IDX_CH2II] -
        k[939]*y[IDX_CH4II] - k[940]*y[IDX_CNII] - k[941]*y[IDX_COII] -
        k[942]*y[IDX_COII] - k[943]*y[IDX_CSII] - k[944]*y[IDX_ClII] -
        k[945]*y[IDX_H2OII] - k[946]*y[IDX_H2SII] - k[947]*y[IDX_HCNII] -
        k[948]*y[IDX_HClII] - k[949]*y[IDX_HSII] - k[950]*y[IDX_HeII] -
        k[951]*y[IDX_HeHII] - k[952]*y[IDX_NII] - k[953]*y[IDX_N2II] -
        k[954]*y[IDX_NHII] - k[955]*y[IDX_NHII] - k[956]*y[IDX_NH2II] -
        k[957]*y[IDX_NH3II] - k[958]*y[IDX_OII] - k[959]*y[IDX_O2HII] -
        k[960]*y[IDX_OHII] - k[961]*y[IDX_SII] - k[962]*y[IDX_SO2II] -
        k[963]*y[IDX_SiH4II] - k[964]*y[IDX_SiOII] - k[1720]*y[IDX_ClI] -
        k[1721]*y[IDX_C2HI] - k[1722]*y[IDX_CI] - k[1723]*y[IDX_CH2I] -
        k[1724]*y[IDX_CH3I] - k[1725]*y[IDX_CHI] - k[1726]*y[IDX_CNI] -
        k[1727]*y[IDX_HSI] - k[1728]*y[IDX_NI] - k[1729]*y[IDX_NH2I] -
        k[1730]*y[IDX_NHI] - k[1731]*y[IDX_O2I] - k[1732]*y[IDX_O2I] -
        k[1733]*y[IDX_OI] - k[1734]*y[IDX_OHI] - k[1735]*y[IDX_SI] -
        k[2112]*y[IDX_CII] - k[2113]*y[IDX_CI] - k[2114]*y[IDX_CH3II] -
        k[2115]*y[IDX_CHI] - k[2116]*y[IDX_HSII] - k[2117]*y[IDX_SII] -
        k[2118]*y[IDX_SiII] - k[2119]*y[IDX_SiHII] - k[2120]*y[IDX_SiH3II];
    data[2182] = 0.0 + k[153]*y[IDX_C2I] + k[154]*y[IDX_C2H2I] +
        k[155]*y[IDX_C2HI] + k[156]*y[IDX_CH2I] + k[157]*y[IDX_CH4I] +
        k[158]*y[IDX_CHI] + k[159]*y[IDX_CNI] + k[160]*y[IDX_COI] +
        k[161]*y[IDX_H2COI] + k[162]*y[IDX_H2OI] + k[163]*y[IDX_H2SI] +
        k[164]*y[IDX_HCNI] + k[165]*y[IDX_HCOI] + k[166]*y[IDX_NH2I] +
        k[167]*y[IDX_NH3I] + k[168]*y[IDX_NHI] + k[169]*y[IDX_NOI] +
        k[170]*y[IDX_O2I] + k[171]*y[IDX_OHI] + k[193]*y[IDX_HI] +
        k[910]*y[IDX_C2H4I] + k[910]*y[IDX_C2H4I] + k[914]*y[IDX_CH4I] -
        k[920]*y[IDX_H2I] + k[921]*y[IDX_H2COI] + k[923]*y[IDX_H2SI] +
        k[924]*y[IDX_H2SI] + k[924]*y[IDX_H2SI];
    data[2183] = 0.0 + k[1593]*y[IDX_CI] + k[1746]*y[IDX_HI] +
        k[1885]*y[IDX_OI];
    data[2184] = 0.0 + k[161]*y[IDX_H2II] + k[399] + k[892]*y[IDX_HII] +
        k[893]*y[IDX_HII] + k[921]*y[IDX_H2II] + k[969]*y[IDX_CH3OH2II] +
        k[1041]*y[IDX_H3II] + k[1216]*y[IDX_HeII] + k[1747]*y[IDX_HI] + k[2011];
    data[2185] = 0.0 + k[503]*y[IDX_EM];
    data[2186] = 0.0 + k[400] + k[1042]*y[IDX_H3II] + k[1219]*y[IDX_HeII] +
        k[2015];
    data[2187] = 0.0 + k[511]*y[IDX_EM];
    data[2188] = 0.0 - k[4]*y[IDX_H2I] + k[4]*y[IDX_H2I] +
        k[162]*y[IDX_H2II] + k[720]*y[IDX_CHII] + k[1043]*y[IDX_H3II] +
        k[1357]*y[IDX_NHII] + k[1748]*y[IDX_HI];
    data[2189] = 0.0 + k[512]*y[IDX_EM] - k[945]*y[IDX_H2I] +
        k[1334]*y[IDX_NI] + k[1497]*y[IDX_OI];
    data[2190] = 0.0 + k[163]*y[IDX_H2II] + k[404] + k[722]*y[IDX_CHII] +
        k[746]*y[IDX_CH2II] + k[778]*y[IDX_CH3II] + k[894]*y[IDX_HII] +
        k[895]*y[IDX_HII] + k[923]*y[IDX_H2II] + k[924]*y[IDX_H2II] +
        k[924]*y[IDX_H2II] + k[1044]*y[IDX_H3II] + k[1176]*y[IDX_HSII] +
        k[1227]*y[IDX_HeII] + k[1316]*y[IDX_N2II] + k[1551]*y[IDX_SII] +
        k[1749]*y[IDX_HI] + k[2022];
    data[2191] = 0.0 - k[946]*y[IDX_H2I] + k[1103]*y[IDX_HI] +
        k[1335]*y[IDX_NI] + k[1499]*y[IDX_OI];
    data[2192] = 0.0 + k[405] + k[896]*y[IDX_HII] + k[2023];
    data[2193] = 0.0 + k[519]*y[IDX_EM] + k[1022]*y[IDX_C2I] +
        k[1024]*y[IDX_C2H5OHI] + k[1025]*y[IDX_C2HI] + k[1026]*y[IDX_C2NI] +
        k[1027]*y[IDX_CI] + k[1028]*y[IDX_CH2I] + k[1029]*y[IDX_CH3I] +
        k[1030]*y[IDX_CH3CNI] + k[1031]*y[IDX_CH3OHI] + k[1032]*y[IDX_CH3OHI] +
        k[1033]*y[IDX_CH4I] + k[1034]*y[IDX_CHI] + k[1035]*y[IDX_CNI] +
        k[1036]*y[IDX_CO2I] + k[1037]*y[IDX_COI] + k[1038]*y[IDX_COI] +
        k[1039]*y[IDX_CSI] + k[1040]*y[IDX_ClI] + k[1041]*y[IDX_H2COI] +
        k[1042]*y[IDX_H2CSI] + k[1043]*y[IDX_H2OI] + k[1044]*y[IDX_H2SI] +
        k[1045]*y[IDX_HCNI] + k[1046]*y[IDX_HCOI] + k[1047]*y[IDX_HCOOCH3I] +
        k[1048]*y[IDX_HCSI] + k[1049]*y[IDX_HClI] + k[1050]*y[IDX_HNCI] +
        k[1051]*y[IDX_HNOI] + k[1052]*y[IDX_HS2I] + k[1053]*y[IDX_HSI] +
        k[1054]*y[IDX_MgI] + k[1055]*y[IDX_N2I] + k[1056]*y[IDX_NH2I] +
        k[1057]*y[IDX_NH3I] + k[1058]*y[IDX_NHI] + k[1059]*y[IDX_NO2I] +
        k[1060]*y[IDX_NOI] + k[1061]*y[IDX_NSI] + k[1062]*y[IDX_O2I] +
        k[1064]*y[IDX_OI] + k[1065]*y[IDX_OCSI] + k[1066]*y[IDX_OHI] +
        k[1067]*y[IDX_S2I] + k[1068]*y[IDX_SI] + k[1069]*y[IDX_SO2I] +
        k[1070]*y[IDX_SOI] + k[1071]*y[IDX_SiI] + k[1072]*y[IDX_SiH2I] +
        k[1073]*y[IDX_SiH3I] + k[1074]*y[IDX_SiH4I] + k[1075]*y[IDX_SiHI] +
        k[1076]*y[IDX_SiOI] + k[1077]*y[IDX_SiSI] + k[2026];
    data[2194] = 0.0 + k[523]*y[IDX_EM];
    data[2195] = 0.0 + k[526]*y[IDX_EM];
    data[2196] = 0.0 + k[529]*y[IDX_EM] + k[530]*y[IDX_EM] +
        k[692]*y[IDX_CI];
    data[2197] = 0.0 + k[533]*y[IDX_EM] + k[535]*y[IDX_EM] +
        k[1104]*y[IDX_HI];
    data[2198] = 0.0 + k[1049]*y[IDX_H3II] + k[1787]*y[IDX_HI];
    data[2199] = 0.0 - k[948]*y[IDX_H2I];
    data[2200] = 0.0 + k[164]*y[IDX_H2II] + k[723]*y[IDX_CHII] +
        k[1045]*y[IDX_H3II] + k[1750]*y[IDX_HI];
    data[2201] = 0.0 - k[947]*y[IDX_H2I];
    data[2202] = 0.0 + k[165]*y[IDX_H2II] + k[897]*y[IDX_HII] +
        k[1046]*y[IDX_H3II] + k[1751]*y[IDX_HI] + k[1780]*y[IDX_HCOI] +
        k[1780]*y[IDX_HCOI];
    data[2203] = 0.0 + k[1047]*y[IDX_H3II];
    data[2204] = 0.0 + k[899]*y[IDX_HII] + k[1048]*y[IDX_H3II] +
        k[1753]*y[IDX_HI];
    data[2205] = 0.0 - k[172]*y[IDX_H2I] - k[950]*y[IDX_H2I] +
        k[1178]*y[IDX_C2H2I] + k[1181]*y[IDX_C2H3I] + k[1183]*y[IDX_C2H4I] +
        k[1184]*y[IDX_C2H4I] + k[1192]*y[IDX_CH2I] + k[1196]*y[IDX_CH3I] +
        k[1197]*y[IDX_CH3CCHI] + k[1197]*y[IDX_CH3CCHI] + k[1202]*y[IDX_CH4I] +
        k[1203]*y[IDX_CH4I] + k[1216]*y[IDX_H2COI] + k[1219]*y[IDX_H2CSI] +
        k[1227]*y[IDX_H2SI] + k[1252]*y[IDX_NH2I] + k[1254]*y[IDX_NH3I] +
        k[1278]*y[IDX_SiH2I] + k[1280]*y[IDX_SiH3I] + k[1282]*y[IDX_SiH4I] +
        k[1282]*y[IDX_SiH4I] + k[1283]*y[IDX_SiH4I];
    data[2206] = 0.0 - k[951]*y[IDX_H2I];
    data[2207] = 0.0 + k[1050]*y[IDX_H3II];
    data[2208] = 0.0 + k[901]*y[IDX_HII] + k[1051]*y[IDX_H3II] +
        k[1756]*y[IDX_HI];
    data[2209] = 0.0 - k[5]*y[IDX_H2I] + k[5]*y[IDX_H2I];
    data[2210] = 0.0 + k[780]*y[IDX_CH3II] + k[902]*y[IDX_HII] +
        k[1053]*y[IDX_H3II] - k[1727]*y[IDX_H2I] + k[1758]*y[IDX_HI];
    data[2211] = 0.0 + k[814]*y[IDX_CH4I] - k[949]*y[IDX_H2I] +
        k[1105]*y[IDX_HI] + k[1176]*y[IDX_H2SI] - k[2116]*y[IDX_H2I];
    data[2212] = 0.0 + k[1052]*y[IDX_H3II];
    data[2213] = 0.0 + k[1054]*y[IDX_H3II];
    data[2214] = 0.0 + k[1328]*y[IDX_C2H2II] + k[1334]*y[IDX_H2OII] +
        k[1335]*y[IDX_H2SII] - k[1728]*y[IDX_H2I] + k[1805]*y[IDX_CH3I];
    data[2215] = 0.0 - k[952]*y[IDX_H2I] + k[1294]*y[IDX_CH4I] +
        k[1306]*y[IDX_NH3I];
    data[2216] = 0.0 + k[1055]*y[IDX_H3II];
    data[2217] = 0.0 + k[815]*y[IDX_CH4I] - k[953]*y[IDX_H2I] +
        k[1316]*y[IDX_H2SI];
    data[2218] = 0.0 + k[168]*y[IDX_H2II] + k[731]*y[IDX_CHII] +
        k[1058]*y[IDX_H3II] + k[1446]*y[IDX_CH3II] - k[1730]*y[IDX_H2I] +
        k[1762]*y[IDX_HI] + k[1843]*y[IDX_NHI] + k[1843]*y[IDX_NHI];
    data[2219] = 0.0 - k[954]*y[IDX_H2I] - k[955]*y[IDX_H2I] +
        k[1357]*y[IDX_H2OI];
    data[2220] = 0.0 + k[166]*y[IDX_H2II] + k[729]*y[IDX_CHII] +
        k[1056]*y[IDX_H3II] + k[1252]*y[IDX_HeII] - k[1729]*y[IDX_H2I] +
        k[1760]*y[IDX_HI];
    data[2221] = 0.0 - k[956]*y[IDX_H2I];
    data[2222] = 0.0 + k[167]*y[IDX_H2II] + k[428] + k[630]*y[IDX_CII] +
        k[1057]*y[IDX_H3II] + k[1254]*y[IDX_HeII] + k[1306]*y[IDX_NII] +
        k[1761]*y[IDX_HI] + k[2053];
    data[2223] = 0.0 - k[957]*y[IDX_H2I] + k[1508]*y[IDX_OI];
    data[2224] = 0.0 + k[572]*y[IDX_EM];
    data[2225] = 0.0 + k[169]*y[IDX_H2II] + k[1060]*y[IDX_H3II];
    data[2226] = 0.0 + k[1059]*y[IDX_H3II];
    data[2227] = 0.0 + k[1061]*y[IDX_H3II];
    data[2228] = 0.0 + k[784]*y[IDX_CH3II] + k[1064]*y[IDX_H3II] +
        k[1494]*y[IDX_CH5II] + k[1497]*y[IDX_H2OII] + k[1499]*y[IDX_H2SII] +
        k[1508]*y[IDX_NH3II] + k[1515]*y[IDX_SiH3II] + k[1637]*y[IDX_CH2I] +
        k[1664]*y[IDX_CH3I] - k[1733]*y[IDX_H2I] + k[1871]*y[IDX_C2H4I] +
        k[1885]*y[IDX_H2CNI] + k[1922]*y[IDX_SiH2I];
    data[2229] = 0.0 - k[958]*y[IDX_H2I];
    data[2230] = 0.0 - k[6]*y[IDX_H2I] + k[6]*y[IDX_H2I] +
        k[170]*y[IDX_H2II] + k[1062]*y[IDX_H3II] + k[1632]*y[IDX_CH2I] -
        k[1731]*y[IDX_H2I] - k[1732]*y[IDX_H2I];
    data[2231] = 0.0 + k[1770]*y[IDX_HI];
    data[2232] = 0.0 - k[959]*y[IDX_H2I];
    data[2233] = 0.0 + k[1065]*y[IDX_H3II];
    data[2234] = 0.0 - k[7]*y[IDX_H2I] + k[7]*y[IDX_H2I] +
        k[171]*y[IDX_H2II] + k[738]*y[IDX_CHII] + k[786]*y[IDX_CH3II] +
        k[1066]*y[IDX_H3II] + k[1667]*y[IDX_CH3I] - k[1734]*y[IDX_H2I] +
        k[1776]*y[IDX_HI];
    data[2235] = 0.0 - k[960]*y[IDX_H2I];
    data[2236] = 0.0 + k[787]*y[IDX_CH3II] + k[1068]*y[IDX_H3II] +
        k[1644]*y[IDX_CH2I] - k[1735]*y[IDX_H2I];
    data[2237] = 0.0 + k[822]*y[IDX_CH4I] - k[961]*y[IDX_H2I] +
        k[1551]*y[IDX_H2SI] - k[2117]*y[IDX_H2I];
    data[2238] = 0.0 + k[1067]*y[IDX_H3II];
    data[2239] = 0.0 + k[670]*y[IDX_C2H2II] + k[1071]*y[IDX_H3II] +
        k[1575]*y[IDX_C2H2I];
    data[2240] = 0.0 - k[2118]*y[IDX_H2I];
    data[2241] = 0.0 + k[908]*y[IDX_HII] + k[1075]*y[IDX_H3II];
    data[2242] = 0.0 + k[1108]*y[IDX_HI] - k[2119]*y[IDX_H2I];
    data[2243] = 0.0 + k[643]*y[IDX_CII] + k[905]*y[IDX_HII] +
        k[1072]*y[IDX_H3II] + k[1278]*y[IDX_HeII] + k[1922]*y[IDX_OI];
    data[2244] = 0.0 + k[593]*y[IDX_EM];
    data[2245] = 0.0 + k[906]*y[IDX_HII] + k[1073]*y[IDX_H3II] +
        k[1280]*y[IDX_HeII] + k[2087];
    data[2246] = 0.0 + k[597]*y[IDX_EM] + k[1515]*y[IDX_OI] -
        k[2120]*y[IDX_H2I];
    data[2247] = 0.0 + k[454] + k[671]*y[IDX_C2H2II] + k[836]*y[IDX_CH5II] +
        k[907]*y[IDX_HII] + k[1074]*y[IDX_H3II] + k[1282]*y[IDX_HeII] +
        k[1282]*y[IDX_HeII] + k[1283]*y[IDX_HeII] + k[2088] + k[2090];
    data[2248] = 0.0 + k[598]*y[IDX_EM] - k[963]*y[IDX_H2I];
    data[2249] = 0.0 + k[600]*y[IDX_EM];
    data[2250] = 0.0 + k[1076]*y[IDX_H3II];
    data[2251] = 0.0 - k[964]*y[IDX_H2I];
    data[2252] = 0.0 + k[1077]*y[IDX_H3II];
    data[2253] = 0.0 + k[788]*y[IDX_CH3II] + k[1070]*y[IDX_H3II];
    data[2254] = 0.0 + k[1069]*y[IDX_H3II];
    data[2255] = 0.0 - k[962]*y[IDX_H2I];
    data[2256] = 0.0 - k[912]*y[IDX_H2II];
    data[2257] = 0.0 - k[153]*y[IDX_H2II] - k[909]*y[IDX_H2II];
    data[2258] = 0.0 - k[155]*y[IDX_H2II] - k[911]*y[IDX_H2II];
    data[2259] = 0.0 - k[154]*y[IDX_H2II];
    data[2260] = 0.0 - k[910]*y[IDX_H2II];
    data[2261] = 0.0 - k[158]*y[IDX_H2II] - k[916]*y[IDX_H2II];
    data[2262] = 0.0 - k[156]*y[IDX_H2II] - k[913]*y[IDX_H2II];
    data[2263] = 0.0 - k[157]*y[IDX_H2II] - k[914]*y[IDX_H2II] -
        k[915]*y[IDX_H2II];
    data[2264] = 0.0 - k[159]*y[IDX_H2II] - k[917]*y[IDX_H2II];
    data[2265] = 0.0 - k[160]*y[IDX_H2II] - k[919]*y[IDX_H2II];
    data[2266] = 0.0 - k[918]*y[IDX_H2II];
    data[2267] = 0.0 - k[501]*y[IDX_H2II];
    data[2268] = 0.0 - k[193]*y[IDX_H2II] + k[1106]*y[IDX_HeHII] +
        k[2110]*y[IDX_HII];
    data[2269] = 0.0 + k[898]*y[IDX_HCOI] + k[2110]*y[IDX_HI];
    data[2270] = 0.0 + k[172]*y[IDX_HeII] + k[360] - k[920]*y[IDX_H2II];
    data[2271] = 0.0 - k[153]*y[IDX_C2I] - k[154]*y[IDX_C2H2I] -
        k[155]*y[IDX_C2HI] - k[156]*y[IDX_CH2I] - k[157]*y[IDX_CH4I] -
        k[158]*y[IDX_CHI] - k[159]*y[IDX_CNI] - k[160]*y[IDX_COI] -
        k[161]*y[IDX_H2COI] - k[162]*y[IDX_H2OI] - k[163]*y[IDX_H2SI] -
        k[164]*y[IDX_HCNI] - k[165]*y[IDX_HCOI] - k[166]*y[IDX_NH2I] -
        k[167]*y[IDX_NH3I] - k[168]*y[IDX_NHI] - k[169]*y[IDX_NOI] -
        k[170]*y[IDX_O2I] - k[171]*y[IDX_OHI] - k[193]*y[IDX_HI] -
        k[501]*y[IDX_EM] - k[909]*y[IDX_C2I] - k[910]*y[IDX_C2H4I] -
        k[911]*y[IDX_C2HI] - k[912]*y[IDX_CI] - k[913]*y[IDX_CH2I] -
        k[914]*y[IDX_CH4I] - k[915]*y[IDX_CH4I] - k[916]*y[IDX_CHI] -
        k[917]*y[IDX_CNI] - k[918]*y[IDX_CO2I] - k[919]*y[IDX_COI] -
        k[920]*y[IDX_H2I] - k[921]*y[IDX_H2COI] - k[922]*y[IDX_H2OI] -
        k[923]*y[IDX_H2SI] - k[924]*y[IDX_H2SI] - k[925]*y[IDX_HCOI] -
        k[926]*y[IDX_HeI] - k[927]*y[IDX_N2I] - k[928]*y[IDX_NI] -
        k[929]*y[IDX_NHI] - k[930]*y[IDX_NOI] - k[931]*y[IDX_O2I] -
        k[932]*y[IDX_OI] - k[933]*y[IDX_OHI] - k[2009];
    data[2272] = 0.0 - k[161]*y[IDX_H2II] - k[921]*y[IDX_H2II];
    data[2273] = 0.0 - k[162]*y[IDX_H2II] - k[922]*y[IDX_H2II];
    data[2274] = 0.0 - k[163]*y[IDX_H2II] - k[923]*y[IDX_H2II] -
        k[924]*y[IDX_H2II];
    data[2275] = 0.0 + k[2025];
    data[2276] = 0.0 - k[164]*y[IDX_H2II];
    data[2277] = 0.0 - k[165]*y[IDX_H2II] + k[898]*y[IDX_HII] -
        k[925]*y[IDX_H2II];
    data[2278] = 0.0 - k[926]*y[IDX_H2II];
    data[2279] = 0.0 + k[172]*y[IDX_H2I];
    data[2280] = 0.0 + k[1106]*y[IDX_HI];
    data[2281] = 0.0 - k[928]*y[IDX_H2II];
    data[2282] = 0.0 - k[927]*y[IDX_H2II];
    data[2283] = 0.0 - k[168]*y[IDX_H2II] - k[929]*y[IDX_H2II];
    data[2284] = 0.0 - k[166]*y[IDX_H2II];
    data[2285] = 0.0 - k[167]*y[IDX_H2II];
    data[2286] = 0.0 - k[169]*y[IDX_H2II] - k[930]*y[IDX_H2II];
    data[2287] = 0.0 - k[932]*y[IDX_H2II];
    data[2288] = 0.0 - k[170]*y[IDX_H2II] - k[931]*y[IDX_H2II];
    data[2289] = 0.0 - k[171]*y[IDX_H2II] - k[933]*y[IDX_H2II];
    data[2290] = 0.0 + k[832]*y[IDX_HClI];
    data[2291] = 0.0 - k[874]*y[IDX_H2ClII];
    data[2292] = 0.0 - k[508]*y[IDX_H2ClII] - k[509]*y[IDX_H2ClII];
    data[2293] = 0.0 + k[948]*y[IDX_HClII];
    data[2294] = 0.0 - k[508]*y[IDX_EM] - k[509]*y[IDX_EM] -
        k[874]*y[IDX_COI] - k[999]*y[IDX_H2OI] - k[2198];
    data[2295] = 0.0 - k[999]*y[IDX_H2ClII];
    data[2296] = 0.0 + k[1049]*y[IDX_HClI];
    data[2297] = 0.0 + k[832]*y[IDX_CH5II] + k[1049]*y[IDX_H3II];
    data[2298] = 0.0 + k[948]*y[IDX_H2I];
    data[2299] = 0.0 + k[2355] + k[2356] + k[2357] + k[2358];
    data[2300] = 0.0 - k[1593]*y[IDX_H2CNI];
    data[2301] = 0.0 + k[1794]*y[IDX_NI];
    data[2302] = 0.0 + k[1804]*y[IDX_NI];
    data[2303] = 0.0 - k[1746]*y[IDX_H2CNI];
    data[2304] = 0.0 - k[398] - k[1593]*y[IDX_CI] - k[1746]*y[IDX_HI] -
        k[1810]*y[IDX_NI] - k[1885]*y[IDX_OI] - k[2010] - k[2275];
    data[2305] = 0.0 + k[1794]*y[IDX_C2H5I] + k[1804]*y[IDX_CH3I] -
        k[1810]*y[IDX_H2CNI];
    data[2306] = 0.0 - k[1885]*y[IDX_H2CNI];
    data[2307] = 0.0 + k[2367] + k[2368] + k[2369] + k[2370];
    data[2308] = 0.0 - k[16]*y[IDX_H2COI] - k[617]*y[IDX_H2COI] -
        k[618]*y[IDX_H2COI];
    data[2309] = 0.0 - k[660]*y[IDX_H2COI];
    data[2310] = 0.0 - k[42]*y[IDX_H2COI];
    data[2311] = 0.0 + k[1576]*y[IDX_O2I];
    data[2312] = 0.0 + k[1559]*y[IDX_SOII] + k[1872]*y[IDX_OI];
    data[2313] = 0.0 + k[1874]*y[IDX_OI];
    data[2314] = 0.0 + k[85]*y[IDX_H2COII] + k[844]*y[IDX_H3COII] -
        k[1678]*y[IDX_H2COI];
    data[2315] = 0.0 + k[709]*y[IDX_CH3OHI] - k[715]*y[IDX_H2COI] -
        k[716]*y[IDX_H2COI] - k[717]*y[IDX_H2COI];
    data[2316] = 0.0 + k[65]*y[IDX_H2COII] + k[773]*y[IDX_SiOII] -
        k[1624]*y[IDX_H2COI] + k[1628]*y[IDX_NO2I] + k[1629]*y[IDX_NOI] +
        k[1635]*y[IDX_O2I] + k[1641]*y[IDX_OHI];
    data[2317] = 0.0 - k[742]*y[IDX_H2COI];
    data[2318] = 0.0 - k[1651]*y[IDX_H2COI] + k[1658]*y[IDX_NO2I] +
        k[1660]*y[IDX_O2I] + k[1665]*y[IDX_OI] + k[1667]*y[IDX_OHI];
    data[2319] = 0.0 - k[777]*y[IDX_H2COI];
    data[2320] = 0.0 + k[388] + k[709]*y[IDX_CHII] + k[793]*y[IDX_S2II] +
        k[1078]*y[IDX_H3COII] + k[1990];
    data[2321] = 0.0 + k[490]*y[IDX_EM] - k[969]*y[IDX_H2COI];
    data[2322] = 0.0 - k[76]*y[IDX_H2COI] - k[798]*y[IDX_H2COI];
    data[2323] = 0.0 - k[827]*y[IDX_H2COI];
    data[2324] = 0.0 - k[1704]*y[IDX_H2COI];
    data[2325] = 0.0 - k[94]*y[IDX_H2COI] - k[865]*y[IDX_H2COI];
    data[2326] = 0.0 - k[101]*y[IDX_H2COI] - k[871]*y[IDX_H2COI];
    data[2327] = 0.0 + k[490]*y[IDX_CH3OH2II] + k[524]*y[IDX_H3COII] +
        k[2136]*y[IDX_H2COII];
    data[2328] = 0.0 - k[1747]*y[IDX_H2COI];
    data[2329] = 0.0 - k[119]*y[IDX_H2COI] - k[892]*y[IDX_H2COI] -
        k[893]*y[IDX_H2COI];
    data[2330] = 0.0 - k[161]*y[IDX_H2COI] - k[921]*y[IDX_H2COI];
    data[2331] = 0.0 - k[16]*y[IDX_CII] - k[42]*y[IDX_C2H2II] -
        k[76]*y[IDX_CH4II] - k[94]*y[IDX_CNII] - k[101]*y[IDX_COII] -
        k[119]*y[IDX_HII] - k[161]*y[IDX_H2II] - k[174]*y[IDX_O2II] -
        k[178]*y[IDX_H2OII] - k[212]*y[IDX_HeII] - k[239]*y[IDX_NII] -
        k[252]*y[IDX_N2II] - k[260]*y[IDX_NHII] - k[311]*y[IDX_OII] -
        k[331]*y[IDX_OHII] - k[399] - k[617]*y[IDX_CII] - k[618]*y[IDX_CII] -
        k[660]*y[IDX_C2HII] - k[715]*y[IDX_CHII] - k[716]*y[IDX_CHII] -
        k[717]*y[IDX_CHII] - k[742]*y[IDX_CH2II] - k[777]*y[IDX_CH3II] -
        k[798]*y[IDX_CH4II] - k[827]*y[IDX_CH5II] - k[865]*y[IDX_CNII] -
        k[871]*y[IDX_COII] - k[892]*y[IDX_HII] - k[893]*y[IDX_HII] -
        k[921]*y[IDX_H2II] - k[966]*y[IDX_H2COII] - k[969]*y[IDX_CH3OH2II] -
        k[970]*y[IDX_H3SII] - k[971]*y[IDX_HNOII] - k[972]*y[IDX_O2II] -
        k[973]*y[IDX_O2HII] - k[974]*y[IDX_SII] - k[975]*y[IDX_SII] -
        k[979]*y[IDX_H2OII] - k[1041]*y[IDX_H3II] - k[1086]*y[IDX_H3OII] -
        k[1112]*y[IDX_HCNII] - k[1132]*y[IDX_HCNHII] - k[1133]*y[IDX_HCNHII] -
        k[1141]*y[IDX_HCOII] - k[1216]*y[IDX_HeII] - k[1217]*y[IDX_HeII] -
        k[1218]*y[IDX_HeII] - k[1298]*y[IDX_NII] - k[1299]*y[IDX_NII] -
        k[1314]*y[IDX_N2II] - k[1323]*y[IDX_N2HII] - k[1354]*y[IDX_NHII] -
        k[1355]*y[IDX_NHII] - k[1376]*y[IDX_NH2II] - k[1377]*y[IDX_NH2II] -
        k[1413]*y[IDX_NH3II] - k[1471]*y[IDX_OII] - k[1522]*y[IDX_OHII] -
        k[1624]*y[IDX_CH2I] - k[1651]*y[IDX_CH3I] - k[1678]*y[IDX_CHI] -
        k[1704]*y[IDX_CNI] - k[1747]*y[IDX_HI] - k[1886]*y[IDX_OI] -
        k[1937]*y[IDX_OHI] - k[2011] - k[2012] - k[2013] - k[2014] - k[2208];
    data[2332] = 0.0 + k[65]*y[IDX_CH2I] + k[85]*y[IDX_CHI] +
        k[173]*y[IDX_SI] + k[202]*y[IDX_HCOI] + k[222]*y[IDX_MgI] +
        k[284]*y[IDX_NH3I] + k[298]*y[IDX_NOI] + k[349]*y[IDX_SiI] -
        k[966]*y[IDX_H2COI] + k[2136]*y[IDX_EM];
    data[2333] = 0.0 + k[1001]*y[IDX_H3COII];
    data[2334] = 0.0 - k[178]*y[IDX_H2COI] - k[979]*y[IDX_H2COI];
    data[2335] = 0.0 + k[1079]*y[IDX_H3COII];
    data[2336] = 0.0 - k[1041]*y[IDX_H2COI];
    data[2337] = 0.0 + k[524]*y[IDX_EM] + k[844]*y[IDX_CHI] +
        k[1001]*y[IDX_H2OI] + k[1078]*y[IDX_CH3OHI] + k[1079]*y[IDX_H2SI] +
        k[1122]*y[IDX_HCNI] + k[1166]*y[IDX_HNCI] + k[1401]*y[IDX_NH2I] +
        k[1427]*y[IDX_NH3I];
    data[2338] = 0.0 - k[1086]*y[IDX_H2COI];
    data[2339] = 0.0 - k[970]*y[IDX_H2COI];
    data[2340] = 0.0 + k[1122]*y[IDX_H3COII];
    data[2341] = 0.0 - k[1112]*y[IDX_H2COI];
    data[2342] = 0.0 - k[1132]*y[IDX_H2COI] - k[1133]*y[IDX_H2COI];
    data[2343] = 0.0 + k[202]*y[IDX_H2COII] + k[1781]*y[IDX_HCOI] +
        k[1781]*y[IDX_HCOI] + k[1782]*y[IDX_HNOI] + k[1786]*y[IDX_O2HI];
    data[2344] = 0.0 - k[1141]*y[IDX_H2COI];
    data[2345] = 0.0 + k[2032] + k[2032];
    data[2346] = 0.0 - k[212]*y[IDX_H2COI] - k[1216]*y[IDX_H2COI] -
        k[1217]*y[IDX_H2COI] - k[1218]*y[IDX_H2COI];
    data[2347] = 0.0 + k[1166]*y[IDX_H3COII];
    data[2348] = 0.0 + k[1782]*y[IDX_HCOI];
    data[2349] = 0.0 - k[971]*y[IDX_H2COI];
    data[2350] = 0.0 + k[222]*y[IDX_H2COII];
    data[2351] = 0.0 - k[239]*y[IDX_H2COI] - k[1298]*y[IDX_H2COI] -
        k[1299]*y[IDX_H2COI];
    data[2352] = 0.0 - k[252]*y[IDX_H2COI] - k[1314]*y[IDX_H2COI];
    data[2353] = 0.0 - k[1323]*y[IDX_H2COI];
    data[2354] = 0.0 - k[260]*y[IDX_H2COI] - k[1354]*y[IDX_H2COI] -
        k[1355]*y[IDX_H2COI];
    data[2355] = 0.0 + k[1401]*y[IDX_H3COII];
    data[2356] = 0.0 - k[1376]*y[IDX_H2COI] - k[1377]*y[IDX_H2COI];
    data[2357] = 0.0 + k[284]*y[IDX_H2COII] + k[1427]*y[IDX_H3COII];
    data[2358] = 0.0 - k[1413]*y[IDX_H2COI];
    data[2359] = 0.0 + k[298]*y[IDX_H2COII] + k[1629]*y[IDX_CH2I];
    data[2360] = 0.0 + k[1628]*y[IDX_CH2I] + k[1658]*y[IDX_CH3I];
    data[2361] = 0.0 + k[1665]*y[IDX_CH3I] + k[1872]*y[IDX_C2H4I] +
        k[1874]*y[IDX_C2H5I] - k[1886]*y[IDX_H2COI];
    data[2362] = 0.0 - k[311]*y[IDX_H2COI] - k[1471]*y[IDX_H2COI];
    data[2363] = 0.0 + k[1576]*y[IDX_C2H3I] + k[1635]*y[IDX_CH2I] +
        k[1660]*y[IDX_CH3I];
    data[2364] = 0.0 - k[174]*y[IDX_H2COI] - k[972]*y[IDX_H2COI];
    data[2365] = 0.0 + k[1786]*y[IDX_HCOI];
    data[2366] = 0.0 - k[973]*y[IDX_H2COI];
    data[2367] = 0.0 + k[1641]*y[IDX_CH2I] + k[1667]*y[IDX_CH3I] -
        k[1937]*y[IDX_H2COI];
    data[2368] = 0.0 - k[331]*y[IDX_H2COI] - k[1522]*y[IDX_H2COI];
    data[2369] = 0.0 + k[173]*y[IDX_H2COII];
    data[2370] = 0.0 - k[974]*y[IDX_H2COI] - k[975]*y[IDX_H2COI];
    data[2371] = 0.0 + k[793]*y[IDX_CH3OHI];
    data[2372] = 0.0 + k[349]*y[IDX_H2COII];
    data[2373] = 0.0 + k[773]*y[IDX_CH2I];
    data[2374] = 0.0 + k[1559]*y[IDX_C2H4I];
    data[2375] = 0.0 + k[16]*y[IDX_H2COI];
    data[2376] = 0.0 - k[651]*y[IDX_H2COII];
    data[2377] = 0.0 - k[678]*y[IDX_H2COII];
    data[2378] = 0.0 + k[42]*y[IDX_H2COI];
    data[2379] = 0.0 - k[85]*y[IDX_H2COII] - k[842]*y[IDX_H2COII];
    data[2380] = 0.0 + k[718]*y[IDX_H2OI];
    data[2381] = 0.0 - k[65]*y[IDX_H2COII] - k[757]*y[IDX_H2COII] +
        k[769]*y[IDX_O2II];
    data[2382] = 0.0 + k[741]*y[IDX_CO2I];
    data[2383] = 0.0 + k[783]*y[IDX_OI] + k[786]*y[IDX_OHI];
    data[2384] = 0.0 - k[965]*y[IDX_H2COII] + k[1289]*y[IDX_NII] +
        k[1466]*y[IDX_OII];
    data[2385] = 0.0 - k[809]*y[IDX_H2COII];
    data[2386] = 0.0 + k[76]*y[IDX_H2COI];
    data[2387] = 0.0 + k[831]*y[IDX_HCOI];
    data[2388] = 0.0 + k[94]*y[IDX_H2COI];
    data[2389] = 0.0 + k[101]*y[IDX_H2COI];
    data[2390] = 0.0 + k[741]*y[IDX_CH2II];
    data[2391] = 0.0 - k[502]*y[IDX_H2COII] - k[503]*y[IDX_H2COII] -
        k[504]*y[IDX_H2COII] - k[505]*y[IDX_H2COII] - k[2136]*y[IDX_H2COII];
    data[2392] = 0.0 + k[119]*y[IDX_H2COI];
    data[2393] = 0.0 + k[161]*y[IDX_H2COI];
    data[2394] = 0.0 + k[16]*y[IDX_CII] + k[42]*y[IDX_C2H2II] +
        k[76]*y[IDX_CH4II] + k[94]*y[IDX_CNII] + k[101]*y[IDX_COII] +
        k[119]*y[IDX_HII] + k[161]*y[IDX_H2II] + k[174]*y[IDX_O2II] +
        k[178]*y[IDX_H2OII] + k[212]*y[IDX_HeII] + k[239]*y[IDX_NII] +
        k[252]*y[IDX_N2II] + k[260]*y[IDX_NHII] + k[311]*y[IDX_OII] +
        k[331]*y[IDX_OHII] - k[966]*y[IDX_H2COII] + k[2013];
    data[2395] = 0.0 - k[65]*y[IDX_CH2I] - k[85]*y[IDX_CHI] -
        k[173]*y[IDX_SI] - k[202]*y[IDX_HCOI] - k[222]*y[IDX_MgI] -
        k[284]*y[IDX_NH3I] - k[298]*y[IDX_NOI] - k[349]*y[IDX_SiI] -
        k[502]*y[IDX_EM] - k[503]*y[IDX_EM] - k[504]*y[IDX_EM] -
        k[505]*y[IDX_EM] - k[651]*y[IDX_C2I] - k[678]*y[IDX_C2HI] -
        k[757]*y[IDX_CH2I] - k[809]*y[IDX_CH4I] - k[842]*y[IDX_CHI] -
        k[965]*y[IDX_CH3OHI] - k[966]*y[IDX_H2COI] - k[967]*y[IDX_O2I] -
        k[968]*y[IDX_SI] - k[998]*y[IDX_H2OI] - k[1121]*y[IDX_HCNI] -
        k[1158]*y[IDX_HCOI] - k[1165]*y[IDX_HNCI] - k[1399]*y[IDX_NH2I] -
        k[1424]*y[IDX_NH3I] - k[1449]*y[IDX_NHI] - k[2136]*y[IDX_EM] - k[2244];
    data[2396] = 0.0 + k[718]*y[IDX_CHII] - k[998]*y[IDX_H2COII];
    data[2397] = 0.0 + k[178]*y[IDX_H2COI] + k[985]*y[IDX_HCOI];
    data[2398] = 0.0 + k[1046]*y[IDX_HCOI];
    data[2399] = 0.0 - k[1121]*y[IDX_H2COII];
    data[2400] = 0.0 + k[1114]*y[IDX_HCOI];
    data[2401] = 0.0 - k[202]*y[IDX_H2COII] + k[831]*y[IDX_CH5II] +
        k[985]*y[IDX_H2OII] + k[1046]*y[IDX_H3II] + k[1114]*y[IDX_HCNII] +
        k[1144]*y[IDX_HCOII] - k[1158]*y[IDX_H2COII] + k[1159]*y[IDX_HNOII] +
        k[1160]*y[IDX_N2HII] + k[1162]*y[IDX_O2HII] + k[1361]*y[IDX_NHII] +
        k[1386]*y[IDX_NH2II] + k[1527]*y[IDX_OHII];
    data[2402] = 0.0 + k[1144]*y[IDX_HCOI];
    data[2403] = 0.0 + k[212]*y[IDX_H2COI];
    data[2404] = 0.0 - k[1165]*y[IDX_H2COII];
    data[2405] = 0.0 + k[1159]*y[IDX_HCOI];
    data[2406] = 0.0 - k[222]*y[IDX_H2COII];
    data[2407] = 0.0 + k[239]*y[IDX_H2COI] + k[1289]*y[IDX_CH3OHI];
    data[2408] = 0.0 + k[252]*y[IDX_H2COI];
    data[2409] = 0.0 + k[1160]*y[IDX_HCOI];
    data[2410] = 0.0 - k[1449]*y[IDX_H2COII];
    data[2411] = 0.0 + k[260]*y[IDX_H2COI] + k[1361]*y[IDX_HCOI];
    data[2412] = 0.0 - k[1399]*y[IDX_H2COII];
    data[2413] = 0.0 + k[1386]*y[IDX_HCOI];
    data[2414] = 0.0 - k[284]*y[IDX_H2COII] - k[1424]*y[IDX_H2COII];
    data[2415] = 0.0 - k[298]*y[IDX_H2COII];
    data[2416] = 0.0 + k[783]*y[IDX_CH3II];
    data[2417] = 0.0 + k[311]*y[IDX_H2COI] + k[1466]*y[IDX_CH3OHI];
    data[2418] = 0.0 - k[967]*y[IDX_H2COII];
    data[2419] = 0.0 + k[174]*y[IDX_H2COI] + k[769]*y[IDX_CH2I];
    data[2420] = 0.0 + k[1162]*y[IDX_HCOI];
    data[2421] = 0.0 + k[786]*y[IDX_CH3II];
    data[2422] = 0.0 + k[331]*y[IDX_H2COI] + k[1527]*y[IDX_HCOI];
    data[2423] = 0.0 - k[173]*y[IDX_H2COII] - k[968]*y[IDX_H2COII];
    data[2424] = 0.0 - k[349]*y[IDX_H2COII];
    data[2425] = 0.0 + k[2443] + k[2444] + k[2445] + k[2446];
    data[2426] = 0.0 - k[619]*y[IDX_H2CSI];
    data[2427] = 0.0 + k[1669]*y[IDX_SI];
    data[2428] = 0.0 + k[527]*y[IDX_H3CSII] + k[2137]*y[IDX_H2CSII];
    data[2429] = 0.0 - k[120]*y[IDX_H2CSI];
    data[2430] = 0.0 - k[120]*y[IDX_HII] - k[400] - k[619]*y[IDX_CII] -
        k[1042]*y[IDX_H3II] - k[1142]*y[IDX_HCOII] - k[1219]*y[IDX_HeII] -
        k[1220]*y[IDX_HeII] - k[1221]*y[IDX_HeII] - k[2015] - k[2263];
    data[2431] = 0.0 + k[2137]*y[IDX_EM];
    data[2432] = 0.0 - k[1042]*y[IDX_H2CSI];
    data[2433] = 0.0 + k[527]*y[IDX_EM];
    data[2434] = 0.0 - k[1142]*y[IDX_H2CSI];
    data[2435] = 0.0 - k[1219]*y[IDX_H2CSI] - k[1220]*y[IDX_H2CSI] -
        k[1221]*y[IDX_H2CSI];
    data[2436] = 0.0 + k[1669]*y[IDX_CH3I];
    data[2437] = 0.0 + k[1556]*y[IDX_SOII];
    data[2438] = 0.0 + k[1559]*y[IDX_SOII];
    data[2439] = 0.0 + k[751]*y[IDX_OCSI];
    data[2440] = 0.0 + k[790]*y[IDX_SII];
    data[2441] = 0.0 + k[780]*y[IDX_HSI];
    data[2442] = 0.0 - k[506]*y[IDX_H2CSII] - k[507]*y[IDX_H2CSII] -
        k[2137]*y[IDX_H2CSII];
    data[2443] = 0.0 + k[120]*y[IDX_H2CSI];
    data[2444] = 0.0 + k[120]*y[IDX_HII];
    data[2445] = 0.0 - k[506]*y[IDX_EM] - k[507]*y[IDX_EM] -
        k[2137]*y[IDX_EM] - k[2279];
    data[2446] = 0.0 + k[1048]*y[IDX_HCSI];
    data[2447] = 0.0 + k[1048]*y[IDX_H3II];
    data[2448] = 0.0 + k[780]*y[IDX_CH3II];
    data[2449] = 0.0 + k[751]*y[IDX_CH2II];
    data[2450] = 0.0 + k[790]*y[IDX_CH3I];
    data[2451] = 0.0 + k[1556]*y[IDX_C2H2I] + k[1559]*y[IDX_C2H4I];
    data[2452] = 0.0 - k[510]*y[IDX_H2NOII] - k[511]*y[IDX_H2NOII];
    data[2453] = 0.0 - k[510]*y[IDX_EM] - k[511]*y[IDX_EM] - k[2273];
    data[2454] = 0.0 + k[1051]*y[IDX_HNOI];
    data[2455] = 0.0 + k[1051]*y[IDX_H3II];
    data[2456] = 0.0 + k[1390]*y[IDX_O2I];
    data[2457] = 0.0 + k[1390]*y[IDX_NH2II];
    data[2458] = 0.0 + k[2311] + k[2312] + k[2313] + k[2314];
    data[2459] = 0.0 - k[620]*y[IDX_H2OI] - k[621]*y[IDX_H2OI];
    data[2460] = 0.0 + k[175]*y[IDX_H2OII] + k[1080]*y[IDX_H3OII];
    data[2461] = 0.0 - k[990]*y[IDX_H2OI];
    data[2462] = 0.0 + k[177]*y[IDX_H2OII];
    data[2463] = 0.0 + k[176]*y[IDX_H2OII] + k[1927]*y[IDX_OHI];
    data[2464] = 0.0 - k[991]*y[IDX_H2OI];
    data[2465] = 0.0 + k[1930]*y[IDX_OHI];
    data[2466] = 0.0 + k[1464]*y[IDX_OII];
    data[2467] = 0.0 + k[1931]*y[IDX_OHI];
    data[2468] = 0.0 + k[1023]*y[IDX_H3II] + k[1081]*y[IDX_H3OII];
    data[2469] = 0.0 + k[464]*y[IDX_EM] + k[465]*y[IDX_EM];
    data[2470] = 0.0 - k[992]*y[IDX_H2OI] - k[993]*y[IDX_H2OI];
    data[2471] = 0.0 - k[994]*y[IDX_H2OI];
    data[2472] = 0.0 + k[86]*y[IDX_H2OII] + k[845]*y[IDX_H3OII];
    data[2473] = 0.0 - k[718]*y[IDX_H2OI] - k[719]*y[IDX_H2OI] -
        k[720]*y[IDX_H2OI];
    data[2474] = 0.0 + k[66]*y[IDX_H2OII] + k[759]*y[IDX_H3OII] +
        k[1634]*y[IDX_O2I] + k[1642]*y[IDX_OHI];
    data[2475] = 0.0 - k[743]*y[IDX_H2OI];
    data[2476] = 0.0 - k[1652]*y[IDX_H2OI] + k[1659]*y[IDX_NOI] +
        k[1661]*y[IDX_O2I] + k[1668]*y[IDX_OHI];
    data[2477] = 0.0 - k[2107]*y[IDX_H2OI];
    data[2478] = 0.0 + k[1082]*y[IDX_H3OII];
    data[2479] = 0.0 + k[1083]*y[IDX_H3OII];
    data[2480] = 0.0 + k[887]*y[IDX_HII] + k[1031]*y[IDX_H3II] +
        k[1084]*y[IDX_H3OII] + k[1466]*y[IDX_OII];
    data[2481] = 0.0 + k[486]*y[IDX_EM] + k[487]*y[IDX_EM] +
        k[1120]*y[IDX_HCNI];
    data[2482] = 0.0 + k[1672]*y[IDX_OHI];
    data[2483] = 0.0 - k[799]*y[IDX_H2OI];
    data[2484] = 0.0 - k[828]*y[IDX_H2OI];
    data[2485] = 0.0 - k[995]*y[IDX_H2OI] - k[996]*y[IDX_H2OI];
    data[2486] = 0.0 - k[187]*y[IDX_H2OI] - k[997]*y[IDX_H2OI];
    data[2487] = 0.0 + k[1085]*y[IDX_H3OII];
    data[2488] = 0.0 + k[464]*y[IDX_C2H5OH2II] + k[465]*y[IDX_C2H5OH2II] +
        k[486]*y[IDX_CH3OH2II] + k[487]*y[IDX_CH3OH2II] + k[522]*y[IDX_H3COII] +
        k[528]*y[IDX_H3OII];
    data[2489] = 0.0 - k[11]*y[IDX_H2OI] - k[1748]*y[IDX_H2OI] +
        k[1769]*y[IDX_O2HI] + k[2125]*y[IDX_OHI];
    data[2490] = 0.0 - k[121]*y[IDX_H2OI] + k[887]*y[IDX_CH3OHI];
    data[2491] = 0.0 - k[4]*y[IDX_H2OI] + k[1734]*y[IDX_OHI];
    data[2492] = 0.0 - k[162]*y[IDX_H2OI] - k[922]*y[IDX_H2OI];
    data[2493] = 0.0 - k[999]*y[IDX_H2OI];
    data[2494] = 0.0 + k[178]*y[IDX_H2OII] + k[1086]*y[IDX_H3OII] +
        k[1937]*y[IDX_OHI];
    data[2495] = 0.0 - k[998]*y[IDX_H2OI];
    data[2496] = 0.0 - k[4]*y[IDX_H2I] - k[11]*y[IDX_HI] - k[121]*y[IDX_HII]
        - k[162]*y[IDX_H2II] - k[187]*y[IDX_COII] - k[188]*y[IDX_HCNII] -
        k[189]*y[IDX_N2II] - k[213]*y[IDX_HeII] - k[240]*y[IDX_NII] -
        k[261]*y[IDX_NHII] - k[312]*y[IDX_OII] - k[332]*y[IDX_OHII] - k[401] -
        k[620]*y[IDX_CII] - k[621]*y[IDX_CII] - k[718]*y[IDX_CHII] -
        k[719]*y[IDX_CHII] - k[720]*y[IDX_CHII] - k[743]*y[IDX_CH2II] -
        k[799]*y[IDX_CH4II] - k[828]*y[IDX_CH5II] - k[922]*y[IDX_H2II] -
        k[980]*y[IDX_H2OII] - k[990]*y[IDX_C2II] - k[991]*y[IDX_C2H2II] -
        k[992]*y[IDX_C2NII] - k[993]*y[IDX_C2NII] - k[994]*y[IDX_C4NII] -
        k[995]*y[IDX_CNII] - k[996]*y[IDX_CNII] - k[997]*y[IDX_COII] -
        k[998]*y[IDX_H2COII] - k[999]*y[IDX_H2ClII] - k[1000]*y[IDX_H2SII] -
        k[1001]*y[IDX_H3COII] - k[1002]*y[IDX_HCNII] - k[1003]*y[IDX_HCOII] -
        k[1004]*y[IDX_HCO2II] - k[1005]*y[IDX_HNOII] - k[1006]*y[IDX_HOCSII] -
        k[1007]*y[IDX_HSII] - k[1008]*y[IDX_HSO2II] - k[1009]*y[IDX_HSiSII] -
        k[1010]*y[IDX_N2II] - k[1011]*y[IDX_N2HII] - k[1012]*y[IDX_O2HII] -
        k[1013]*y[IDX_SiII] - k[1014]*y[IDX_SiHII] - k[1015]*y[IDX_SiH4II] -
        k[1016]*y[IDX_SiH5II] - k[1043]*y[IDX_H3II] - k[1222]*y[IDX_HeII] -
        k[1223]*y[IDX_HeII] - k[1356]*y[IDX_NHII] - k[1357]*y[IDX_NHII] -
        k[1358]*y[IDX_NHII] - k[1359]*y[IDX_NHII] - k[1378]*y[IDX_NH2II] -
        k[1379]*y[IDX_NH2II] - k[1380]*y[IDX_NH2II] - k[1414]*y[IDX_NH3II] -
        k[1523]*y[IDX_OHII] - k[1652]*y[IDX_CH3I] - k[1748]*y[IDX_HI] -
        k[1841]*y[IDX_NHI] - k[1887]*y[IDX_OI] - k[2017] - k[2018] -
        k[2107]*y[IDX_CH3II] - k[2214];
    data[2497] = 0.0 + k[66]*y[IDX_CH2I] + k[86]*y[IDX_CHI] +
        k[175]*y[IDX_C2I] + k[176]*y[IDX_C2H2I] + k[177]*y[IDX_C2HI] +
        k[178]*y[IDX_H2COI] + k[179]*y[IDX_H2SI] + k[180]*y[IDX_HCOI] +
        k[181]*y[IDX_MgI] + k[182]*y[IDX_NOI] + k[183]*y[IDX_O2I] +
        k[184]*y[IDX_OCSI] + k[185]*y[IDX_SI] + k[186]*y[IDX_SiI] +
        k[274]*y[IDX_NH2I] + k[285]*y[IDX_NH3I] - k[980]*y[IDX_H2OI];
    data[2498] = 0.0 + k[179]*y[IDX_H2OII] + k[1021]*y[IDX_SOII] +
        k[1087]*y[IDX_H3OII] + k[1473]*y[IDX_OII] + k[1938]*y[IDX_OHI];
    data[2499] = 0.0 - k[1000]*y[IDX_H2OI];
    data[2500] = 0.0 + k[1023]*y[IDX_C2H5OHI] + k[1031]*y[IDX_CH3OHI] -
        k[1043]*y[IDX_H2OI];
    data[2501] = 0.0 + k[522]*y[IDX_EM] - k[1001]*y[IDX_H2OI];
    data[2502] = 0.0 + k[528]*y[IDX_EM] + k[759]*y[IDX_CH2I] +
        k[845]*y[IDX_CHI] + k[1080]*y[IDX_C2I] + k[1081]*y[IDX_C2H5OHI] +
        k[1082]*y[IDX_CH3CCHI] + k[1083]*y[IDX_CH3CNI] + k[1084]*y[IDX_CH3OHI] +
        k[1085]*y[IDX_CSI] + k[1086]*y[IDX_H2COI] + k[1087]*y[IDX_H2SI] +
        k[1088]*y[IDX_HCNI] + k[1089]*y[IDX_HCOOCH3I] + k[1090]*y[IDX_HNCI] +
        k[1091]*y[IDX_HS2I] + k[1092]*y[IDX_S2I] + k[1093]*y[IDX_SiI] +
        k[1094]*y[IDX_SiH2I] + k[1095]*y[IDX_SiHI] + k[1096]*y[IDX_SiOI] +
        k[1402]*y[IDX_NH2I] + k[1428]*y[IDX_NH3I];
    data[2503] = 0.0 + k[1088]*y[IDX_H3OII] + k[1120]*y[IDX_CH3OH2II] +
        k[1939]*y[IDX_OHI];
    data[2504] = 0.0 - k[188]*y[IDX_H2OI] - k[1002]*y[IDX_H2OI];
    data[2505] = 0.0 + k[180]*y[IDX_H2OII] + k[1941]*y[IDX_OHI];
    data[2506] = 0.0 - k[1003]*y[IDX_H2OI];
    data[2507] = 0.0 - k[1004]*y[IDX_H2OI];
    data[2508] = 0.0 + k[1089]*y[IDX_H3OII];
    data[2509] = 0.0 - k[213]*y[IDX_H2OI] - k[1222]*y[IDX_H2OI] -
        k[1223]*y[IDX_H2OI];
    data[2510] = 0.0 + k[1090]*y[IDX_H3OII];
    data[2511] = 0.0 + k[1942]*y[IDX_OHI];
    data[2512] = 0.0 - k[1005]*y[IDX_H2OI];
    data[2513] = 0.0 - k[1006]*y[IDX_H2OI];
    data[2514] = 0.0 - k[1007]*y[IDX_H2OI];
    data[2515] = 0.0 + k[1091]*y[IDX_H3OII];
    data[2516] = 0.0 - k[1009]*y[IDX_H2OI];
    data[2517] = 0.0 - k[1008]*y[IDX_H2OI];
    data[2518] = 0.0 + k[181]*y[IDX_H2OII];
    data[2519] = 0.0 - k[240]*y[IDX_H2OI];
    data[2520] = 0.0 - k[189]*y[IDX_H2OI] - k[1010]*y[IDX_H2OI];
    data[2521] = 0.0 - k[1011]*y[IDX_H2OI];
    data[2522] = 0.0 - k[1841]*y[IDX_H2OI] + k[1853]*y[IDX_OHI];
    data[2523] = 0.0 - k[261]*y[IDX_H2OI] - k[1356]*y[IDX_H2OI] -
        k[1357]*y[IDX_H2OI] - k[1358]*y[IDX_H2OI] - k[1359]*y[IDX_H2OI];
    data[2524] = 0.0 + k[274]*y[IDX_H2OII] + k[1402]*y[IDX_H3OII] +
        k[1834]*y[IDX_NOI] + k[1836]*y[IDX_OHI];
    data[2525] = 0.0 - k[1378]*y[IDX_H2OI] - k[1379]*y[IDX_H2OI] -
        k[1380]*y[IDX_H2OI];
    data[2526] = 0.0 + k[285]*y[IDX_H2OII] + k[1428]*y[IDX_H3OII] +
        k[1944]*y[IDX_OHI];
    data[2527] = 0.0 - k[1414]*y[IDX_H2OI];
    data[2528] = 0.0 + k[182]*y[IDX_H2OII] + k[1659]*y[IDX_CH3I] +
        k[1834]*y[IDX_NH2I];
    data[2529] = 0.0 - k[1887]*y[IDX_H2OI];
    data[2530] = 0.0 - k[312]*y[IDX_H2OI] + k[1464]*y[IDX_C2H4I] +
        k[1466]*y[IDX_CH3OHI] + k[1473]*y[IDX_H2SI];
    data[2531] = 0.0 + k[183]*y[IDX_H2OII] + k[1634]*y[IDX_CH2I] +
        k[1661]*y[IDX_CH3I];
    data[2532] = 0.0 + k[1769]*y[IDX_HI] + k[1946]*y[IDX_OHI];
    data[2533] = 0.0 - k[1012]*y[IDX_H2OI];
    data[2534] = 0.0 + k[184]*y[IDX_H2OII];
    data[2535] = 0.0 + k[1642]*y[IDX_CH2I] + k[1668]*y[IDX_CH3I] +
        k[1672]*y[IDX_CH4I] + k[1734]*y[IDX_H2I] + k[1836]*y[IDX_NH2I] +
        k[1853]*y[IDX_NHI] + k[1927]*y[IDX_C2H2I] + k[1930]*y[IDX_C2H3I] +
        k[1931]*y[IDX_C2H5I] + k[1937]*y[IDX_H2COI] + k[1938]*y[IDX_H2SI] +
        k[1939]*y[IDX_HCNI] + k[1941]*y[IDX_HCOI] + k[1942]*y[IDX_HNOI] +
        k[1944]*y[IDX_NH3I] + k[1946]*y[IDX_O2HI] + k[1947]*y[IDX_OHI] +
        k[1947]*y[IDX_OHI] + k[2125]*y[IDX_HI];
    data[2536] = 0.0 - k[332]*y[IDX_H2OI] - k[1523]*y[IDX_H2OI];
    data[2537] = 0.0 + k[185]*y[IDX_H2OII];
    data[2538] = 0.0 + k[1092]*y[IDX_H3OII];
    data[2539] = 0.0 + k[186]*y[IDX_H2OII] + k[1093]*y[IDX_H3OII];
    data[2540] = 0.0 - k[1013]*y[IDX_H2OI];
    data[2541] = 0.0 + k[1095]*y[IDX_H3OII];
    data[2542] = 0.0 - k[1014]*y[IDX_H2OI];
    data[2543] = 0.0 + k[1094]*y[IDX_H3OII];
    data[2544] = 0.0 - k[1015]*y[IDX_H2OI];
    data[2545] = 0.0 - k[1016]*y[IDX_H2OI];
    data[2546] = 0.0 + k[1096]*y[IDX_H3OII];
    data[2547] = 0.0 + k[1021]*y[IDX_H2SI];
    data[2548] = 0.0 - k[690]*y[IDX_H2OII];
    data[2549] = 0.0 - k[175]*y[IDX_H2OII] - k[976]*y[IDX_H2OII];
    data[2550] = 0.0 - k[177]*y[IDX_H2OII] - k[977]*y[IDX_H2OII];
    data[2551] = 0.0 - k[176]*y[IDX_H2OII];
    data[2552] = 0.0 - k[86]*y[IDX_H2OII] - k[843]*y[IDX_H2OII];
    data[2553] = 0.0 - k[66]*y[IDX_H2OII] - k[758]*y[IDX_H2OII];
    data[2554] = 0.0 - k[810]*y[IDX_H2OII];
    data[2555] = 0.0 + k[1538]*y[IDX_OHI];
    data[2556] = 0.0 - k[978]*y[IDX_H2OII];
    data[2557] = 0.0 + k[187]*y[IDX_H2OI];
    data[2558] = 0.0 - k[512]*y[IDX_H2OII] - k[513]*y[IDX_H2OII] -
        k[514]*y[IDX_H2OII];
    data[2559] = 0.0 + k[121]*y[IDX_H2OI];
    data[2560] = 0.0 - k[945]*y[IDX_H2OII] + k[960]*y[IDX_OHII];
    data[2561] = 0.0 + k[162]*y[IDX_H2OI] + k[933]*y[IDX_OHI];
    data[2562] = 0.0 - k[178]*y[IDX_H2OII] - k[979]*y[IDX_H2OII];
    data[2563] = 0.0 + k[121]*y[IDX_HII] + k[162]*y[IDX_H2II] +
        k[187]*y[IDX_COII] + k[188]*y[IDX_HCNII] + k[189]*y[IDX_N2II] +
        k[213]*y[IDX_HeII] + k[240]*y[IDX_NII] + k[261]*y[IDX_NHII] +
        k[312]*y[IDX_OII] + k[332]*y[IDX_OHII] - k[980]*y[IDX_H2OII] + k[2017];
    data[2564] = 0.0 - k[66]*y[IDX_CH2I] - k[86]*y[IDX_CHI] -
        k[175]*y[IDX_C2I] - k[176]*y[IDX_C2H2I] - k[177]*y[IDX_C2HI] -
        k[178]*y[IDX_H2COI] - k[179]*y[IDX_H2SI] - k[180]*y[IDX_HCOI] -
        k[181]*y[IDX_MgI] - k[182]*y[IDX_NOI] - k[183]*y[IDX_O2I] -
        k[184]*y[IDX_OCSI] - k[185]*y[IDX_SI] - k[186]*y[IDX_SiI] -
        k[274]*y[IDX_NH2I] - k[285]*y[IDX_NH3I] - k[512]*y[IDX_EM] -
        k[513]*y[IDX_EM] - k[514]*y[IDX_EM] - k[690]*y[IDX_CI] -
        k[758]*y[IDX_CH2I] - k[810]*y[IDX_CH4I] - k[843]*y[IDX_CHI] -
        k[945]*y[IDX_H2I] - k[976]*y[IDX_C2I] - k[977]*y[IDX_C2HI] -
        k[978]*y[IDX_COI] - k[979]*y[IDX_H2COI] - k[980]*y[IDX_H2OI] -
        k[981]*y[IDX_H2SI] - k[982]*y[IDX_H2SI] - k[983]*y[IDX_HCNI] -
        k[984]*y[IDX_HCOI] - k[985]*y[IDX_HCOI] - k[986]*y[IDX_HNCI] -
        k[987]*y[IDX_SI] - k[988]*y[IDX_SI] - k[989]*y[IDX_SO2I] -
        k[1333]*y[IDX_NI] - k[1334]*y[IDX_NI] - k[1400]*y[IDX_NH2I] -
        k[1425]*y[IDX_NH3I] - k[1450]*y[IDX_NHI] - k[1497]*y[IDX_OI] -
        k[1540]*y[IDX_OHI] - k[2016] - k[2239];
    data[2565] = 0.0 - k[179]*y[IDX_H2OII] - k[981]*y[IDX_H2OII] -
        k[982]*y[IDX_H2OII];
    data[2566] = 0.0 + k[1063]*y[IDX_OI] + k[1066]*y[IDX_OHI];
    data[2567] = 0.0 - k[983]*y[IDX_H2OII];
    data[2568] = 0.0 + k[188]*y[IDX_H2OI] + k[1541]*y[IDX_OHI];
    data[2569] = 0.0 - k[180]*y[IDX_H2OII] - k[984]*y[IDX_H2OII] -
        k[985]*y[IDX_H2OII] + k[1526]*y[IDX_OHII];
    data[2570] = 0.0 + k[1542]*y[IDX_OHI];
    data[2571] = 0.0 + k[213]*y[IDX_H2OI];
    data[2572] = 0.0 - k[986]*y[IDX_H2OII];
    data[2573] = 0.0 + k[1544]*y[IDX_OHI];
    data[2574] = 0.0 - k[181]*y[IDX_H2OII];
    data[2575] = 0.0 - k[1333]*y[IDX_H2OII] - k[1334]*y[IDX_H2OII];
    data[2576] = 0.0 + k[240]*y[IDX_H2OI];
    data[2577] = 0.0 + k[189]*y[IDX_H2OI];
    data[2578] = 0.0 + k[1545]*y[IDX_OHI];
    data[2579] = 0.0 - k[1450]*y[IDX_H2OII];
    data[2580] = 0.0 + k[261]*y[IDX_H2OI] + k[1371]*y[IDX_OHI];
    data[2581] = 0.0 - k[274]*y[IDX_H2OII] - k[1400]*y[IDX_H2OII];
    data[2582] = 0.0 - k[285]*y[IDX_H2OII] - k[1425]*y[IDX_H2OII];
    data[2583] = 0.0 - k[182]*y[IDX_H2OII];
    data[2584] = 0.0 + k[1063]*y[IDX_H3II] - k[1497]*y[IDX_H2OII];
    data[2585] = 0.0 + k[312]*y[IDX_H2OI];
    data[2586] = 0.0 - k[183]*y[IDX_H2OII];
    data[2587] = 0.0 + k[1547]*y[IDX_OHI];
    data[2588] = 0.0 - k[184]*y[IDX_H2OII];
    data[2589] = 0.0 + k[933]*y[IDX_H2II] + k[1066]*y[IDX_H3II] +
        k[1371]*y[IDX_NHII] + k[1532]*y[IDX_OHII] + k[1538]*y[IDX_CH5II] -
        k[1540]*y[IDX_H2OII] + k[1541]*y[IDX_HCNII] + k[1542]*y[IDX_HCOII] +
        k[1544]*y[IDX_HNOII] + k[1545]*y[IDX_N2HII] + k[1547]*y[IDX_O2HII];
    data[2590] = 0.0 + k[332]*y[IDX_H2OI] + k[960]*y[IDX_H2I] +
        k[1526]*y[IDX_HCOI] + k[1532]*y[IDX_OHI];
    data[2591] = 0.0 - k[185]*y[IDX_H2OII] - k[987]*y[IDX_H2OII] -
        k[988]*y[IDX_H2OII];
    data[2592] = 0.0 - k[186]*y[IDX_H2OII];
    data[2593] = 0.0 - k[989]*y[IDX_H2OII];
    data[2594] = 0.0 + k[2391] + k[2392] + k[2393] + k[2394];
    data[2595] = 0.0 - k[17]*y[IDX_H2SI] - k[622]*y[IDX_H2SI];
    data[2596] = 0.0 - k[43]*y[IDX_H2SI] - k[667]*y[IDX_H2SI];
    data[2597] = 0.0 - k[1018]*y[IDX_H2SI];
    data[2598] = 0.0 - k[721]*y[IDX_H2SI] - k[722]*y[IDX_H2SI];
    data[2599] = 0.0 - k[744]*y[IDX_H2SI] - k[745]*y[IDX_H2SI] -
        k[746]*y[IDX_H2SI];
    data[2600] = 0.0 - k[1653]*y[IDX_H2SI];
    data[2601] = 0.0 - k[778]*y[IDX_H2SI];
    data[2602] = 0.0 - k[77]*y[IDX_H2SI] - k[800]*y[IDX_H2SI];
    data[2603] = 0.0 - k[829]*y[IDX_H2SI];
    data[2604] = 0.0 - k[102]*y[IDX_H2SI] - k[872]*y[IDX_H2SI];
    data[2605] = 0.0 + k[532]*y[IDX_H3SII] + k[2138]*y[IDX_H2SII];
    data[2606] = 0.0 - k[1749]*y[IDX_H2SI];
    data[2607] = 0.0 - k[123]*y[IDX_H2SI] - k[894]*y[IDX_H2SI] -
        k[895]*y[IDX_H2SI];
    data[2608] = 0.0 + k[1727]*y[IDX_HSI];
    data[2609] = 0.0 - k[163]*y[IDX_H2SI] - k[923]*y[IDX_H2SI] -
        k[924]*y[IDX_H2SI];
    data[2610] = 0.0 + k[970]*y[IDX_H3SII];
    data[2611] = 0.0 + k[1009]*y[IDX_HSiSII];
    data[2612] = 0.0 - k[179]*y[IDX_H2SI] - k[981]*y[IDX_H2SI] -
        k[982]*y[IDX_H2SI];
    data[2613] = 0.0 - k[17]*y[IDX_CII] - k[43]*y[IDX_C2H2II] -
        k[77]*y[IDX_CH4II] - k[102]*y[IDX_COII] - k[123]*y[IDX_HII] -
        k[163]*y[IDX_H2II] - k[179]*y[IDX_H2OII] - k[190]*y[IDX_OCSII] -
        k[214]*y[IDX_HeII] - k[241]*y[IDX_NII] - k[253]*y[IDX_N2II] -
        k[266]*y[IDX_NH2II] - k[313]*y[IDX_OII] - k[322]*y[IDX_O2II] -
        k[333]*y[IDX_OHII] - k[403] - k[404] - k[622]*y[IDX_CII] -
        k[667]*y[IDX_C2H2II] - k[721]*y[IDX_CHII] - k[722]*y[IDX_CHII] -
        k[744]*y[IDX_CH2II] - k[745]*y[IDX_CH2II] - k[746]*y[IDX_CH2II] -
        k[778]*y[IDX_CH3II] - k[800]*y[IDX_CH4II] - k[829]*y[IDX_CH5II] -
        k[872]*y[IDX_COII] - k[894]*y[IDX_HII] - k[895]*y[IDX_HII] -
        k[923]*y[IDX_H2II] - k[924]*y[IDX_H2II] - k[981]*y[IDX_H2OII] -
        k[982]*y[IDX_H2OII] - k[1017]*y[IDX_H2SII] - k[1018]*y[IDX_C2NII] -
        k[1019]*y[IDX_HS2II] - k[1020]*y[IDX_HSiSII] - k[1021]*y[IDX_SOII] -
        k[1044]*y[IDX_H3II] - k[1079]*y[IDX_H3COII] - k[1087]*y[IDX_H3OII] -
        k[1134]*y[IDX_HCNHII] - k[1135]*y[IDX_HCNHII] - k[1143]*y[IDX_HCOII] -
        k[1175]*y[IDX_HSII] - k[1176]*y[IDX_HSII] - k[1226]*y[IDX_HeII] -
        k[1227]*y[IDX_HeII] - k[1300]*y[IDX_NII] - k[1301]*y[IDX_NII] -
        k[1302]*y[IDX_NII] - k[1315]*y[IDX_N2II] - k[1316]*y[IDX_N2II] -
        k[1381]*y[IDX_NH2II] - k[1382]*y[IDX_NH2II] - k[1383]*y[IDX_NH2II] -
        k[1384]*y[IDX_NH2II] - k[1415]*y[IDX_NH3II] - k[1472]*y[IDX_OII] -
        k[1473]*y[IDX_OII] - k[1524]*y[IDX_OHII] - k[1550]*y[IDX_SII] -
        k[1551]*y[IDX_SII] - k[1653]*y[IDX_CH3I] - k[1749]*y[IDX_HI] -
        k[1888]*y[IDX_OI] - k[1938]*y[IDX_OHI] - k[2020] - k[2021] - k[2022] -
        k[2257];
    data[2614] = 0.0 + k[203]*y[IDX_HCOI] + k[223]*y[IDX_MgI] +
        k[286]*y[IDX_NH3I] + k[299]*y[IDX_NOI] + k[346]*y[IDX_SI] +
        k[350]*y[IDX_SiI] - k[1017]*y[IDX_H2SI] + k[2138]*y[IDX_EM];
    data[2615] = 0.0 - k[1044]*y[IDX_H2SI];
    data[2616] = 0.0 - k[1079]*y[IDX_H2SI];
    data[2617] = 0.0 - k[1087]*y[IDX_H2SI];
    data[2618] = 0.0 + k[532]*y[IDX_EM] + k[970]*y[IDX_H2COI] +
        k[1123]*y[IDX_HCNI] + k[1167]*y[IDX_HNCI] + k[1429]*y[IDX_NH3I];
    data[2619] = 0.0 + k[1123]*y[IDX_H3SII];
    data[2620] = 0.0 - k[1134]*y[IDX_H2SI] - k[1135]*y[IDX_H2SI];
    data[2621] = 0.0 + k[203]*y[IDX_H2SII];
    data[2622] = 0.0 - k[1143]*y[IDX_H2SI];
    data[2623] = 0.0 - k[214]*y[IDX_H2SI] - k[1226]*y[IDX_H2SI] -
        k[1227]*y[IDX_H2SI];
    data[2624] = 0.0 + k[1167]*y[IDX_H3SII];
    data[2625] = 0.0 + k[1727]*y[IDX_H2I] + k[1789]*y[IDX_HSI] +
        k[1789]*y[IDX_HSI];
    data[2626] = 0.0 - k[1175]*y[IDX_H2SI] - k[1176]*y[IDX_H2SI];
    data[2627] = 0.0 - k[1019]*y[IDX_H2SI];
    data[2628] = 0.0 + k[1009]*y[IDX_H2OI] - k[1020]*y[IDX_H2SI];
    data[2629] = 0.0 + k[223]*y[IDX_H2SII];
    data[2630] = 0.0 - k[241]*y[IDX_H2SI] - k[1300]*y[IDX_H2SI] -
        k[1301]*y[IDX_H2SI] - k[1302]*y[IDX_H2SI];
    data[2631] = 0.0 - k[253]*y[IDX_H2SI] - k[1315]*y[IDX_H2SI] -
        k[1316]*y[IDX_H2SI];
    data[2632] = 0.0 - k[266]*y[IDX_H2SI] - k[1381]*y[IDX_H2SI] -
        k[1382]*y[IDX_H2SI] - k[1383]*y[IDX_H2SI] - k[1384]*y[IDX_H2SI];
    data[2633] = 0.0 + k[286]*y[IDX_H2SII] + k[1429]*y[IDX_H3SII];
    data[2634] = 0.0 - k[1415]*y[IDX_H2SI];
    data[2635] = 0.0 + k[299]*y[IDX_H2SII];
    data[2636] = 0.0 - k[1888]*y[IDX_H2SI];
    data[2637] = 0.0 - k[313]*y[IDX_H2SI] - k[1472]*y[IDX_H2SI] -
        k[1473]*y[IDX_H2SI];
    data[2638] = 0.0 - k[322]*y[IDX_H2SI];
    data[2639] = 0.0 - k[190]*y[IDX_H2SI];
    data[2640] = 0.0 - k[1938]*y[IDX_H2SI];
    data[2641] = 0.0 - k[333]*y[IDX_H2SI] - k[1524]*y[IDX_H2SI];
    data[2642] = 0.0 + k[346]*y[IDX_H2SII];
    data[2643] = 0.0 - k[1550]*y[IDX_H2SI] - k[1551]*y[IDX_H2SI];
    data[2644] = 0.0 + k[350]*y[IDX_H2SII];
    data[2645] = 0.0 - k[1021]*y[IDX_H2SI];
    data[2646] = 0.0 - k[691]*y[IDX_H2SII];
    data[2647] = 0.0 + k[17]*y[IDX_H2SI];
    data[2648] = 0.0 + k[43]*y[IDX_H2SI];
    data[2649] = 0.0 + k[77]*y[IDX_H2SI];
    data[2650] = 0.0 + k[102]*y[IDX_H2SI];
    data[2651] = 0.0 - k[515]*y[IDX_H2SII] - k[516]*y[IDX_H2SII] -
        k[2138]*y[IDX_H2SII];
    data[2652] = 0.0 - k[1103]*y[IDX_H2SII] + k[1104]*y[IDX_H3SII];
    data[2653] = 0.0 + k[123]*y[IDX_H2SI];
    data[2654] = 0.0 - k[946]*y[IDX_H2SII] + k[949]*y[IDX_HSII] +
        k[2117]*y[IDX_SII];
    data[2655] = 0.0 + k[163]*y[IDX_H2SI];
    data[2656] = 0.0 + k[974]*y[IDX_SII];
    data[2657] = 0.0 - k[1000]*y[IDX_H2SII];
    data[2658] = 0.0 + k[179]*y[IDX_H2SI];
    data[2659] = 0.0 + k[17]*y[IDX_CII] + k[43]*y[IDX_C2H2II] +
        k[77]*y[IDX_CH4II] + k[102]*y[IDX_COII] + k[123]*y[IDX_HII] +
        k[163]*y[IDX_H2II] + k[179]*y[IDX_H2OII] + k[190]*y[IDX_OCSII] +
        k[214]*y[IDX_HeII] + k[241]*y[IDX_NII] + k[253]*y[IDX_N2II] +
        k[266]*y[IDX_NH2II] + k[313]*y[IDX_OII] + k[322]*y[IDX_O2II] +
        k[333]*y[IDX_OHII] + k[403] - k[1017]*y[IDX_H2SII] + k[2020];
    data[2660] = 0.0 - k[203]*y[IDX_HCOI] - k[223]*y[IDX_MgI] -
        k[286]*y[IDX_NH3I] - k[299]*y[IDX_NOI] - k[346]*y[IDX_SI] -
        k[350]*y[IDX_SiI] - k[515]*y[IDX_EM] - k[516]*y[IDX_EM] -
        k[691]*y[IDX_CI] - k[946]*y[IDX_H2I] - k[1000]*y[IDX_H2OI] -
        k[1017]*y[IDX_H2SI] - k[1103]*y[IDX_HI] - k[1335]*y[IDX_NI] -
        k[1426]*y[IDX_NH3I] - k[1498]*y[IDX_OI] - k[1499]*y[IDX_OI] -
        k[2138]*y[IDX_EM] - k[2293];
    data[2661] = 0.0 + k[1053]*y[IDX_HSI];
    data[2662] = 0.0 + k[1104]*y[IDX_HI];
    data[2663] = 0.0 - k[203]*y[IDX_H2SII];
    data[2664] = 0.0 + k[1147]*y[IDX_HSI];
    data[2665] = 0.0 + k[214]*y[IDX_H2SI];
    data[2666] = 0.0 + k[1053]*y[IDX_H3II] + k[1147]*y[IDX_HCOII];
    data[2667] = 0.0 + k[949]*y[IDX_H2I];
    data[2668] = 0.0 - k[223]*y[IDX_H2SII];
    data[2669] = 0.0 - k[1335]*y[IDX_H2SII];
    data[2670] = 0.0 + k[241]*y[IDX_H2SI];
    data[2671] = 0.0 + k[253]*y[IDX_H2SI];
    data[2672] = 0.0 + k[266]*y[IDX_H2SI];
    data[2673] = 0.0 - k[286]*y[IDX_H2SII] - k[1426]*y[IDX_H2SII];
    data[2674] = 0.0 - k[299]*y[IDX_H2SII];
    data[2675] = 0.0 - k[1498]*y[IDX_H2SII] - k[1499]*y[IDX_H2SII];
    data[2676] = 0.0 + k[313]*y[IDX_H2SI];
    data[2677] = 0.0 + k[322]*y[IDX_H2SI];
    data[2678] = 0.0 + k[190]*y[IDX_H2SI];
    data[2679] = 0.0 + k[333]*y[IDX_H2SI];
    data[2680] = 0.0 - k[346]*y[IDX_H2SII];
    data[2681] = 0.0 + k[974]*y[IDX_H2COI] + k[2117]*y[IDX_H2I];
    data[2682] = 0.0 - k[350]*y[IDX_H2SII];
    data[2683] = 0.0 + k[2503] + k[2504] + k[2505] + k[2506];
    data[2684] = 0.0 - k[122]*y[IDX_H2S2I];
    data[2685] = 0.0 - k[122]*y[IDX_HII] - k[402] - k[1224]*y[IDX_HeII] -
        k[1225]*y[IDX_HeII] - k[2019] - k[2203];
    data[2686] = 0.0 - k[1224]*y[IDX_H2S2I] - k[1225]*y[IDX_H2S2I];
    data[2687] = 0.0 + k[793]*y[IDX_S2II];
    data[2688] = 0.0 - k[517]*y[IDX_H2S2II] - k[518]*y[IDX_H2S2II];
    data[2689] = 0.0 + k[122]*y[IDX_H2S2I];
    data[2690] = 0.0 + k[122]*y[IDX_HII];
    data[2691] = 0.0 - k[517]*y[IDX_EM] - k[518]*y[IDX_EM] - k[2280];
    data[2692] = 0.0 + k[1052]*y[IDX_HS2I];
    data[2693] = 0.0 + k[1091]*y[IDX_HS2I];
    data[2694] = 0.0 + k[1553]*y[IDX_SI];
    data[2695] = 0.0 + k[1146]*y[IDX_HS2I];
    data[2696] = 0.0 + k[1052]*y[IDX_H3II] + k[1091]*y[IDX_H3OII] +
        k[1146]*y[IDX_HCOII];
    data[2697] = 0.0 + k[1553]*y[IDX_H3SII];
    data[2698] = 0.0 + k[793]*y[IDX_CH3OHI];
    data[2699] = 0.0 + k[2455] + k[2456] + k[2457] + k[2458];
    data[2700] = 0.0 - k[896]*y[IDX_H2SiOI];
    data[2701] = 0.0 - k[405] - k[896]*y[IDX_HII] - k[1228]*y[IDX_HeII] -
        k[2023] - k[2024] - k[2192];
    data[2702] = 0.0 - k[1228]*y[IDX_H2SiOI];
    data[2703] = 0.0 + k[1924]*y[IDX_SiH3I];
    data[2704] = 0.0 + k[1924]*y[IDX_OI];
    data[2705] = 0.0 - k[1027]*y[IDX_H3II];
    data[2706] = 0.0 - k[1022]*y[IDX_H3II];
    data[2707] = 0.0 - k[1025]*y[IDX_H3II];
    data[2708] = 0.0 - k[1023]*y[IDX_H3II] - k[1024]*y[IDX_H3II];
    data[2709] = 0.0 - k[1026]*y[IDX_H3II];
    data[2710] = 0.0 - k[1034]*y[IDX_H3II];
    data[2711] = 0.0 - k[1028]*y[IDX_H3II];
    data[2712] = 0.0 - k[1029]*y[IDX_H3II];
    data[2713] = 0.0 - k[1030]*y[IDX_H3II];
    data[2714] = 0.0 - k[1031]*y[IDX_H3II] - k[1032]*y[IDX_H3II];
    data[2715] = 0.0 - k[1033]*y[IDX_H3II];
    data[2716] = 0.0 - k[1040]*y[IDX_H3II];
    data[2717] = 0.0 - k[1035]*y[IDX_H3II];
    data[2718] = 0.0 - k[1037]*y[IDX_H3II] - k[1038]*y[IDX_H3II];
    data[2719] = 0.0 - k[1036]*y[IDX_H3II];
    data[2720] = 0.0 - k[1039]*y[IDX_H3II];
    data[2721] = 0.0 - k[519]*y[IDX_H3II] - k[520]*y[IDX_H3II];
    data[2722] = 0.0 + k[920]*y[IDX_H2II] + k[951]*y[IDX_HeHII] +
        k[954]*y[IDX_NHII] + k[959]*y[IDX_O2HII];
    data[2723] = 0.0 + k[920]*y[IDX_H2I] + k[925]*y[IDX_HCOI];
    data[2724] = 0.0 - k[1041]*y[IDX_H3II];
    data[2725] = 0.0 - k[1042]*y[IDX_H3II];
    data[2726] = 0.0 - k[1043]*y[IDX_H3II];
    data[2727] = 0.0 - k[1044]*y[IDX_H3II];
    data[2728] = 0.0 - k[519]*y[IDX_EM] - k[520]*y[IDX_EM] -
        k[1022]*y[IDX_C2I] - k[1023]*y[IDX_C2H5OHI] - k[1024]*y[IDX_C2H5OHI] -
        k[1025]*y[IDX_C2HI] - k[1026]*y[IDX_C2NI] - k[1027]*y[IDX_CI] -
        k[1028]*y[IDX_CH2I] - k[1029]*y[IDX_CH3I] - k[1030]*y[IDX_CH3CNI] -
        k[1031]*y[IDX_CH3OHI] - k[1032]*y[IDX_CH3OHI] - k[1033]*y[IDX_CH4I] -
        k[1034]*y[IDX_CHI] - k[1035]*y[IDX_CNI] - k[1036]*y[IDX_CO2I] -
        k[1037]*y[IDX_COI] - k[1038]*y[IDX_COI] - k[1039]*y[IDX_CSI] -
        k[1040]*y[IDX_ClI] - k[1041]*y[IDX_H2COI] - k[1042]*y[IDX_H2CSI] -
        k[1043]*y[IDX_H2OI] - k[1044]*y[IDX_H2SI] - k[1045]*y[IDX_HCNI] -
        k[1046]*y[IDX_HCOI] - k[1047]*y[IDX_HCOOCH3I] - k[1048]*y[IDX_HCSI] -
        k[1049]*y[IDX_HClI] - k[1050]*y[IDX_HNCI] - k[1051]*y[IDX_HNOI] -
        k[1052]*y[IDX_HS2I] - k[1053]*y[IDX_HSI] - k[1054]*y[IDX_MgI] -
        k[1055]*y[IDX_N2I] - k[1056]*y[IDX_NH2I] - k[1057]*y[IDX_NH3I] -
        k[1058]*y[IDX_NHI] - k[1059]*y[IDX_NO2I] - k[1060]*y[IDX_NOI] -
        k[1061]*y[IDX_NSI] - k[1062]*y[IDX_O2I] - k[1063]*y[IDX_OI] -
        k[1064]*y[IDX_OI] - k[1065]*y[IDX_OCSI] - k[1066]*y[IDX_OHI] -
        k[1067]*y[IDX_S2I] - k[1068]*y[IDX_SI] - k[1069]*y[IDX_SO2I] -
        k[1070]*y[IDX_SOI] - k[1071]*y[IDX_SiI] - k[1072]*y[IDX_SiH2I] -
        k[1073]*y[IDX_SiH3I] - k[1074]*y[IDX_SiH4I] - k[1075]*y[IDX_SiHI] -
        k[1076]*y[IDX_SiOI] - k[1077]*y[IDX_SiSI] - k[2025] - k[2026];
    data[2729] = 0.0 - k[1049]*y[IDX_H3II];
    data[2730] = 0.0 - k[1045]*y[IDX_H3II];
    data[2731] = 0.0 + k[925]*y[IDX_H2II] - k[1046]*y[IDX_H3II];
    data[2732] = 0.0 - k[1047]*y[IDX_H3II];
    data[2733] = 0.0 - k[1048]*y[IDX_H3II];
    data[2734] = 0.0 + k[951]*y[IDX_H2I];
    data[2735] = 0.0 - k[1050]*y[IDX_H3II];
    data[2736] = 0.0 - k[1051]*y[IDX_H3II];
    data[2737] = 0.0 - k[1053]*y[IDX_H3II];
    data[2738] = 0.0 - k[1052]*y[IDX_H3II];
    data[2739] = 0.0 - k[1054]*y[IDX_H3II];
    data[2740] = 0.0 - k[1055]*y[IDX_H3II];
    data[2741] = 0.0 - k[1058]*y[IDX_H3II];
    data[2742] = 0.0 + k[954]*y[IDX_H2I];
    data[2743] = 0.0 - k[1056]*y[IDX_H3II];
    data[2744] = 0.0 - k[1057]*y[IDX_H3II];
    data[2745] = 0.0 - k[1060]*y[IDX_H3II];
    data[2746] = 0.0 - k[1059]*y[IDX_H3II];
    data[2747] = 0.0 - k[1061]*y[IDX_H3II];
    data[2748] = 0.0 - k[1063]*y[IDX_H3II] - k[1064]*y[IDX_H3II];
    data[2749] = 0.0 - k[1062]*y[IDX_H3II];
    data[2750] = 0.0 + k[959]*y[IDX_H2I];
    data[2751] = 0.0 - k[1065]*y[IDX_H3II];
    data[2752] = 0.0 - k[1066]*y[IDX_H3II];
    data[2753] = 0.0 - k[1068]*y[IDX_H3II];
    data[2754] = 0.0 - k[1067]*y[IDX_H3II];
    data[2755] = 0.0 - k[1071]*y[IDX_H3II];
    data[2756] = 0.0 - k[1075]*y[IDX_H3II];
    data[2757] = 0.0 - k[1072]*y[IDX_H3II];
    data[2758] = 0.0 - k[1073]*y[IDX_H3II];
    data[2759] = 0.0 - k[1074]*y[IDX_H3II];
    data[2760] = 0.0 - k[1076]*y[IDX_H3II];
    data[2761] = 0.0 - k[1077]*y[IDX_H3II];
    data[2762] = 0.0 - k[1070]*y[IDX_H3II];
    data[2763] = 0.0 - k[1069]*y[IDX_H3II];
    data[2764] = 0.0 + k[606]*y[IDX_C2H5OHI] + k[612]*y[IDX_CH3OHI];
    data[2765] = 0.0 + k[660]*y[IDX_H2COI];
    data[2766] = 0.0 + k[1560]*y[IDX_SOII];
    data[2767] = 0.0 + k[606]*y[IDX_CII] + k[1024]*y[IDX_H3II];
    data[2768] = 0.0 - k[844]*y[IDX_H3COII];
    data[2769] = 0.0 + k[710]*y[IDX_CH3OHI] + k[716]*y[IDX_H2COI];
    data[2770] = 0.0 + k[743]*y[IDX_H2OI];
    data[2771] = 0.0 + k[776]*y[IDX_CH3OHI] + k[782]*y[IDX_O2I];
    data[2772] = 0.0 + k[612]*y[IDX_CII] + k[710]*y[IDX_CHII] +
        k[776]*y[IDX_CH3II] + k[888]*y[IDX_HII] - k[1078]*y[IDX_H3COII] +
        k[1290]*y[IDX_NII] + k[1467]*y[IDX_OII] + k[1483]*y[IDX_O2II] + k[1991];
    data[2773] = 0.0 + k[809]*y[IDX_H2COII];
    data[2774] = 0.0 + k[798]*y[IDX_H2COI];
    data[2775] = 0.0 + k[827]*y[IDX_H2COI] + k[1494]*y[IDX_OI];
    data[2776] = 0.0 - k[521]*y[IDX_H3COII] - k[522]*y[IDX_H3COII] -
        k[523]*y[IDX_H3COII] - k[524]*y[IDX_H3COII] - k[525]*y[IDX_H3COII];
    data[2777] = 0.0 + k[888]*y[IDX_CH3OHI];
    data[2778] = 0.0 + k[660]*y[IDX_C2HII] + k[716]*y[IDX_CHII] +
        k[798]*y[IDX_CH4II] + k[827]*y[IDX_CH5II] + k[966]*y[IDX_H2COII] +
        k[970]*y[IDX_H3SII] + k[971]*y[IDX_HNOII] + k[973]*y[IDX_O2HII] +
        k[979]*y[IDX_H2OII] + k[1041]*y[IDX_H3II] + k[1086]*y[IDX_H3OII] +
        k[1112]*y[IDX_HCNII] + k[1132]*y[IDX_HCNHII] + k[1133]*y[IDX_HCNHII] +
        k[1141]*y[IDX_HCOII] + k[1323]*y[IDX_N2HII] + k[1354]*y[IDX_NHII] +
        k[1376]*y[IDX_NH2II] + k[1522]*y[IDX_OHII];
    data[2779] = 0.0 + k[809]*y[IDX_CH4I] + k[966]*y[IDX_H2COI] +
        k[1158]*y[IDX_HCOI] + k[1449]*y[IDX_NHI];
    data[2780] = 0.0 + k[743]*y[IDX_CH2II] - k[1001]*y[IDX_H3COII];
    data[2781] = 0.0 + k[979]*y[IDX_H2COI];
    data[2782] = 0.0 - k[1079]*y[IDX_H3COII];
    data[2783] = 0.0 + k[1024]*y[IDX_C2H5OHI] + k[1041]*y[IDX_H2COI];
    data[2784] = 0.0 - k[521]*y[IDX_EM] - k[522]*y[IDX_EM] -
        k[523]*y[IDX_EM] - k[524]*y[IDX_EM] - k[525]*y[IDX_EM] -
        k[844]*y[IDX_CHI] - k[1001]*y[IDX_H2OI] - k[1078]*y[IDX_CH3OHI] -
        k[1079]*y[IDX_H2SI] - k[1122]*y[IDX_HCNI] - k[1166]*y[IDX_HNCI] -
        k[1401]*y[IDX_NH2I] - k[1427]*y[IDX_NH3I] - k[2202];
    data[2785] = 0.0 + k[1086]*y[IDX_H2COI];
    data[2786] = 0.0 + k[970]*y[IDX_H2COI];
    data[2787] = 0.0 - k[1122]*y[IDX_H3COII];
    data[2788] = 0.0 + k[1112]*y[IDX_H2COI];
    data[2789] = 0.0 + k[1132]*y[IDX_H2COI] + k[1133]*y[IDX_H2COI];
    data[2790] = 0.0 + k[1158]*y[IDX_H2COII];
    data[2791] = 0.0 + k[1141]*y[IDX_H2COI];
    data[2792] = 0.0 - k[1166]*y[IDX_H3COII];
    data[2793] = 0.0 + k[971]*y[IDX_H2COI];
    data[2794] = 0.0 + k[1290]*y[IDX_CH3OHI];
    data[2795] = 0.0 + k[1323]*y[IDX_H2COI];
    data[2796] = 0.0 + k[1449]*y[IDX_H2COII];
    data[2797] = 0.0 + k[1354]*y[IDX_H2COI];
    data[2798] = 0.0 - k[1401]*y[IDX_H3COII];
    data[2799] = 0.0 + k[1376]*y[IDX_H2COI];
    data[2800] = 0.0 - k[1427]*y[IDX_H3COII];
    data[2801] = 0.0 + k[1494]*y[IDX_CH5II];
    data[2802] = 0.0 + k[1467]*y[IDX_CH3OHI];
    data[2803] = 0.0 + k[782]*y[IDX_CH3II];
    data[2804] = 0.0 + k[1483]*y[IDX_CH3OHI];
    data[2805] = 0.0 + k[973]*y[IDX_H2COI];
    data[2806] = 0.0 + k[1522]*y[IDX_H2COI];
    data[2807] = 0.0 + k[1560]*y[IDX_C2H4I];
    data[2808] = 0.0 + k[1561]*y[IDX_SOII];
    data[2809] = 0.0 + k[744]*y[IDX_H2SI];
    data[2810] = 0.0 + k[778]*y[IDX_H2SI] + k[785]*y[IDX_OCSI];
    data[2811] = 0.0 + k[814]*y[IDX_HSII] + k[821]*y[IDX_SII];
    data[2812] = 0.0 - k[526]*y[IDX_H3CSII] - k[527]*y[IDX_H3CSII];
    data[2813] = 0.0 + k[1042]*y[IDX_H3II] + k[1142]*y[IDX_HCOII];
    data[2814] = 0.0 + k[744]*y[IDX_CH2II] + k[778]*y[IDX_CH3II];
    data[2815] = 0.0 + k[1042]*y[IDX_H2CSI];
    data[2816] = 0.0 - k[526]*y[IDX_EM] - k[527]*y[IDX_EM] - k[2282];
    data[2817] = 0.0 + k[1142]*y[IDX_H2CSI];
    data[2818] = 0.0 + k[814]*y[IDX_CH4I];
    data[2819] = 0.0 + k[785]*y[IDX_CH3II];
    data[2820] = 0.0 + k[821]*y[IDX_CH4I];
    data[2821] = 0.0 + k[1561]*y[IDX_C2H4I];
    data[2822] = 0.0 - k[692]*y[IDX_H3OII];
    data[2823] = 0.0 - k[1080]*y[IDX_H3OII];
    data[2824] = 0.0 + k[991]*y[IDX_H2OI];
    data[2825] = 0.0 - k[2121]*y[IDX_H3OII];
    data[2826] = 0.0 - k[1081]*y[IDX_H3OII];
    data[2827] = 0.0 - k[845]*y[IDX_H3OII];
    data[2828] = 0.0 + k[719]*y[IDX_H2OI];
    data[2829] = 0.0 - k[759]*y[IDX_H3OII];
    data[2830] = 0.0 - k[1082]*y[IDX_H3OII];
    data[2831] = 0.0 - k[1083]*y[IDX_H3OII];
    data[2832] = 0.0 - k[1084]*y[IDX_H3OII];
    data[2833] = 0.0 + k[810]*y[IDX_H2OII] + k[820]*y[IDX_OHII];
    data[2834] = 0.0 + k[799]*y[IDX_H2OI];
    data[2835] = 0.0 + k[828]*y[IDX_H2OI] + k[1495]*y[IDX_OI];
    data[2836] = 0.0 - k[1085]*y[IDX_H3OII];
    data[2837] = 0.0 - k[528]*y[IDX_H3OII] - k[529]*y[IDX_H3OII] -
        k[530]*y[IDX_H3OII] - k[531]*y[IDX_H3OII];
    data[2838] = 0.0 + k[945]*y[IDX_H2OII];
    data[2839] = 0.0 + k[922]*y[IDX_H2OI];
    data[2840] = 0.0 + k[999]*y[IDX_H2OI];
    data[2841] = 0.0 - k[1086]*y[IDX_H3OII];
    data[2842] = 0.0 + k[998]*y[IDX_H2OI];
    data[2843] = 0.0 + k[719]*y[IDX_CHII] + k[799]*y[IDX_CH4II] +
        k[828]*y[IDX_CH5II] + k[922]*y[IDX_H2II] + k[980]*y[IDX_H2OII] +
        k[991]*y[IDX_C2H2II] + k[998]*y[IDX_H2COII] + k[999]*y[IDX_H2ClII] +
        k[1000]*y[IDX_H2SII] + k[1001]*y[IDX_H3COII] + k[1002]*y[IDX_HCNII] +
        k[1003]*y[IDX_HCOII] + k[1004]*y[IDX_HCO2II] + k[1005]*y[IDX_HNOII] +
        k[1006]*y[IDX_HOCSII] + k[1007]*y[IDX_HSII] + k[1008]*y[IDX_HSO2II] +
        k[1011]*y[IDX_N2HII] + k[1012]*y[IDX_O2HII] + k[1014]*y[IDX_SiHII] +
        k[1015]*y[IDX_SiH4II] + k[1016]*y[IDX_SiH5II] + k[1043]*y[IDX_H3II] +
        k[1356]*y[IDX_NHII] + k[1378]*y[IDX_NH2II] + k[1523]*y[IDX_OHII];
    data[2844] = 0.0 + k[810]*y[IDX_CH4I] + k[945]*y[IDX_H2I] +
        k[980]*y[IDX_H2OI] + k[982]*y[IDX_H2SI] + k[984]*y[IDX_HCOI] +
        k[1450]*y[IDX_NHI] + k[1540]*y[IDX_OHI];
    data[2845] = 0.0 + k[982]*y[IDX_H2OII] - k[1087]*y[IDX_H3OII];
    data[2846] = 0.0 + k[1000]*y[IDX_H2OI];
    data[2847] = 0.0 + k[1043]*y[IDX_H2OI];
    data[2848] = 0.0 + k[1001]*y[IDX_H2OI];
    data[2849] = 0.0 - k[528]*y[IDX_EM] - k[529]*y[IDX_EM] -
        k[530]*y[IDX_EM] - k[531]*y[IDX_EM] - k[692]*y[IDX_CI] -
        k[759]*y[IDX_CH2I] - k[845]*y[IDX_CHI] - k[1080]*y[IDX_C2I] -
        k[1081]*y[IDX_C2H5OHI] - k[1082]*y[IDX_CH3CCHI] - k[1083]*y[IDX_CH3CNI]
        - k[1084]*y[IDX_CH3OHI] - k[1085]*y[IDX_CSI] - k[1086]*y[IDX_H2COI] -
        k[1087]*y[IDX_H2SI] - k[1088]*y[IDX_HCNI] - k[1089]*y[IDX_HCOOCH3I] -
        k[1090]*y[IDX_HNCI] - k[1091]*y[IDX_HS2I] - k[1092]*y[IDX_S2I] -
        k[1093]*y[IDX_SiI] - k[1094]*y[IDX_SiH2I] - k[1095]*y[IDX_SiHI] -
        k[1096]*y[IDX_SiOI] - k[1402]*y[IDX_NH2I] - k[1428]*y[IDX_NH3I] -
        k[2121]*y[IDX_C2H4I] - k[2246];
    data[2850] = 0.0 - k[1088]*y[IDX_H3OII];
    data[2851] = 0.0 + k[1002]*y[IDX_H2OI];
    data[2852] = 0.0 + k[984]*y[IDX_H2OII];
    data[2853] = 0.0 + k[1003]*y[IDX_H2OI];
    data[2854] = 0.0 + k[1004]*y[IDX_H2OI];
    data[2855] = 0.0 - k[1089]*y[IDX_H3OII];
    data[2856] = 0.0 - k[1090]*y[IDX_H3OII];
    data[2857] = 0.0 + k[1005]*y[IDX_H2OI];
    data[2858] = 0.0 + k[1006]*y[IDX_H2OI];
    data[2859] = 0.0 + k[1007]*y[IDX_H2OI];
    data[2860] = 0.0 - k[1091]*y[IDX_H3OII];
    data[2861] = 0.0 + k[1008]*y[IDX_H2OI];
    data[2862] = 0.0 + k[1011]*y[IDX_H2OI];
    data[2863] = 0.0 + k[1450]*y[IDX_H2OII];
    data[2864] = 0.0 + k[1356]*y[IDX_H2OI];
    data[2865] = 0.0 - k[1402]*y[IDX_H3OII];
    data[2866] = 0.0 + k[1378]*y[IDX_H2OI];
    data[2867] = 0.0 - k[1428]*y[IDX_H3OII];
    data[2868] = 0.0 + k[1495]*y[IDX_CH5II];
    data[2869] = 0.0 + k[1012]*y[IDX_H2OI];
    data[2870] = 0.0 + k[1540]*y[IDX_H2OII];
    data[2871] = 0.0 + k[820]*y[IDX_CH4I] + k[1523]*y[IDX_H2OI];
    data[2872] = 0.0 - k[1092]*y[IDX_H3OII];
    data[2873] = 0.0 - k[1093]*y[IDX_H3OII];
    data[2874] = 0.0 - k[1095]*y[IDX_H3OII];
    data[2875] = 0.0 + k[1014]*y[IDX_H2OI];
    data[2876] = 0.0 - k[1094]*y[IDX_H3OII];
    data[2877] = 0.0 + k[1015]*y[IDX_H2OI];
    data[2878] = 0.0 + k[1016]*y[IDX_H2OI];
    data[2879] = 0.0 - k[1096]*y[IDX_H3OII];
    data[2880] = 0.0 + k[667]*y[IDX_H2SI];
    data[2881] = 0.0 + k[721]*y[IDX_H2SI];
    data[2882] = 0.0 + k[745]*y[IDX_H2SI];
    data[2883] = 0.0 + k[800]*y[IDX_H2SI];
    data[2884] = 0.0 + k[829]*y[IDX_H2SI];
    data[2885] = 0.0 - k[532]*y[IDX_H3SII] - k[533]*y[IDX_H3SII] -
        k[534]*y[IDX_H3SII] - k[535]*y[IDX_H3SII];
    data[2886] = 0.0 - k[1104]*y[IDX_H3SII];
    data[2887] = 0.0 + k[946]*y[IDX_H2SII] + k[2116]*y[IDX_HSII];
    data[2888] = 0.0 - k[970]*y[IDX_H3SII];
    data[2889] = 0.0 + k[981]*y[IDX_H2SI];
    data[2890] = 0.0 + k[667]*y[IDX_C2H2II] + k[721]*y[IDX_CHII] +
        k[745]*y[IDX_CH2II] + k[800]*y[IDX_CH4II] + k[829]*y[IDX_CH5II] +
        k[981]*y[IDX_H2OII] + k[1017]*y[IDX_H2SII] + k[1019]*y[IDX_HS2II] +
        k[1020]*y[IDX_HSiSII] + k[1044]*y[IDX_H3II] + k[1079]*y[IDX_H3COII] +
        k[1087]*y[IDX_H3OII] + k[1134]*y[IDX_HCNHII] + k[1135]*y[IDX_HCNHII] +
        k[1143]*y[IDX_HCOII] + k[1175]*y[IDX_HSII] + k[1381]*y[IDX_NH2II] +
        k[1524]*y[IDX_OHII];
    data[2891] = 0.0 + k[946]*y[IDX_H2I] + k[1017]*y[IDX_H2SI];
    data[2892] = 0.0 + k[1044]*y[IDX_H2SI];
    data[2893] = 0.0 + k[1079]*y[IDX_H2SI];
    data[2894] = 0.0 + k[1087]*y[IDX_H2SI];
    data[2895] = 0.0 - k[532]*y[IDX_EM] - k[533]*y[IDX_EM] -
        k[534]*y[IDX_EM] - k[535]*y[IDX_EM] - k[970]*y[IDX_H2COI] -
        k[1104]*y[IDX_HI] - k[1123]*y[IDX_HCNI] - k[1167]*y[IDX_HNCI] -
        k[1429]*y[IDX_NH3I] - k[1553]*y[IDX_SI] - k[2278];
    data[2896] = 0.0 - k[1123]*y[IDX_H3SII];
    data[2897] = 0.0 + k[1134]*y[IDX_H2SI] + k[1135]*y[IDX_H2SI];
    data[2898] = 0.0 + k[1143]*y[IDX_H2SI];
    data[2899] = 0.0 - k[1167]*y[IDX_H3SII];
    data[2900] = 0.0 + k[1175]*y[IDX_H2SI] + k[2116]*y[IDX_H2I];
    data[2901] = 0.0 + k[1019]*y[IDX_H2SI];
    data[2902] = 0.0 + k[1020]*y[IDX_H2SI];
    data[2903] = 0.0 + k[1381]*y[IDX_H2SI];
    data[2904] = 0.0 - k[1429]*y[IDX_H3SII];
    data[2905] = 0.0 + k[1524]*y[IDX_H2SI];
    data[2906] = 0.0 - k[1553]*y[IDX_H3SII];
    data[2907] = 0.0 + k[969]*y[IDX_H2COI];
    data[2908] = 0.0 - k[536]*y[IDX_H5C2O2II] - k[537]*y[IDX_H5C2O2II];
    data[2909] = 0.0 + k[969]*y[IDX_CH3OH2II];
    data[2910] = 0.0 + k[1047]*y[IDX_HCOOCH3I];
    data[2911] = 0.0 + k[1089]*y[IDX_HCOOCH3I];
    data[2912] = 0.0 - k[536]*y[IDX_EM] - k[537]*y[IDX_EM] - k[2167];
    data[2913] = 0.0 + k[1145]*y[IDX_HCOOCH3I];
    data[2914] = 0.0 + k[1047]*y[IDX_H3II] + k[1089]*y[IDX_H3OII] +
        k[1145]*y[IDX_HCOII];
    data[2915] = 0.0 + k[2467] + k[2468] + k[2469] + k[2470];
    data[2916] = 0.0 - k[623]*y[IDX_HC3NI] - k[624]*y[IDX_HC3NI] -
        k[625]*y[IDX_HC3NI];
    data[2917] = 0.0 + k[1578]*y[IDX_HCNI] + k[1579]*y[IDX_HNCI] +
        k[1580]*y[IDX_NCCNI] + k[2100]*y[IDX_CNI];
    data[2918] = 0.0 + k[1701]*y[IDX_CNI];
    data[2919] = 0.0 + k[1797]*y[IDX_NI];
    data[2920] = 0.0 + k[994]*y[IDX_H2OI];
    data[2921] = 0.0 + k[1701]*y[IDX_C2H2I] + k[2100]*y[IDX_C2HI];
    data[2922] = 0.0 + k[994]*y[IDX_C4NII];
    data[2923] = 0.0 - k[407] - k[623]*y[IDX_CII] - k[624]*y[IDX_CII] -
        k[625]*y[IDX_CII] - k[1229]*y[IDX_HeII] - k[1230]*y[IDX_HeII] - k[2027]
        - k[2145];
    data[2924] = 0.0 + k[1578]*y[IDX_C2HI];
    data[2925] = 0.0 - k[1229]*y[IDX_HC3NI] - k[1230]*y[IDX_HC3NI];
    data[2926] = 0.0 + k[1579]*y[IDX_C2HI];
    data[2927] = 0.0 + k[1797]*y[IDX_C3H2I];
    data[2928] = 0.0 + k[1580]*y[IDX_C2HI];
    data[2929] = 0.0 + k[2395] + k[2396] + k[2397] + k[2398];
    data[2930] = 0.0 - k[832]*y[IDX_HClI];
    data[2931] = 0.0 + k[1720]*y[IDX_H2I];
    data[2932] = 0.0 + k[874]*y[IDX_H2ClII];
    data[2933] = 0.0 + k[509]*y[IDX_H2ClII];
    data[2934] = 0.0 - k[1787]*y[IDX_HClI];
    data[2935] = 0.0 - k[126]*y[IDX_HClI];
    data[2936] = 0.0 + k[1720]*y[IDX_ClI];
    data[2937] = 0.0 + k[509]*y[IDX_EM] + k[874]*y[IDX_COI] +
        k[999]*y[IDX_H2OI];
    data[2938] = 0.0 + k[999]*y[IDX_H2ClII];
    data[2939] = 0.0 - k[1049]*y[IDX_HClI];
    data[2940] = 0.0 - k[126]*y[IDX_HII] - k[413] - k[832]*y[IDX_CH5II] -
        k[1049]*y[IDX_H3II] - k[1241]*y[IDX_HeII] - k[1787]*y[IDX_HI] - k[2034]
        - k[2035] - k[2195];
    data[2941] = 0.0 - k[1241]*y[IDX_HClI];
    data[2942] = 0.0 + k[1040]*y[IDX_H3II];
    data[2943] = 0.0 + k[944]*y[IDX_H2I];
    data[2944] = 0.0 - k[548]*y[IDX_HClII];
    data[2945] = 0.0 + k[126]*y[IDX_HClI];
    data[2946] = 0.0 + k[944]*y[IDX_ClII] - k[948]*y[IDX_HClII];
    data[2947] = 0.0 + k[1040]*y[IDX_ClI];
    data[2948] = 0.0 + k[126]*y[IDX_HII] + k[2035];
    data[2949] = 0.0 - k[548]*y[IDX_EM] - k[948]*y[IDX_H2I] - k[2197];
    data[2950] = 0.0 + k[2331] + k[2332] + k[2333] + k[2334];
    data[2951] = 0.0 + k[1599]*y[IDX_NH2I];
    data[2952] = 0.0 + k[624]*y[IDX_HC3NI];
    data[2953] = 0.0 - k[1571]*y[IDX_HCNI];
    data[2954] = 0.0 - k[1578]*y[IDX_HCNI];
    data[2955] = 0.0 - k[661]*y[IDX_HCNI] - k[662]*y[IDX_HCNI];
    data[2956] = 0.0 + k[46]*y[IDX_HCNII] + k[1574]*y[IDX_NOI];
    data[2957] = 0.0 - k[668]*y[IDX_HCNI] + k[1330]*y[IDX_NI];
    data[2958] = 0.0 + k[1702]*y[IDX_CNI] + k[1792]*y[IDX_NI];
    data[2959] = 0.0 + k[993]*y[IDX_H2OI] + k[1018]*y[IDX_H2SI] +
        k[1421]*y[IDX_NH3I];
    data[2960] = 0.0 - k[1118]*y[IDX_HCNI];
    data[2961] = 0.0 - k[1119]*y[IDX_HCNI];
    data[2962] = 0.0 + k[847]*y[IDX_HCNHII] + k[1681]*y[IDX_N2I] +
        k[1684]*y[IDX_NOI];
    data[2963] = 0.0 - k[723]*y[IDX_HCNI] - k[724]*y[IDX_HCNI] -
        k[725]*y[IDX_HCNI];
    data[2964] = 0.0 + k[761]*y[IDX_HCNHII] + k[1623]*y[IDX_CNI] +
        k[1627]*y[IDX_N2I] + k[1630]*y[IDX_NOI] + k[1801]*y[IDX_NI];
    data[2965] = 0.0 + k[1650]*y[IDX_CNI] + k[1659]*y[IDX_NOI] +
        k[1805]*y[IDX_NI] + k[1806]*y[IDX_NI];
    data[2966] = 0.0 - k[2108]*y[IDX_HCNI];
    data[2967] = 0.0 + k[886]*y[IDX_HII] + k[1130]*y[IDX_HCNHII];
    data[2968] = 0.0 - k[1120]*y[IDX_HCNI];
    data[2969] = 0.0 + k[1670]*y[IDX_CNI];
    data[2970] = 0.0 - k[830]*y[IDX_HCNI];
    data[2971] = 0.0 + k[1623]*y[IDX_CH2I] + k[1650]*y[IDX_CH3I] +
        k[1670]*y[IDX_CH4I] + k[1702]*y[IDX_C2H4I] + k[1704]*y[IDX_H2COI] -
        k[1705]*y[IDX_HCNI] + k[1706]*y[IDX_HCOI] + k[1708]*y[IDX_HNOI] +
        k[1715]*y[IDX_SiH4I] + k[1726]*y[IDX_H2I] + k[1838]*y[IDX_NH3I] +
        k[1840]*y[IDX_NHI] + k[1932]*y[IDX_OHI];
    data[2972] = 0.0 - k[95]*y[IDX_HCNI] + k[865]*y[IDX_H2COI] -
        k[866]*y[IDX_HCNI];
    data[2973] = 0.0 - k[200]*y[IDX_HCNI];
    data[2974] = 0.0 + k[540]*y[IDX_HCNHII];
    data[2975] = 0.0 + k[194]*y[IDX_HCNII] + k[1746]*y[IDX_H2CNI] -
        k[1750]*y[IDX_HCNI] + k[1754]*y[IDX_HNCI] + k[1759]*y[IDX_NCCNI] +
        k[1772]*y[IDX_OCNI];
    data[2976] = 0.0 + k[1]*y[IDX_HNCI] - k[124]*y[IDX_HCNI] +
        k[886]*y[IDX_CH3CNI];
    data[2977] = 0.0 + k[1726]*y[IDX_CNI];
    data[2978] = 0.0 - k[164]*y[IDX_HCNI];
    data[2979] = 0.0 + k[398] + k[1746]*y[IDX_HI] + k[1810]*y[IDX_NI] +
        k[2010];
    data[2980] = 0.0 + k[865]*y[IDX_CNII] + k[1132]*y[IDX_HCNHII] +
        k[1704]*y[IDX_CNI];
    data[2981] = 0.0 - k[1121]*y[IDX_HCNI];
    data[2982] = 0.0 + k[188]*y[IDX_HCNII] + k[993]*y[IDX_C2NII];
    data[2983] = 0.0 - k[983]*y[IDX_HCNI];
    data[2984] = 0.0 + k[1018]*y[IDX_C2NII] + k[1134]*y[IDX_HCNHII];
    data[2985] = 0.0 - k[1045]*y[IDX_HCNI];
    data[2986] = 0.0 - k[1122]*y[IDX_HCNI];
    data[2987] = 0.0 - k[1088]*y[IDX_HCNI];
    data[2988] = 0.0 - k[1123]*y[IDX_HCNI];
    data[2989] = 0.0 + k[624]*y[IDX_CII];
    data[2990] = 0.0 - k[95]*y[IDX_CNII] - k[124]*y[IDX_HII] -
        k[164]*y[IDX_H2II] - k[200]*y[IDX_COII] - k[201]*y[IDX_N2II] -
        k[242]*y[IDX_NII] - k[408] - k[661]*y[IDX_C2HII] - k[662]*y[IDX_C2HII] -
        k[668]*y[IDX_C2H2II] - k[723]*y[IDX_CHII] - k[724]*y[IDX_CHII] -
        k[725]*y[IDX_CHII] - k[830]*y[IDX_CH5II] - k[866]*y[IDX_CNII] -
        k[983]*y[IDX_H2OII] - k[1045]*y[IDX_H3II] - k[1088]*y[IDX_H3OII] -
        k[1113]*y[IDX_HCNII] - k[1118]*y[IDX_C2N2II] - k[1119]*y[IDX_C3II] -
        k[1120]*y[IDX_CH3OH2II] - k[1121]*y[IDX_H2COII] - k[1122]*y[IDX_H3COII]
        - k[1123]*y[IDX_H3SII] - k[1124]*y[IDX_HCOII] - k[1125]*y[IDX_HNOII] -
        k[1126]*y[IDX_HSII] - k[1127]*y[IDX_HSiSII] - k[1128]*y[IDX_N2HII] -
        k[1129]*y[IDX_O2HII] - k[1231]*y[IDX_HeII] - k[1232]*y[IDX_HeII] -
        k[1233]*y[IDX_HeII] - k[1234]*y[IDX_HeII] - k[1360]*y[IDX_NHII] -
        k[1385]*y[IDX_NH2II] - k[1474]*y[IDX_OII] - k[1475]*y[IDX_OII] -
        k[1525]*y[IDX_OHII] - k[1571]*y[IDX_C2I] - k[1578]*y[IDX_C2HI] -
        k[1705]*y[IDX_CNI] - k[1750]*y[IDX_HI] - k[1889]*y[IDX_OI] -
        k[1890]*y[IDX_OI] - k[1891]*y[IDX_OI] - k[1939]*y[IDX_OHI] -
        k[1940]*y[IDX_OHI] - k[2028] - k[2108]*y[IDX_CH3II] - k[2223];
    data[2991] = 0.0 + k[46]*y[IDX_C2H2I] + k[188]*y[IDX_H2OI] +
        k[194]*y[IDX_HI] + k[197]*y[IDX_NOI] + k[198]*y[IDX_O2I] +
        k[199]*y[IDX_SI] + k[287]*y[IDX_NH3I] - k[1113]*y[IDX_HCNI];
    data[2992] = 0.0 + k[540]*y[IDX_EM] + k[761]*y[IDX_CH2I] +
        k[847]*y[IDX_CHI] + k[1130]*y[IDX_CH3CNI] + k[1132]*y[IDX_H2COI] +
        k[1134]*y[IDX_H2SI] + k[1404]*y[IDX_NH2I] + k[1431]*y[IDX_NH3I];
    data[2993] = 0.0 + k[1706]*y[IDX_CNI] + k[1812]*y[IDX_NI];
    data[2994] = 0.0 - k[1124]*y[IDX_HCNI];
    data[2995] = 0.0 + k[1814]*y[IDX_NI];
    data[2996] = 0.0 - k[1231]*y[IDX_HCNI] - k[1232]*y[IDX_HCNI] -
        k[1233]*y[IDX_HCNI] - k[1234]*y[IDX_HCNI];
    data[2997] = 0.0 + k[1]*y[IDX_HII] + k[1754]*y[IDX_HI];
    data[2998] = 0.0 + k[1708]*y[IDX_CNI];
    data[2999] = 0.0 - k[1125]*y[IDX_HCNI];
    data[3000] = 0.0 - k[1126]*y[IDX_HCNI];
    data[3001] = 0.0 - k[1127]*y[IDX_HCNI];
    data[3002] = 0.0 + k[1330]*y[IDX_C2H2II] + k[1792]*y[IDX_C2H4I] +
        k[1801]*y[IDX_CH2I] + k[1805]*y[IDX_CH3I] + k[1806]*y[IDX_CH3I] +
        k[1810]*y[IDX_H2CNI] + k[1812]*y[IDX_HCOI] + k[1814]*y[IDX_HCSI];
    data[3003] = 0.0 - k[242]*y[IDX_HCNI];
    data[3004] = 0.0 + k[1627]*y[IDX_CH2I] + k[1681]*y[IDX_CHI];
    data[3005] = 0.0 - k[201]*y[IDX_HCNI];
    data[3006] = 0.0 - k[1128]*y[IDX_HCNI];
    data[3007] = 0.0 + k[1759]*y[IDX_HI] + k[1943]*y[IDX_OHI];
    data[3008] = 0.0 + k[1840]*y[IDX_CNI];
    data[3009] = 0.0 - k[1360]*y[IDX_HCNI];
    data[3010] = 0.0 + k[1404]*y[IDX_HCNHII] + k[1599]*y[IDX_CI];
    data[3011] = 0.0 - k[1385]*y[IDX_HCNI];
    data[3012] = 0.0 + k[287]*y[IDX_HCNII] + k[1421]*y[IDX_C2NII] +
        k[1431]*y[IDX_HCNHII] + k[1838]*y[IDX_CNI];
    data[3013] = 0.0 + k[197]*y[IDX_HCNII] + k[1574]*y[IDX_C2H2I] +
        k[1630]*y[IDX_CH2I] + k[1659]*y[IDX_CH3I] + k[1684]*y[IDX_CHI];
    data[3014] = 0.0 - k[1889]*y[IDX_HCNI] - k[1890]*y[IDX_HCNI] -
        k[1891]*y[IDX_HCNI];
    data[3015] = 0.0 - k[1474]*y[IDX_HCNI] - k[1475]*y[IDX_HCNI];
    data[3016] = 0.0 + k[198]*y[IDX_HCNII];
    data[3017] = 0.0 - k[1129]*y[IDX_HCNI];
    data[3018] = 0.0 + k[1772]*y[IDX_HI];
    data[3019] = 0.0 + k[1932]*y[IDX_CNI] - k[1939]*y[IDX_HCNI] -
        k[1940]*y[IDX_HCNI] + k[1943]*y[IDX_NCCNI];
    data[3020] = 0.0 - k[1525]*y[IDX_HCNI];
    data[3021] = 0.0 + k[199]*y[IDX_HCNII];
    data[3022] = 0.0 + k[1715]*y[IDX_CNI];
    data[3023] = 0.0 - k[693]*y[IDX_HCNII];
    data[3024] = 0.0 + k[629]*y[IDX_NH2I] + k[630]*y[IDX_NH3I];
    data[3025] = 0.0 - k[652]*y[IDX_HCNII] + k[1347]*y[IDX_NHII];
    data[3026] = 0.0 - k[679]*y[IDX_HCNII];
    data[3027] = 0.0 - k[46]*y[IDX_HCNII];
    data[3028] = 0.0 + k[1097]*y[IDX_HI] + k[1118]*y[IDX_HCNI];
    data[3029] = 0.0 - k[846]*y[IDX_HCNII];
    data[3030] = 0.0 + k[729]*y[IDX_NH2I];
    data[3031] = 0.0 - k[760]*y[IDX_HCNII];
    data[3032] = 0.0 + k[1331]*y[IDX_NI];
    data[3033] = 0.0 - k[811]*y[IDX_HCNII] + k[1294]*y[IDX_NII];
    data[3034] = 0.0 + k[869]*y[IDX_HNOII] + k[870]*y[IDX_O2HII] +
        k[917]*y[IDX_H2II] + k[1035]*y[IDX_H3II] + k[1349]*y[IDX_NHII] +
        k[1519]*y[IDX_OHII];
    data[3035] = 0.0 + k[95]*y[IDX_HCNI] + k[867]*y[IDX_HCOI] +
        k[940]*y[IDX_H2I] + k[995]*y[IDX_H2OI];
    data[3036] = 0.0 - k[1111]*y[IDX_HCNII];
    data[3037] = 0.0 + k[200]*y[IDX_HCNI];
    data[3038] = 0.0 - k[1110]*y[IDX_HCNII];
    data[3039] = 0.0 - k[538]*y[IDX_HCNII];
    data[3040] = 0.0 - k[194]*y[IDX_HCNII] + k[1097]*y[IDX_C2N2II];
    data[3041] = 0.0 + k[124]*y[IDX_HCNI];
    data[3042] = 0.0 + k[940]*y[IDX_CNII] - k[947]*y[IDX_HCNII];
    data[3043] = 0.0 + k[164]*y[IDX_HCNI] + k[917]*y[IDX_CNI];
    data[3044] = 0.0 - k[1112]*y[IDX_HCNII];
    data[3045] = 0.0 - k[188]*y[IDX_HCNII] + k[995]*y[IDX_CNII] -
        k[1002]*y[IDX_HCNII];
    data[3046] = 0.0 + k[1035]*y[IDX_CNI];
    data[3047] = 0.0 + k[95]*y[IDX_CNII] + k[124]*y[IDX_HII] +
        k[164]*y[IDX_H2II] + k[200]*y[IDX_COII] + k[201]*y[IDX_N2II] +
        k[242]*y[IDX_NII] - k[1113]*y[IDX_HCNII] + k[1118]*y[IDX_C2N2II];
    data[3048] = 0.0 - k[46]*y[IDX_C2H2I] - k[188]*y[IDX_H2OI] -
        k[194]*y[IDX_HI] - k[197]*y[IDX_NOI] - k[198]*y[IDX_O2I] -
        k[199]*y[IDX_SI] - k[287]*y[IDX_NH3I] - k[538]*y[IDX_EM] -
        k[652]*y[IDX_C2I] - k[679]*y[IDX_C2HI] - k[693]*y[IDX_CI] -
        k[760]*y[IDX_CH2I] - k[811]*y[IDX_CH4I] - k[846]*y[IDX_CHI] -
        k[947]*y[IDX_H2I] - k[1002]*y[IDX_H2OI] - k[1110]*y[IDX_CO2I] -
        k[1111]*y[IDX_COI] - k[1112]*y[IDX_H2COI] - k[1113]*y[IDX_HCNI] -
        k[1114]*y[IDX_HCOI] - k[1115]*y[IDX_HCOI] - k[1116]*y[IDX_HNCI] -
        k[1117]*y[IDX_SI] - k[1403]*y[IDX_NH2I] - k[1430]*y[IDX_NH3I] -
        k[1451]*y[IDX_NHI] - k[1541]*y[IDX_OHI] - k[2241];
    data[3049] = 0.0 + k[867]*y[IDX_CNII] - k[1114]*y[IDX_HCNII] -
        k[1115]*y[IDX_HCNII];
    data[3050] = 0.0 - k[1116]*y[IDX_HCNII];
    data[3051] = 0.0 + k[869]*y[IDX_CNI];
    data[3052] = 0.0 + k[1331]*y[IDX_CH2II];
    data[3053] = 0.0 + k[242]*y[IDX_HCNI] + k[1294]*y[IDX_CH4I];
    data[3054] = 0.0 + k[201]*y[IDX_HCNI];
    data[3055] = 0.0 - k[1451]*y[IDX_HCNII];
    data[3056] = 0.0 + k[1347]*y[IDX_C2I] + k[1349]*y[IDX_CNI];
    data[3057] = 0.0 + k[629]*y[IDX_CII] + k[729]*y[IDX_CHII] -
        k[1403]*y[IDX_HCNII];
    data[3058] = 0.0 - k[287]*y[IDX_HCNII] + k[630]*y[IDX_CII] -
        k[1430]*y[IDX_HCNII];
    data[3059] = 0.0 - k[197]*y[IDX_HCNII];
    data[3060] = 0.0 - k[198]*y[IDX_HCNII];
    data[3061] = 0.0 + k[870]*y[IDX_CNI];
    data[3062] = 0.0 - k[1541]*y[IDX_HCNII];
    data[3063] = 0.0 + k[1519]*y[IDX_CNI];
    data[3064] = 0.0 - k[199]*y[IDX_HCNII] - k[1117]*y[IDX_HCNII];
    data[3065] = 0.0 + k[662]*y[IDX_HCNI] + k[664]*y[IDX_HNCI];
    data[3066] = 0.0 + k[668]*y[IDX_HCNI] + k[669]*y[IDX_HNCI];
    data[3067] = 0.0 + k[1421]*y[IDX_NH3I];
    data[3068] = 0.0 - k[847]*y[IDX_HCNHII] - k[848]*y[IDX_HCNHII];
    data[3069] = 0.0 + k[725]*y[IDX_HCNI] + k[727]*y[IDX_HNCI];
    data[3070] = 0.0 - k[761]*y[IDX_HCNHII] - k[762]*y[IDX_HCNHII];
    data[3071] = 0.0 + k[775]*y[IDX_CH3CNI] + k[1446]*y[IDX_NHI];
    data[3072] = 0.0 + k[775]*y[IDX_CH3II] - k[1130]*y[IDX_HCNHII] -
        k[1131]*y[IDX_HCNHII];
    data[3073] = 0.0 + k[811]*y[IDX_HCNII] + k[1295]*y[IDX_NII];
    data[3074] = 0.0 + k[830]*y[IDX_HCNI] + k[833]*y[IDX_HNCI];
    data[3075] = 0.0 - k[539]*y[IDX_HCNHII] - k[540]*y[IDX_HCNHII] -
        k[541]*y[IDX_HCNHII];
    data[3076] = 0.0 + k[947]*y[IDX_HCNII];
    data[3077] = 0.0 - k[1132]*y[IDX_HCNHII] - k[1133]*y[IDX_HCNHII];
    data[3078] = 0.0 + k[1121]*y[IDX_HCNI] + k[1165]*y[IDX_HNCI];
    data[3079] = 0.0 + k[983]*y[IDX_HCNI] + k[986]*y[IDX_HNCI];
    data[3080] = 0.0 - k[1134]*y[IDX_HCNHII] - k[1135]*y[IDX_HCNHII];
    data[3081] = 0.0 + k[1045]*y[IDX_HCNI] + k[1050]*y[IDX_HNCI];
    data[3082] = 0.0 + k[1122]*y[IDX_HCNI] + k[1166]*y[IDX_HNCI];
    data[3083] = 0.0 + k[1088]*y[IDX_HCNI] + k[1090]*y[IDX_HNCI];
    data[3084] = 0.0 + k[1123]*y[IDX_HCNI] + k[1167]*y[IDX_HNCI];
    data[3085] = 0.0 + k[662]*y[IDX_C2HII] + k[668]*y[IDX_C2H2II] +
        k[725]*y[IDX_CHII] + k[830]*y[IDX_CH5II] + k[983]*y[IDX_H2OII] +
        k[1045]*y[IDX_H3II] + k[1088]*y[IDX_H3OII] + k[1113]*y[IDX_HCNII] +
        k[1121]*y[IDX_H2COII] + k[1122]*y[IDX_H3COII] + k[1123]*y[IDX_H3SII] +
        k[1124]*y[IDX_HCOII] + k[1125]*y[IDX_HNOII] + k[1126]*y[IDX_HSII] +
        k[1127]*y[IDX_HSiSII] + k[1128]*y[IDX_N2HII] + k[1129]*y[IDX_O2HII] +
        k[1360]*y[IDX_NHII] + k[1385]*y[IDX_NH2II] + k[1525]*y[IDX_OHII];
    data[3086] = 0.0 + k[811]*y[IDX_CH4I] + k[947]*y[IDX_H2I] +
        k[1113]*y[IDX_HCNI] + k[1115]*y[IDX_HCOI] + k[1116]*y[IDX_HNCI] +
        k[1430]*y[IDX_NH3I];
    data[3087] = 0.0 - k[539]*y[IDX_EM] - k[540]*y[IDX_EM] -
        k[541]*y[IDX_EM] - k[761]*y[IDX_CH2I] - k[762]*y[IDX_CH2I] -
        k[847]*y[IDX_CHI] - k[848]*y[IDX_CHI] - k[1130]*y[IDX_CH3CNI] -
        k[1131]*y[IDX_CH3CNI] - k[1132]*y[IDX_H2COI] - k[1133]*y[IDX_H2COI] -
        k[1134]*y[IDX_H2SI] - k[1135]*y[IDX_H2SI] - k[1404]*y[IDX_NH2I] -
        k[1405]*y[IDX_NH2I] - k[1431]*y[IDX_NH3I] - k[1432]*y[IDX_NH3I] -
        k[2289];
    data[3088] = 0.0 + k[1115]*y[IDX_HCNII];
    data[3089] = 0.0 + k[1124]*y[IDX_HCNI] + k[1168]*y[IDX_HNCI];
    data[3090] = 0.0 + k[664]*y[IDX_C2HII] + k[669]*y[IDX_C2H2II] +
        k[727]*y[IDX_CHII] + k[833]*y[IDX_CH5II] + k[986]*y[IDX_H2OII] +
        k[1050]*y[IDX_H3II] + k[1090]*y[IDX_H3OII] + k[1116]*y[IDX_HCNII] +
        k[1165]*y[IDX_H2COII] + k[1166]*y[IDX_H3COII] + k[1167]*y[IDX_H3SII] +
        k[1168]*y[IDX_HCOII] + k[1169]*y[IDX_HNOII] + k[1170]*y[IDX_HSII] +
        k[1171]*y[IDX_N2HII] + k[1172]*y[IDX_O2HII] + k[1362]*y[IDX_NHII] +
        k[1387]*y[IDX_NH2II] + k[1528]*y[IDX_OHII];
    data[3091] = 0.0 + k[1125]*y[IDX_HCNI] + k[1169]*y[IDX_HNCI];
    data[3092] = 0.0 + k[1126]*y[IDX_HCNI] + k[1170]*y[IDX_HNCI];
    data[3093] = 0.0 + k[1127]*y[IDX_HCNI];
    data[3094] = 0.0 + k[1295]*y[IDX_CH4I];
    data[3095] = 0.0 + k[1128]*y[IDX_HCNI] + k[1171]*y[IDX_HNCI];
    data[3096] = 0.0 + k[1446]*y[IDX_CH3II];
    data[3097] = 0.0 + k[1360]*y[IDX_HCNI] + k[1362]*y[IDX_HNCI];
    data[3098] = 0.0 - k[1404]*y[IDX_HCNHII] - k[1405]*y[IDX_HCNHII];
    data[3099] = 0.0 + k[1385]*y[IDX_HCNI] + k[1387]*y[IDX_HNCI];
    data[3100] = 0.0 + k[1421]*y[IDX_C2NII] + k[1430]*y[IDX_HCNII] -
        k[1431]*y[IDX_HCNHII] - k[1432]*y[IDX_HCNHII];
    data[3101] = 0.0 + k[1129]*y[IDX_HCNI] + k[1172]*y[IDX_HNCI];
    data[3102] = 0.0 + k[1525]*y[IDX_HCNI] + k[1528]*y[IDX_HNCI];
    data[3103] = 0.0 - k[1594]*y[IDX_HCOI];
    data[3104] = 0.0 - k[18]*y[IDX_HCOI] + k[613]*y[IDX_CH3OHI] -
        k[626]*y[IDX_HCOI];
    data[3105] = 0.0 + k[651]*y[IDX_H2COII];
    data[3106] = 0.0 - k[33]*y[IDX_HCOI] - k[648]*y[IDX_HCOI];
    data[3107] = 0.0 + k[678]*y[IDX_H2COII] + k[1581]*y[IDX_O2I];
    data[3108] = 0.0 - k[663]*y[IDX_HCOI];
    data[3109] = 0.0 + k[1558]*y[IDX_SOII];
    data[3110] = 0.0 - k[44]*y[IDX_HCOI];
    data[3111] = 0.0 + k[1576]*y[IDX_O2I];
    data[3112] = 0.0 + k[1561]*y[IDX_SOII] + k[1873]*y[IDX_OI];
    data[3113] = 0.0 + k[842]*y[IDX_H2COII] + k[1677]*y[IDX_CO2I] +
        k[1678]*y[IDX_H2COI] - k[1679]*y[IDX_HCOI] + k[1685]*y[IDX_NOI] +
        k[1690]*y[IDX_O2I] + k[1691]*y[IDX_O2HI] + k[1696]*y[IDX_OHI];
    data[3114] = 0.0 - k[55]*y[IDX_HCOI] - k[726]*y[IDX_HCOI] +
        k[734]*y[IDX_O2I];
    data[3115] = 0.0 + k[757]*y[IDX_H2COII] + k[1624]*y[IDX_H2COI] -
        k[1625]*y[IDX_HCOI] + k[1636]*y[IDX_O2I] + k[1639]*y[IDX_OI];
    data[3116] = 0.0 - k[747]*y[IDX_HCOI] + k[752]*y[IDX_OCSI];
    data[3117] = 0.0 + k[1651]*y[IDX_H2COI] - k[1654]*y[IDX_HCOI] +
        k[1661]*y[IDX_O2I];
    data[3118] = 0.0 - k[72]*y[IDX_HCOI] - k[779]*y[IDX_HCOI];
    data[3119] = 0.0 + k[613]*y[IDX_CII] + k[965]*y[IDX_H2COII];
    data[3120] = 0.0 - k[831]*y[IDX_HCOI];
    data[3121] = 0.0 + k[1704]*y[IDX_H2COI] - k[1706]*y[IDX_HCOI];
    data[3122] = 0.0 - k[96]*y[IDX_HCOI] - k[867]*y[IDX_HCOI];
    data[3123] = 0.0 - k[103]*y[IDX_HCOI] + k[871]*y[IDX_H2COI];
    data[3124] = 0.0 + k[1352]*y[IDX_NHII] + k[1677]*y[IDX_CHI];
    data[3125] = 0.0 + k[505]*y[IDX_H2COII] + k[525]*y[IDX_H3COII] +
        k[536]*y[IDX_H5C2O2II];
    data[3126] = 0.0 + k[1747]*y[IDX_H2COI] - k[1751]*y[IDX_HCOI] -
        k[1752]*y[IDX_HCOI];
    data[3127] = 0.0 - k[125]*y[IDX_HCOI] - k[897]*y[IDX_HCOI] -
        k[898]*y[IDX_HCOI];
    data[3128] = 0.0 - k[165]*y[IDX_HCOI] - k[925]*y[IDX_HCOI];
    data[3129] = 0.0 + k[871]*y[IDX_COII] + k[966]*y[IDX_H2COII] +
        k[1377]*y[IDX_NH2II] + k[1413]*y[IDX_NH3II] + k[1624]*y[IDX_CH2I] +
        k[1651]*y[IDX_CH3I] + k[1678]*y[IDX_CHI] + k[1704]*y[IDX_CNI] +
        k[1747]*y[IDX_HI] + k[1886]*y[IDX_OI] + k[1937]*y[IDX_OHI];
    data[3130] = 0.0 - k[202]*y[IDX_HCOI] + k[505]*y[IDX_EM] +
        k[651]*y[IDX_C2I] + k[678]*y[IDX_C2HI] + k[757]*y[IDX_CH2I] +
        k[842]*y[IDX_CHI] + k[965]*y[IDX_CH3OHI] + k[966]*y[IDX_H2COI] +
        k[968]*y[IDX_SI] + k[998]*y[IDX_H2OI] + k[1121]*y[IDX_HCNI] -
        k[1158]*y[IDX_HCOI] + k[1165]*y[IDX_HNCI] + k[1399]*y[IDX_NH2I] +
        k[1424]*y[IDX_NH3I];
    data[3131] = 0.0 + k[998]*y[IDX_H2COII];
    data[3132] = 0.0 - k[180]*y[IDX_HCOI] - k[984]*y[IDX_HCOI] -
        k[985]*y[IDX_HCOI];
    data[3133] = 0.0 - k[203]*y[IDX_HCOI];
    data[3134] = 0.0 - k[1046]*y[IDX_HCOI];
    data[3135] = 0.0 + k[525]*y[IDX_EM];
    data[3136] = 0.0 + k[536]*y[IDX_EM];
    data[3137] = 0.0 + k[1121]*y[IDX_H2COII];
    data[3138] = 0.0 - k[1114]*y[IDX_HCOI] - k[1115]*y[IDX_HCOI];
    data[3139] = 0.0 - k[18]*y[IDX_CII] - k[33]*y[IDX_C2II] -
        k[44]*y[IDX_C2H2II] - k[55]*y[IDX_CHII] - k[72]*y[IDX_CH3II] -
        k[96]*y[IDX_CNII] - k[103]*y[IDX_COII] - k[125]*y[IDX_HII] -
        k[165]*y[IDX_H2II] - k[180]*y[IDX_H2OII] - k[202]*y[IDX_H2COII] -
        k[203]*y[IDX_H2SII] - k[204]*y[IDX_O2II] - k[205]*y[IDX_SII] -
        k[206]*y[IDX_SiOII] - k[243]*y[IDX_NII] - k[254]*y[IDX_N2II] -
        k[267]*y[IDX_NH2II] - k[278]*y[IDX_NH3II] - k[314]*y[IDX_OII] -
        k[334]*y[IDX_OHII] - k[409] - k[410] - k[626]*y[IDX_CII] -
        k[648]*y[IDX_C2II] - k[663]*y[IDX_C2HII] - k[726]*y[IDX_CHII] -
        k[747]*y[IDX_CH2II] - k[779]*y[IDX_CH3II] - k[831]*y[IDX_CH5II] -
        k[867]*y[IDX_CNII] - k[897]*y[IDX_HII] - k[898]*y[IDX_HII] -
        k[925]*y[IDX_H2II] - k[984]*y[IDX_H2OII] - k[985]*y[IDX_H2OII] -
        k[1046]*y[IDX_H3II] - k[1114]*y[IDX_HCNII] - k[1115]*y[IDX_HCNII] -
        k[1144]*y[IDX_HCOII] - k[1158]*y[IDX_H2COII] - k[1159]*y[IDX_HNOII] -
        k[1160]*y[IDX_N2HII] - k[1161]*y[IDX_O2II] - k[1162]*y[IDX_O2HII] -
        k[1163]*y[IDX_SII] - k[1235]*y[IDX_HeII] - k[1236]*y[IDX_HeII] -
        k[1237]*y[IDX_HeII] - k[1303]*y[IDX_NII] - k[1317]*y[IDX_N2II] -
        k[1361]*y[IDX_NHII] - k[1386]*y[IDX_NH2II] - k[1416]*y[IDX_NH3II] -
        k[1476]*y[IDX_OII] - k[1526]*y[IDX_OHII] - k[1527]*y[IDX_OHII] -
        k[1594]*y[IDX_CI] - k[1625]*y[IDX_CH2I] - k[1654]*y[IDX_CH3I] -
        k[1679]*y[IDX_CHI] - k[1706]*y[IDX_CNI] - k[1751]*y[IDX_HI] -
        k[1752]*y[IDX_HI] - k[1780]*y[IDX_HCOI] - k[1780]*y[IDX_HCOI] -
        k[1780]*y[IDX_HCOI] - k[1780]*y[IDX_HCOI] - k[1781]*y[IDX_HCOI] -
        k[1781]*y[IDX_HCOI] - k[1781]*y[IDX_HCOI] - k[1781]*y[IDX_HCOI] -
        k[1782]*y[IDX_HNOI] - k[1783]*y[IDX_NOI] - k[1784]*y[IDX_O2I] -
        k[1785]*y[IDX_O2I] - k[1786]*y[IDX_O2HI] - k[1811]*y[IDX_NI] -
        k[1812]*y[IDX_NI] - k[1813]*y[IDX_NI] - k[1892]*y[IDX_OI] -
        k[1893]*y[IDX_OI] - k[1941]*y[IDX_OHI] - k[1951]*y[IDX_SI] -
        k[1952]*y[IDX_SI] - k[2030] - k[2031] - k[2218];
    data[3140] = 0.0 + k[224]*y[IDX_MgI] - k[1144]*y[IDX_HCOI];
    data[3141] = 0.0 - k[1235]*y[IDX_HCOI] - k[1236]*y[IDX_HCOI] -
        k[1237]*y[IDX_HCOI];
    data[3142] = 0.0 + k[1165]*y[IDX_H2COII];
    data[3143] = 0.0 - k[1782]*y[IDX_HCOI];
    data[3144] = 0.0 - k[1159]*y[IDX_HCOI];
    data[3145] = 0.0 + k[224]*y[IDX_HCOII];
    data[3146] = 0.0 - k[1811]*y[IDX_HCOI] - k[1812]*y[IDX_HCOI] -
        k[1813]*y[IDX_HCOI];
    data[3147] = 0.0 - k[243]*y[IDX_HCOI] - k[1303]*y[IDX_HCOI];
    data[3148] = 0.0 - k[254]*y[IDX_HCOI] - k[1317]*y[IDX_HCOI];
    data[3149] = 0.0 - k[1160]*y[IDX_HCOI];
    data[3150] = 0.0 + k[1352]*y[IDX_CO2I] - k[1361]*y[IDX_HCOI];
    data[3151] = 0.0 + k[1399]*y[IDX_H2COII];
    data[3152] = 0.0 - k[267]*y[IDX_HCOI] + k[1377]*y[IDX_H2COI] -
        k[1386]*y[IDX_HCOI];
    data[3153] = 0.0 + k[1424]*y[IDX_H2COII];
    data[3154] = 0.0 - k[278]*y[IDX_HCOI] + k[1413]*y[IDX_H2COI] -
        k[1416]*y[IDX_HCOI];
    data[3155] = 0.0 + k[1685]*y[IDX_CHI] - k[1783]*y[IDX_HCOI];
    data[3156] = 0.0 + k[1639]*y[IDX_CH2I] + k[1873]*y[IDX_C2H4I] +
        k[1886]*y[IDX_H2COI] - k[1892]*y[IDX_HCOI] - k[1893]*y[IDX_HCOI];
    data[3157] = 0.0 - k[314]*y[IDX_HCOI] - k[1476]*y[IDX_HCOI];
    data[3158] = 0.0 + k[734]*y[IDX_CHII] + k[1576]*y[IDX_C2H3I] +
        k[1581]*y[IDX_C2HI] + k[1636]*y[IDX_CH2I] + k[1661]*y[IDX_CH3I] +
        k[1690]*y[IDX_CHI] - k[1784]*y[IDX_HCOI] - k[1785]*y[IDX_HCOI];
    data[3159] = 0.0 - k[204]*y[IDX_HCOI] - k[1161]*y[IDX_HCOI];
    data[3160] = 0.0 + k[1691]*y[IDX_CHI] - k[1786]*y[IDX_HCOI];
    data[3161] = 0.0 - k[1162]*y[IDX_HCOI];
    data[3162] = 0.0 + k[752]*y[IDX_CH2II];
    data[3163] = 0.0 + k[1696]*y[IDX_CHI] + k[1937]*y[IDX_H2COI] -
        k[1941]*y[IDX_HCOI];
    data[3164] = 0.0 - k[334]*y[IDX_HCOI] - k[1526]*y[IDX_HCOI] -
        k[1527]*y[IDX_HCOI];
    data[3165] = 0.0 + k[968]*y[IDX_H2COII] - k[1951]*y[IDX_HCOI] -
        k[1952]*y[IDX_HCOI];
    data[3166] = 0.0 - k[205]*y[IDX_HCOI] - k[1163]*y[IDX_HCOI];
    data[3167] = 0.0 - k[206]*y[IDX_HCOI];
    data[3168] = 0.0 + k[1558]*y[IDX_C2H2I] + k[1561]*y[IDX_C2H4I];
    data[3169] = 0.0 + k[692]*y[IDX_H3OII] - k[694]*y[IDX_HCOII];
    data[3170] = 0.0 + k[18]*y[IDX_HCOI] + k[618]*y[IDX_H2COI] +
        k[620]*y[IDX_H2OI];
    data[3171] = 0.0 - k[653]*y[IDX_HCOII];
    data[3172] = 0.0 + k[33]*y[IDX_HCOI];
    data[3173] = 0.0 + k[677]*y[IDX_COII] - k[680]*y[IDX_HCOII];
    data[3174] = 0.0 + k[1491]*y[IDX_OI];
    data[3175] = 0.0 + k[1482]*y[IDX_O2II] + k[1557]*y[IDX_SOII];
    data[3176] = 0.0 + k[44]*y[IDX_HCOI] + k[1492]*y[IDX_OI];
    data[3177] = 0.0 - k[1136]*y[IDX_HCOII];
    data[3178] = 0.0 + k[993]*y[IDX_H2OI];
    data[3179] = 0.0 + k[994]*y[IDX_H2OI];
    data[3180] = 0.0 + k[0]*y[IDX_OI] + k[841]*y[IDX_COII] -
        k[849]*y[IDX_HCOII] + k[858]*y[IDX_O2II] + k[864]*y[IDX_SiOII];
    data[3181] = 0.0 + k[55]*y[IDX_HCOI] + k[714]*y[IDX_CO2I] +
        k[717]*y[IDX_H2COI] + k[720]*y[IDX_H2OI] + k[733]*y[IDX_O2I];
    data[3182] = 0.0 + k[756]*y[IDX_COII] - k[763]*y[IDX_HCOII];
    data[3183] = 0.0 + k[742]*y[IDX_H2COI] + k[749]*y[IDX_O2I] +
        k[750]*y[IDX_OI];
    data[3184] = 0.0 + k[72]*y[IDX_HCOI] + k[777]*y[IDX_H2COI] +
        k[784]*y[IDX_OI];
    data[3185] = 0.0 - k[1137]*y[IDX_HCOII];
    data[3186] = 0.0 - k[1138]*y[IDX_HCOII];
    data[3187] = 0.0 + k[889]*y[IDX_HII] - k[1139]*y[IDX_HCOII];
    data[3188] = 0.0 + k[807]*y[IDX_COII];
    data[3189] = 0.0 + k[797]*y[IDX_COI];
    data[3190] = 0.0 + k[826]*y[IDX_COI];
    data[3191] = 0.0 + k[96]*y[IDX_HCOI] + k[865]*y[IDX_H2COI] +
        k[996]*y[IDX_H2OI];
    data[3192] = 0.0 + k[797]*y[IDX_CH4II] + k[826]*y[IDX_CH5II] +
        k[874]*y[IDX_H2ClII] + k[875]*y[IDX_HCO2II] + k[876]*y[IDX_HNOII] +
        k[877]*y[IDX_N2HII] + k[878]*y[IDX_O2HII] + k[880]*y[IDX_SiH4II] +
        k[919]*y[IDX_H2II] + k[978]*y[IDX_H2OII] + k[1037]*y[IDX_H3II] +
        k[1111]*y[IDX_HCNII] + k[1353]*y[IDX_NHII] + k[1521]*y[IDX_OHII];
    data[3193] = 0.0 + k[103]*y[IDX_HCOI] + k[677]*y[IDX_C2HI] +
        k[756]*y[IDX_CH2I] + k[807]*y[IDX_CH4I] + k[841]*y[IDX_CHI] +
        k[871]*y[IDX_H2COI] + k[872]*y[IDX_H2SI] + k[941]*y[IDX_H2I] +
        k[997]*y[IDX_H2OI] + k[1398]*y[IDX_NH2I] + k[1423]*y[IDX_NH3I] +
        k[1448]*y[IDX_NHI] + k[1539]*y[IDX_OHI];
    data[3194] = 0.0 + k[714]*y[IDX_CHII] + k[891]*y[IDX_HII];
    data[3195] = 0.0 - k[1140]*y[IDX_HCOII];
    data[3196] = 0.0 - k[542]*y[IDX_HCOII];
    data[3197] = 0.0 + k[125]*y[IDX_HCOI] + k[889]*y[IDX_CH3OHI] +
        k[891]*y[IDX_CO2I] + k[893]*y[IDX_H2COI];
    data[3198] = 0.0 + k[5]*y[IDX_HOCII] + k[941]*y[IDX_COII];
    data[3199] = 0.0 + k[165]*y[IDX_HCOI] + k[919]*y[IDX_COI] +
        k[921]*y[IDX_H2COI];
    data[3200] = 0.0 + k[874]*y[IDX_COI];
    data[3201] = 0.0 + k[618]*y[IDX_CII] + k[717]*y[IDX_CHII] +
        k[742]*y[IDX_CH2II] + k[777]*y[IDX_CH3II] + k[865]*y[IDX_CNII] +
        k[871]*y[IDX_COII] + k[893]*y[IDX_HII] + k[921]*y[IDX_H2II] +
        k[972]*y[IDX_O2II] + k[975]*y[IDX_SII] - k[1141]*y[IDX_HCOII] +
        k[1217]*y[IDX_HeII] + k[1298]*y[IDX_NII] + k[1314]*y[IDX_N2II] +
        k[1355]*y[IDX_NHII] + k[1471]*y[IDX_OII] + k[2014];
    data[3202] = 0.0 + k[202]*y[IDX_HCOI] + k[967]*y[IDX_O2I];
    data[3203] = 0.0 - k[1142]*y[IDX_HCOII];
    data[3204] = 0.0 + k[620]*y[IDX_CII] + k[720]*y[IDX_CHII] +
        k[993]*y[IDX_C2NII] + k[994]*y[IDX_C4NII] + k[996]*y[IDX_CNII] +
        k[997]*y[IDX_COII] - k[1003]*y[IDX_HCOII];
    data[3205] = 0.0 + k[180]*y[IDX_HCOI] + k[978]*y[IDX_COI];
    data[3206] = 0.0 + k[872]*y[IDX_COII] - k[1143]*y[IDX_HCOII];
    data[3207] = 0.0 + k[203]*y[IDX_HCOI];
    data[3208] = 0.0 + k[1037]*y[IDX_COI];
    data[3209] = 0.0 + k[692]*y[IDX_CI];
    data[3210] = 0.0 - k[1124]*y[IDX_HCOII] + k[1474]*y[IDX_OII];
    data[3211] = 0.0 + k[1111]*y[IDX_COI];
    data[3212] = 0.0 + k[18]*y[IDX_CII] + k[33]*y[IDX_C2II] +
        k[44]*y[IDX_C2H2II] + k[55]*y[IDX_CHII] + k[72]*y[IDX_CH3II] +
        k[96]*y[IDX_CNII] + k[103]*y[IDX_COII] + k[125]*y[IDX_HII] +
        k[165]*y[IDX_H2II] + k[180]*y[IDX_H2OII] + k[202]*y[IDX_H2COII] +
        k[203]*y[IDX_H2SII] + k[204]*y[IDX_O2II] + k[205]*y[IDX_SII] +
        k[206]*y[IDX_SiOII] + k[243]*y[IDX_NII] + k[254]*y[IDX_N2II] +
        k[267]*y[IDX_NH2II] + k[278]*y[IDX_NH3II] + k[314]*y[IDX_OII] +
        k[334]*y[IDX_OHII] + k[410] - k[1144]*y[IDX_HCOII] + k[2031];
    data[3213] = 0.0 - k[224]*y[IDX_MgI] - k[542]*y[IDX_EM] -
        k[653]*y[IDX_C2I] - k[680]*y[IDX_C2HI] - k[694]*y[IDX_CI] -
        k[763]*y[IDX_CH2I] - k[849]*y[IDX_CHI] - k[1003]*y[IDX_H2OI] -
        k[1124]*y[IDX_HCNI] - k[1136]*y[IDX_C2H5OHI] - k[1137]*y[IDX_CH3CCHI] -
        k[1138]*y[IDX_CH3CNI] - k[1139]*y[IDX_CH3OHI] - k[1140]*y[IDX_CSI] -
        k[1141]*y[IDX_H2COI] - k[1142]*y[IDX_H2CSI] - k[1143]*y[IDX_H2SI] -
        k[1144]*y[IDX_HCOI] - k[1145]*y[IDX_HCOOCH3I] - k[1146]*y[IDX_HS2I] -
        k[1147]*y[IDX_HSI] - k[1148]*y[IDX_NSI] - k[1149]*y[IDX_OCSI] -
        k[1150]*y[IDX_S2I] - k[1151]*y[IDX_SI] - k[1152]*y[IDX_SOI] -
        k[1153]*y[IDX_SiH2I] - k[1154]*y[IDX_SiH4I] - k[1155]*y[IDX_SiHI] -
        k[1156]*y[IDX_SiOI] - k[1157]*y[IDX_SiSI] - k[1168]*y[IDX_HNCI] -
        k[1406]*y[IDX_NH2I] - k[1433]*y[IDX_NH3I] - k[1452]*y[IDX_NHI] -
        k[1542]*y[IDX_OHI] - k[1543]*y[IDX_OHI] - k[1566]*y[IDX_SiI] - k[2029] -
        k[2240];
    data[3214] = 0.0 + k[875]*y[IDX_COI] + k[1500]*y[IDX_OI];
    data[3215] = 0.0 - k[1145]*y[IDX_HCOII];
    data[3216] = 0.0 + k[1502]*y[IDX_OI];
    data[3217] = 0.0 + k[1217]*y[IDX_H2COI];
    data[3218] = 0.0 - k[1168]*y[IDX_HCOII];
    data[3219] = 0.0 + k[876]*y[IDX_COI];
    data[3220] = 0.0 + k[5]*y[IDX_H2I];
    data[3221] = 0.0 - k[1147]*y[IDX_HCOII];
    data[3222] = 0.0 - k[1146]*y[IDX_HCOII];
    data[3223] = 0.0 - k[224]*y[IDX_HCOII];
    data[3224] = 0.0 + k[243]*y[IDX_HCOI] + k[1298]*y[IDX_H2COI];
    data[3225] = 0.0 + k[254]*y[IDX_HCOI] + k[1314]*y[IDX_H2COI];
    data[3226] = 0.0 + k[877]*y[IDX_COI];
    data[3227] = 0.0 + k[1448]*y[IDX_COII] - k[1452]*y[IDX_HCOII];
    data[3228] = 0.0 + k[1353]*y[IDX_COI] + k[1355]*y[IDX_H2COI];
    data[3229] = 0.0 + k[1398]*y[IDX_COII] - k[1406]*y[IDX_HCOII];
    data[3230] = 0.0 + k[267]*y[IDX_HCOI];
    data[3231] = 0.0 + k[1423]*y[IDX_COII] - k[1433]*y[IDX_HCOII];
    data[3232] = 0.0 + k[278]*y[IDX_HCOI];
    data[3233] = 0.0 - k[1148]*y[IDX_HCOII];
    data[3234] = 0.0 + k[0]*y[IDX_CHI] + k[750]*y[IDX_CH2II] +
        k[784]*y[IDX_CH3II] + k[1491]*y[IDX_C2HII] + k[1492]*y[IDX_C2H2II] +
        k[1500]*y[IDX_HCO2II] + k[1502]*y[IDX_HCSII];
    data[3235] = 0.0 + k[314]*y[IDX_HCOI] + k[1471]*y[IDX_H2COI] +
        k[1474]*y[IDX_HCNI];
    data[3236] = 0.0 + k[733]*y[IDX_CHII] + k[749]*y[IDX_CH2II] +
        k[967]*y[IDX_H2COII];
    data[3237] = 0.0 + k[204]*y[IDX_HCOI] + k[858]*y[IDX_CHI] +
        k[972]*y[IDX_H2COI] + k[1482]*y[IDX_C2H2I];
    data[3238] = 0.0 + k[878]*y[IDX_COI];
    data[3239] = 0.0 - k[1149]*y[IDX_HCOII];
    data[3240] = 0.0 + k[1539]*y[IDX_COII] - k[1542]*y[IDX_HCOII] -
        k[1543]*y[IDX_HCOII];
    data[3241] = 0.0 + k[334]*y[IDX_HCOI] + k[1521]*y[IDX_COI];
    data[3242] = 0.0 - k[1151]*y[IDX_HCOII];
    data[3243] = 0.0 + k[205]*y[IDX_HCOI] + k[975]*y[IDX_H2COI];
    data[3244] = 0.0 - k[1150]*y[IDX_HCOII];
    data[3245] = 0.0 - k[1566]*y[IDX_HCOII];
    data[3246] = 0.0 - k[1155]*y[IDX_HCOII];
    data[3247] = 0.0 - k[1153]*y[IDX_HCOII];
    data[3248] = 0.0 - k[1154]*y[IDX_HCOII];
    data[3249] = 0.0 + k[880]*y[IDX_COI];
    data[3250] = 0.0 - k[1156]*y[IDX_HCOII];
    data[3251] = 0.0 + k[206]*y[IDX_HCOI] + k[864]*y[IDX_CHI];
    data[3252] = 0.0 - k[1157]*y[IDX_HCOII];
    data[3253] = 0.0 - k[1152]*y[IDX_HCOII];
    data[3254] = 0.0 + k[1557]*y[IDX_C2H2I];
    data[3255] = 0.0 - k[695]*y[IDX_HCO2II];
    data[3256] = 0.0 - k[791]*y[IDX_HCO2II];
    data[3257] = 0.0 - k[812]*y[IDX_HCO2II];
    data[3258] = 0.0 + k[796]*y[IDX_CO2I];
    data[3259] = 0.0 + k[825]*y[IDX_CO2I];
    data[3260] = 0.0 - k[875]*y[IDX_HCO2II];
    data[3261] = 0.0 + k[796]*y[IDX_CH4II] + k[825]*y[IDX_CH5II] +
        k[918]*y[IDX_H2II] + k[1036]*y[IDX_H3II] + k[1110]*y[IDX_HCNII] +
        k[1173]*y[IDX_HNOII] + k[1322]*y[IDX_N2HII] + k[1350]*y[IDX_NHII] +
        k[1489]*y[IDX_O2HII] + k[1520]*y[IDX_OHII];
    data[3262] = 0.0 - k[543]*y[IDX_HCO2II] - k[544]*y[IDX_HCO2II] -
        k[545]*y[IDX_HCO2II];
    data[3263] = 0.0 + k[918]*y[IDX_CO2I];
    data[3264] = 0.0 - k[1004]*y[IDX_HCO2II];
    data[3265] = 0.0 + k[1036]*y[IDX_CO2I];
    data[3266] = 0.0 + k[1110]*y[IDX_CO2I];
    data[3267] = 0.0 + k[1543]*y[IDX_OHI];
    data[3268] = 0.0 - k[543]*y[IDX_EM] - k[544]*y[IDX_EM] -
        k[545]*y[IDX_EM] - k[695]*y[IDX_CI] - k[791]*y[IDX_CH3CNI] -
        k[812]*y[IDX_CH4I] - k[875]*y[IDX_COI] - k[1004]*y[IDX_H2OI] -
        k[1434]*y[IDX_NH3I] - k[1500]*y[IDX_OI] - k[2247];
    data[3269] = 0.0 + k[1238]*y[IDX_HeII];
    data[3270] = 0.0 + k[1238]*y[IDX_HCOOCH3I];
    data[3271] = 0.0 + k[1173]*y[IDX_CO2I];
    data[3272] = 0.0 + k[1322]*y[IDX_CO2I];
    data[3273] = 0.0 + k[1350]*y[IDX_CO2I];
    data[3274] = 0.0 - k[1434]*y[IDX_HCO2II];
    data[3275] = 0.0 - k[1500]*y[IDX_HCO2II];
    data[3276] = 0.0 + k[1489]*y[IDX_CO2I];
    data[3277] = 0.0 + k[1543]*y[IDX_HCOII];
    data[3278] = 0.0 + k[1520]*y[IDX_CO2I];
    data[3279] = 0.0 + k[2479] + k[2480] + k[2481] + k[2482];
    data[3280] = 0.0 + k[537]*y[IDX_H5C2O2II];
    data[3281] = 0.0 - k[1047]*y[IDX_HCOOCH3I];
    data[3282] = 0.0 - k[1089]*y[IDX_HCOOCH3I];
    data[3283] = 0.0 + k[537]*y[IDX_EM];
    data[3284] = 0.0 - k[1145]*y[IDX_HCOOCH3I];
    data[3285] = 0.0 - k[411] - k[1047]*y[IDX_H3II] - k[1089]*y[IDX_H3OII] -
        k[1145]*y[IDX_HCOII] - k[1238]*y[IDX_HeII] - k[2032] - k[2153];
    data[3286] = 0.0 - k[1238]*y[IDX_HCOOCH3I];
    data[3287] = 0.0 + k[1557]*y[IDX_SOII];
    data[3288] = 0.0 + k[1560]*y[IDX_SOII];
    data[3289] = 0.0 + k[1645]*y[IDX_SI];
    data[3290] = 0.0 + k[507]*y[IDX_H2CSII];
    data[3291] = 0.0 - k[1753]*y[IDX_HCSI];
    data[3292] = 0.0 - k[899]*y[IDX_HCSI];
    data[3293] = 0.0 + k[507]*y[IDX_EM];
    data[3294] = 0.0 - k[1048]*y[IDX_HCSI];
    data[3295] = 0.0 - k[412] - k[899]*y[IDX_HII] - k[1048]*y[IDX_H3II] -
        k[1239]*y[IDX_HeII] - k[1240]*y[IDX_HeII] - k[1753]*y[IDX_HI] -
        k[1814]*y[IDX_NI] - k[1894]*y[IDX_OI] - k[1895]*y[IDX_OI] - k[2033] -
        k[2258];
    data[3296] = 0.0 - k[1239]*y[IDX_HCSI] - k[1240]*y[IDX_HCSI];
    data[3297] = 0.0 - k[1814]*y[IDX_HCSI];
    data[3298] = 0.0 - k[1894]*y[IDX_HCSI] - k[1895]*y[IDX_HCSI];
    data[3299] = 0.0 + k[1645]*y[IDX_CH2I];
    data[3300] = 0.0 + k[1557]*y[IDX_C2H2I] + k[1560]*y[IDX_C2H4I];
    data[3301] = 0.0 + k[691]*y[IDX_H2SII];
    data[3302] = 0.0 + k[622]*y[IDX_H2SI];
    data[3303] = 0.0 + k[1558]*y[IDX_SOII];
    data[3304] = 0.0 + k[676]*y[IDX_SII];
    data[3305] = 0.0 - k[1164]*y[IDX_HCSII];
    data[3306] = 0.0 + k[1018]*y[IDX_H2SI];
    data[3307] = 0.0 + k[722]*y[IDX_H2SI] + k[736]*y[IDX_OCSI];
    data[3308] = 0.0 + k[772]*y[IDX_SII];
    data[3309] = 0.0 + k[746]*y[IDX_H2SI] + k[752]*y[IDX_OCSI] +
        k[753]*y[IDX_SI];
    data[3310] = 0.0 + k[787]*y[IDX_SI];
    data[3311] = 0.0 + k[808]*y[IDX_CSII] + k[822]*y[IDX_SII];
    data[3312] = 0.0 + k[1039]*y[IDX_H3II] + k[1085]*y[IDX_H3OII] +
        k[1140]*y[IDX_HCOII];
    data[3313] = 0.0 + k[808]*y[IDX_CH4I] + k[943]*y[IDX_H2I];
    data[3314] = 0.0 - k[546]*y[IDX_HCSII] - k[547]*y[IDX_HCSII];
    data[3315] = 0.0 + k[943]*y[IDX_CSII];
    data[3316] = 0.0 + k[622]*y[IDX_CII] + k[722]*y[IDX_CHII] +
        k[746]*y[IDX_CH2II] + k[1018]*y[IDX_C2NII];
    data[3317] = 0.0 + k[691]*y[IDX_CI];
    data[3318] = 0.0 + k[1039]*y[IDX_CSI];
    data[3319] = 0.0 + k[1085]*y[IDX_CSI];
    data[3320] = 0.0 + k[1140]*y[IDX_CSI];
    data[3321] = 0.0 + k[412] + k[2033];
    data[3322] = 0.0 - k[546]*y[IDX_EM] - k[547]*y[IDX_EM] -
        k[1164]*y[IDX_C2H5OHI] - k[1435]*y[IDX_NH3I] - k[1501]*y[IDX_OI] -
        k[1502]*y[IDX_OI] - k[2268];
    data[3323] = 0.0 - k[1435]*y[IDX_HCSII];
    data[3324] = 0.0 - k[1501]*y[IDX_HCSII] - k[1502]*y[IDX_HCSII];
    data[3325] = 0.0 + k[736]*y[IDX_CHII] + k[752]*y[IDX_CH2II];
    data[3326] = 0.0 + k[753]*y[IDX_CH2II] + k[787]*y[IDX_CH3II];
    data[3327] = 0.0 + k[676]*y[IDX_C2H4I] + k[772]*y[IDX_CH2I] +
        k[822]*y[IDX_CH4I];
    data[3328] = 0.0 + k[1558]*y[IDX_C2H2I];
    data[3329] = 0.0 + k[209]*y[IDX_HeII];
    data[3330] = 0.0 + k[207]*y[IDX_HeII] + k[1177]*y[IDX_HeII];
    data[3331] = 0.0 + k[1186]*y[IDX_HeII] + k[1187]*y[IDX_HeII] +
        k[1188]*y[IDX_HeII];
    data[3332] = 0.0 + k[208]*y[IDX_HeII] + k[1178]*y[IDX_HeII] +
        k[1179]*y[IDX_HeII] + k[1180]*y[IDX_HeII];
    data[3333] = 0.0 + k[1181]*y[IDX_HeII] + k[1182]*y[IDX_HeII];
    data[3334] = 0.0 + k[1183]*y[IDX_HeII] + k[1184]*y[IDX_HeII] +
        k[1185]*y[IDX_HeII];
    data[3335] = 0.0 + k[1189]*y[IDX_HeII];
    data[3336] = 0.0 + k[1190]*y[IDX_HeII];
    data[3337] = 0.0 + k[1191]*y[IDX_HeII];
    data[3338] = 0.0 + k[211]*y[IDX_HeII] + k[1206]*y[IDX_HeII];
    data[3339] = 0.0 + k[1192]*y[IDX_HeII] + k[1193]*y[IDX_HeII];
    data[3340] = 0.0 + k[1194]*y[IDX_HeII] + k[1195]*y[IDX_HeII];
    data[3341] = 0.0 + k[1196]*y[IDX_HeII];
    data[3342] = 0.0 + k[1197]*y[IDX_HeII];
    data[3343] = 0.0 + k[1198]*y[IDX_HeII] + k[1199]*y[IDX_HeII];
    data[3344] = 0.0 + k[1200]*y[IDX_HeII] + k[1201]*y[IDX_HeII];
    data[3345] = 0.0 + k[210]*y[IDX_HeII] + k[1202]*y[IDX_HeII] +
        k[1203]*y[IDX_HeII] + k[1204]*y[IDX_HeII] + k[1205]*y[IDX_HeII];
    data[3346] = 0.0 + k[1207]*y[IDX_HeII] + k[1208]*y[IDX_HeII];
    data[3347] = 0.0 + k[1213]*y[IDX_HeII];
    data[3348] = 0.0 + k[1209]*y[IDX_HeII] + k[1210]*y[IDX_HeII] +
        k[1211]*y[IDX_HeII] + k[1212]*y[IDX_HeII];
    data[3349] = 0.0 + k[1214]*y[IDX_HeII] + k[1215]*y[IDX_HeII];
    data[3350] = 0.0 + k[563]*y[IDX_HeHII] + k[2139]*y[IDX_HeII];
    data[3351] = 0.0 + k[195]*y[IDX_HeII] + k[1106]*y[IDX_HeHII];
    data[3352] = 0.0 - k[2111]*y[IDX_HeI];
    data[3353] = 0.0 + k[172]*y[IDX_HeII] + k[950]*y[IDX_HeII] +
        k[951]*y[IDX_HeHII];
    data[3354] = 0.0 - k[926]*y[IDX_HeI];
    data[3355] = 0.0 + k[212]*y[IDX_HeII] + k[1216]*y[IDX_HeII] +
        k[1217]*y[IDX_HeII] + k[1218]*y[IDX_HeII];
    data[3356] = 0.0 + k[1219]*y[IDX_HeII] + k[1220]*y[IDX_HeII] +
        k[1221]*y[IDX_HeII];
    data[3357] = 0.0 + k[213]*y[IDX_HeII] + k[1222]*y[IDX_HeII] +
        k[1223]*y[IDX_HeII];
    data[3358] = 0.0 + k[214]*y[IDX_HeII] + k[1226]*y[IDX_HeII] +
        k[1227]*y[IDX_HeII];
    data[3359] = 0.0 + k[1224]*y[IDX_HeII] + k[1225]*y[IDX_HeII];
    data[3360] = 0.0 + k[1228]*y[IDX_HeII];
    data[3361] = 0.0 + k[1229]*y[IDX_HeII] + k[1230]*y[IDX_HeII];
    data[3362] = 0.0 + k[1241]*y[IDX_HeII];
    data[3363] = 0.0 + k[1231]*y[IDX_HeII] + k[1232]*y[IDX_HeII] +
        k[1233]*y[IDX_HeII] + k[1234]*y[IDX_HeII];
    data[3364] = 0.0 + k[1235]*y[IDX_HeII] + k[1237]*y[IDX_HeII];
    data[3365] = 0.0 + k[1238]*y[IDX_HeII];
    data[3366] = 0.0 + k[1239]*y[IDX_HeII] + k[1240]*y[IDX_HeII];
    data[3367] = 0.0 - k[363] - k[419] - k[926]*y[IDX_H2II] -
        k[2111]*y[IDX_HII];
    data[3368] = 0.0 + k[172]*y[IDX_H2I] + k[195]*y[IDX_HI] +
        k[207]*y[IDX_C2I] + k[208]*y[IDX_C2H2I] + k[209]*y[IDX_CI] +
        k[210]*y[IDX_CH4I] + k[211]*y[IDX_CHI] + k[212]*y[IDX_H2COI] +
        k[213]*y[IDX_H2OI] + k[214]*y[IDX_H2SI] + k[215]*y[IDX_N2I] +
        k[216]*y[IDX_NH3I] + k[217]*y[IDX_O2I] + k[218]*y[IDX_SO2I] +
        k[219]*y[IDX_SiI] + k[950]*y[IDX_H2I] + k[1177]*y[IDX_C2I] +
        k[1178]*y[IDX_C2H2I] + k[1179]*y[IDX_C2H2I] + k[1180]*y[IDX_C2H2I] +
        k[1181]*y[IDX_C2H3I] + k[1182]*y[IDX_C2H3I] + k[1183]*y[IDX_C2H4I] +
        k[1184]*y[IDX_C2H4I] + k[1185]*y[IDX_C2H4I] + k[1186]*y[IDX_C2HI] +
        k[1187]*y[IDX_C2HI] + k[1188]*y[IDX_C2HI] + k[1189]*y[IDX_C2NI] +
        k[1190]*y[IDX_C3NI] + k[1191]*y[IDX_C4HI] + k[1192]*y[IDX_CH2I] +
        k[1193]*y[IDX_CH2I] + k[1194]*y[IDX_CH2COI] + k[1195]*y[IDX_CH2COI] +
        k[1196]*y[IDX_CH3I] + k[1197]*y[IDX_CH3CCHI] + k[1198]*y[IDX_CH3CNI] +
        k[1199]*y[IDX_CH3CNI] + k[1200]*y[IDX_CH3OHI] + k[1201]*y[IDX_CH3OHI] +
        k[1202]*y[IDX_CH4I] + k[1203]*y[IDX_CH4I] + k[1204]*y[IDX_CH4I] +
        k[1205]*y[IDX_CH4I] + k[1206]*y[IDX_CHI] + k[1207]*y[IDX_CNI] +
        k[1208]*y[IDX_CNI] + k[1209]*y[IDX_CO2I] + k[1210]*y[IDX_CO2I] +
        k[1211]*y[IDX_CO2I] + k[1212]*y[IDX_CO2I] + k[1213]*y[IDX_COI] +
        k[1214]*y[IDX_CSI] + k[1215]*y[IDX_CSI] + k[1216]*y[IDX_H2COI] +
        k[1217]*y[IDX_H2COI] + k[1218]*y[IDX_H2COI] + k[1219]*y[IDX_H2CSI] +
        k[1220]*y[IDX_H2CSI] + k[1221]*y[IDX_H2CSI] + k[1222]*y[IDX_H2OI] +
        k[1223]*y[IDX_H2OI] + k[1224]*y[IDX_H2S2I] + k[1225]*y[IDX_H2S2I] +
        k[1226]*y[IDX_H2SI] + k[1227]*y[IDX_H2SI] + k[1228]*y[IDX_H2SiOI] +
        k[1229]*y[IDX_HC3NI] + k[1230]*y[IDX_HC3NI] + k[1231]*y[IDX_HCNI] +
        k[1232]*y[IDX_HCNI] + k[1233]*y[IDX_HCNI] + k[1234]*y[IDX_HCNI] +
        k[1235]*y[IDX_HCOI] + k[1237]*y[IDX_HCOI] + k[1238]*y[IDX_HCOOCH3I] +
        k[1239]*y[IDX_HCSI] + k[1240]*y[IDX_HCSI] + k[1241]*y[IDX_HClI] +
        k[1242]*y[IDX_HNCI] + k[1243]*y[IDX_HNCI] + k[1244]*y[IDX_HNCI] +
        k[1245]*y[IDX_HNOI] + k[1246]*y[IDX_HNOI] + k[1247]*y[IDX_HS2I] +
        k[1248]*y[IDX_HS2I] + k[1249]*y[IDX_HSI] + k[1250]*y[IDX_N2I] +
        k[1251]*y[IDX_NCCNI] + k[1252]*y[IDX_NH2I] + k[1253]*y[IDX_NH2I] +
        k[1254]*y[IDX_NH3I] + k[1255]*y[IDX_NH3I] + k[1256]*y[IDX_NHI] +
        k[1257]*y[IDX_NOI] + k[1258]*y[IDX_NOI] + k[1259]*y[IDX_NSI] +
        k[1260]*y[IDX_NSI] + k[1261]*y[IDX_O2I] + k[1262]*y[IDX_OCNI] +
        k[1263]*y[IDX_OCNI] + k[1264]*y[IDX_OCSI] + k[1265]*y[IDX_OCSI] +
        k[1266]*y[IDX_OCSI] + k[1267]*y[IDX_OCSI] + k[1268]*y[IDX_OHI] +
        k[1269]*y[IDX_S2I] + k[1270]*y[IDX_SO2I] + k[1271]*y[IDX_SO2I] +
        k[1272]*y[IDX_SOI] + k[1273]*y[IDX_SOI] + k[1274]*y[IDX_SiC2I] +
        k[1275]*y[IDX_SiC3I] + k[1276]*y[IDX_SiCI] + k[1277]*y[IDX_SiCI] +
        k[1278]*y[IDX_SiH2I] + k[1279]*y[IDX_SiH2I] + k[1280]*y[IDX_SiH3I] +
        k[1281]*y[IDX_SiH3I] + k[1282]*y[IDX_SiH4I] + k[1283]*y[IDX_SiH4I] +
        k[1284]*y[IDX_SiHI] + k[1285]*y[IDX_SiOI] + k[1286]*y[IDX_SiOI] +
        k[1287]*y[IDX_SiSI] + k[1288]*y[IDX_SiSI] + k[2139]*y[IDX_EM];
    data[3369] = 0.0 + k[563]*y[IDX_EM] + k[951]*y[IDX_H2I] +
        k[1106]*y[IDX_HI];
    data[3370] = 0.0 + k[1242]*y[IDX_HeII] + k[1243]*y[IDX_HeII] +
        k[1244]*y[IDX_HeII];
    data[3371] = 0.0 + k[1245]*y[IDX_HeII] + k[1246]*y[IDX_HeII];
    data[3372] = 0.0 + k[1249]*y[IDX_HeII];
    data[3373] = 0.0 + k[1247]*y[IDX_HeII] + k[1248]*y[IDX_HeII];
    data[3374] = 0.0 + k[215]*y[IDX_HeII] + k[1250]*y[IDX_HeII];
    data[3375] = 0.0 + k[1251]*y[IDX_HeII];
    data[3376] = 0.0 + k[1256]*y[IDX_HeII];
    data[3377] = 0.0 + k[1252]*y[IDX_HeII] + k[1253]*y[IDX_HeII];
    data[3378] = 0.0 + k[216]*y[IDX_HeII] + k[1254]*y[IDX_HeII] +
        k[1255]*y[IDX_HeII];
    data[3379] = 0.0 + k[1257]*y[IDX_HeII] + k[1258]*y[IDX_HeII];
    data[3380] = 0.0 + k[1259]*y[IDX_HeII] + k[1260]*y[IDX_HeII];
    data[3381] = 0.0 + k[217]*y[IDX_HeII] + k[1261]*y[IDX_HeII];
    data[3382] = 0.0 + k[1262]*y[IDX_HeII] + k[1263]*y[IDX_HeII];
    data[3383] = 0.0 + k[1264]*y[IDX_HeII] + k[1265]*y[IDX_HeII] +
        k[1266]*y[IDX_HeII] + k[1267]*y[IDX_HeII];
    data[3384] = 0.0 + k[1268]*y[IDX_HeII];
    data[3385] = 0.0 + k[1269]*y[IDX_HeII];
    data[3386] = 0.0 + k[219]*y[IDX_HeII];
    data[3387] = 0.0 + k[1276]*y[IDX_HeII] + k[1277]*y[IDX_HeII];
    data[3388] = 0.0 + k[1274]*y[IDX_HeII];
    data[3389] = 0.0 + k[1275]*y[IDX_HeII];
    data[3390] = 0.0 + k[1284]*y[IDX_HeII];
    data[3391] = 0.0 + k[1278]*y[IDX_HeII] + k[1279]*y[IDX_HeII];
    data[3392] = 0.0 + k[1280]*y[IDX_HeII] + k[1281]*y[IDX_HeII];
    data[3393] = 0.0 + k[1282]*y[IDX_HeII] + k[1283]*y[IDX_HeII];
    data[3394] = 0.0 + k[1285]*y[IDX_HeII] + k[1286]*y[IDX_HeII];
    data[3395] = 0.0 + k[1287]*y[IDX_HeII] + k[1288]*y[IDX_HeII];
    data[3396] = 0.0 + k[1272]*y[IDX_HeII] + k[1273]*y[IDX_HeII];
    data[3397] = 0.0 + k[218]*y[IDX_HeII] + k[1270]*y[IDX_HeII] +
        k[1271]*y[IDX_HeII];
    data[3398] = 0.0 - k[209]*y[IDX_HeII];
    data[3399] = 0.0 - k[207]*y[IDX_HeII] - k[1177]*y[IDX_HeII];
    data[3400] = 0.0 - k[1186]*y[IDX_HeII] - k[1187]*y[IDX_HeII] -
        k[1188]*y[IDX_HeII];
    data[3401] = 0.0 - k[208]*y[IDX_HeII] - k[1178]*y[IDX_HeII] -
        k[1179]*y[IDX_HeII] - k[1180]*y[IDX_HeII];
    data[3402] = 0.0 - k[1181]*y[IDX_HeII] - k[1182]*y[IDX_HeII];
    data[3403] = 0.0 - k[1183]*y[IDX_HeII] - k[1184]*y[IDX_HeII] -
        k[1185]*y[IDX_HeII];
    data[3404] = 0.0 - k[1189]*y[IDX_HeII];
    data[3405] = 0.0 - k[1190]*y[IDX_HeII];
    data[3406] = 0.0 - k[1191]*y[IDX_HeII];
    data[3407] = 0.0 - k[211]*y[IDX_HeII] - k[1206]*y[IDX_HeII];
    data[3408] = 0.0 - k[1192]*y[IDX_HeII] - k[1193]*y[IDX_HeII];
    data[3409] = 0.0 - k[1194]*y[IDX_HeII] - k[1195]*y[IDX_HeII];
    data[3410] = 0.0 - k[1196]*y[IDX_HeII];
    data[3411] = 0.0 - k[1197]*y[IDX_HeII];
    data[3412] = 0.0 - k[1198]*y[IDX_HeII] - k[1199]*y[IDX_HeII];
    data[3413] = 0.0 - k[1200]*y[IDX_HeII] - k[1201]*y[IDX_HeII];
    data[3414] = 0.0 - k[210]*y[IDX_HeII] - k[1202]*y[IDX_HeII] -
        k[1203]*y[IDX_HeII] - k[1204]*y[IDX_HeII] - k[1205]*y[IDX_HeII];
    data[3415] = 0.0 - k[1207]*y[IDX_HeII] - k[1208]*y[IDX_HeII];
    data[3416] = 0.0 - k[1213]*y[IDX_HeII];
    data[3417] = 0.0 - k[1209]*y[IDX_HeII] - k[1210]*y[IDX_HeII] -
        k[1211]*y[IDX_HeII] - k[1212]*y[IDX_HeII];
    data[3418] = 0.0 - k[1214]*y[IDX_HeII] - k[1215]*y[IDX_HeII];
    data[3419] = 0.0 - k[2139]*y[IDX_HeII];
    data[3420] = 0.0 - k[195]*y[IDX_HeII];
    data[3421] = 0.0 - k[172]*y[IDX_HeII] - k[950]*y[IDX_HeII];
    data[3422] = 0.0 - k[212]*y[IDX_HeII] - k[1216]*y[IDX_HeII] -
        k[1217]*y[IDX_HeII] - k[1218]*y[IDX_HeII];
    data[3423] = 0.0 - k[1219]*y[IDX_HeII] - k[1220]*y[IDX_HeII] -
        k[1221]*y[IDX_HeII];
    data[3424] = 0.0 - k[213]*y[IDX_HeII] - k[1222]*y[IDX_HeII] -
        k[1223]*y[IDX_HeII];
    data[3425] = 0.0 - k[214]*y[IDX_HeII] - k[1226]*y[IDX_HeII] -
        k[1227]*y[IDX_HeII];
    data[3426] = 0.0 - k[1224]*y[IDX_HeII] - k[1225]*y[IDX_HeII];
    data[3427] = 0.0 - k[1228]*y[IDX_HeII];
    data[3428] = 0.0 - k[1229]*y[IDX_HeII] - k[1230]*y[IDX_HeII];
    data[3429] = 0.0 - k[1241]*y[IDX_HeII];
    data[3430] = 0.0 - k[1231]*y[IDX_HeII] - k[1232]*y[IDX_HeII] -
        k[1233]*y[IDX_HeII] - k[1234]*y[IDX_HeII];
    data[3431] = 0.0 - k[1235]*y[IDX_HeII] - k[1236]*y[IDX_HeII] -
        k[1237]*y[IDX_HeII];
    data[3432] = 0.0 - k[1238]*y[IDX_HeII];
    data[3433] = 0.0 - k[1239]*y[IDX_HeII] - k[1240]*y[IDX_HeII];
    data[3434] = 0.0 + k[363] + k[419];
    data[3435] = 0.0 - k[172]*y[IDX_H2I] - k[195]*y[IDX_HI] -
        k[207]*y[IDX_C2I] - k[208]*y[IDX_C2H2I] - k[209]*y[IDX_CI] -
        k[210]*y[IDX_CH4I] - k[211]*y[IDX_CHI] - k[212]*y[IDX_H2COI] -
        k[213]*y[IDX_H2OI] - k[214]*y[IDX_H2SI] - k[215]*y[IDX_N2I] -
        k[216]*y[IDX_NH3I] - k[217]*y[IDX_O2I] - k[218]*y[IDX_SO2I] -
        k[219]*y[IDX_SiI] - k[950]*y[IDX_H2I] - k[1177]*y[IDX_C2I] -
        k[1178]*y[IDX_C2H2I] - k[1179]*y[IDX_C2H2I] - k[1180]*y[IDX_C2H2I] -
        k[1181]*y[IDX_C2H3I] - k[1182]*y[IDX_C2H3I] - k[1183]*y[IDX_C2H4I] -
        k[1184]*y[IDX_C2H4I] - k[1185]*y[IDX_C2H4I] - k[1186]*y[IDX_C2HI] -
        k[1187]*y[IDX_C2HI] - k[1188]*y[IDX_C2HI] - k[1189]*y[IDX_C2NI] -
        k[1190]*y[IDX_C3NI] - k[1191]*y[IDX_C4HI] - k[1192]*y[IDX_CH2I] -
        k[1193]*y[IDX_CH2I] - k[1194]*y[IDX_CH2COI] - k[1195]*y[IDX_CH2COI] -
        k[1196]*y[IDX_CH3I] - k[1197]*y[IDX_CH3CCHI] - k[1198]*y[IDX_CH3CNI] -
        k[1199]*y[IDX_CH3CNI] - k[1200]*y[IDX_CH3OHI] - k[1201]*y[IDX_CH3OHI] -
        k[1202]*y[IDX_CH4I] - k[1203]*y[IDX_CH4I] - k[1204]*y[IDX_CH4I] -
        k[1205]*y[IDX_CH4I] - k[1206]*y[IDX_CHI] - k[1207]*y[IDX_CNI] -
        k[1208]*y[IDX_CNI] - k[1209]*y[IDX_CO2I] - k[1210]*y[IDX_CO2I] -
        k[1211]*y[IDX_CO2I] - k[1212]*y[IDX_CO2I] - k[1213]*y[IDX_COI] -
        k[1214]*y[IDX_CSI] - k[1215]*y[IDX_CSI] - k[1216]*y[IDX_H2COI] -
        k[1217]*y[IDX_H2COI] - k[1218]*y[IDX_H2COI] - k[1219]*y[IDX_H2CSI] -
        k[1220]*y[IDX_H2CSI] - k[1221]*y[IDX_H2CSI] - k[1222]*y[IDX_H2OI] -
        k[1223]*y[IDX_H2OI] - k[1224]*y[IDX_H2S2I] - k[1225]*y[IDX_H2S2I] -
        k[1226]*y[IDX_H2SI] - k[1227]*y[IDX_H2SI] - k[1228]*y[IDX_H2SiOI] -
        k[1229]*y[IDX_HC3NI] - k[1230]*y[IDX_HC3NI] - k[1231]*y[IDX_HCNI] -
        k[1232]*y[IDX_HCNI] - k[1233]*y[IDX_HCNI] - k[1234]*y[IDX_HCNI] -
        k[1235]*y[IDX_HCOI] - k[1236]*y[IDX_HCOI] - k[1237]*y[IDX_HCOI] -
        k[1238]*y[IDX_HCOOCH3I] - k[1239]*y[IDX_HCSI] - k[1240]*y[IDX_HCSI] -
        k[1241]*y[IDX_HClI] - k[1242]*y[IDX_HNCI] - k[1243]*y[IDX_HNCI] -
        k[1244]*y[IDX_HNCI] - k[1245]*y[IDX_HNOI] - k[1246]*y[IDX_HNOI] -
        k[1247]*y[IDX_HS2I] - k[1248]*y[IDX_HS2I] - k[1249]*y[IDX_HSI] -
        k[1250]*y[IDX_N2I] - k[1251]*y[IDX_NCCNI] - k[1252]*y[IDX_NH2I] -
        k[1253]*y[IDX_NH2I] - k[1254]*y[IDX_NH3I] - k[1255]*y[IDX_NH3I] -
        k[1256]*y[IDX_NHI] - k[1257]*y[IDX_NOI] - k[1258]*y[IDX_NOI] -
        k[1259]*y[IDX_NSI] - k[1260]*y[IDX_NSI] - k[1261]*y[IDX_O2I] -
        k[1262]*y[IDX_OCNI] - k[1263]*y[IDX_OCNI] - k[1264]*y[IDX_OCSI] -
        k[1265]*y[IDX_OCSI] - k[1266]*y[IDX_OCSI] - k[1267]*y[IDX_OCSI] -
        k[1268]*y[IDX_OHI] - k[1269]*y[IDX_S2I] - k[1270]*y[IDX_SO2I] -
        k[1271]*y[IDX_SO2I] - k[1272]*y[IDX_SOI] - k[1273]*y[IDX_SOI] -
        k[1274]*y[IDX_SiC2I] - k[1275]*y[IDX_SiC3I] - k[1276]*y[IDX_SiCI] -
        k[1277]*y[IDX_SiCI] - k[1278]*y[IDX_SiH2I] - k[1279]*y[IDX_SiH2I] -
        k[1280]*y[IDX_SiH3I] - k[1281]*y[IDX_SiH3I] - k[1282]*y[IDX_SiH4I] -
        k[1283]*y[IDX_SiH4I] - k[1284]*y[IDX_SiHI] - k[1285]*y[IDX_SiOI] -
        k[1286]*y[IDX_SiOI] - k[1287]*y[IDX_SiSI] - k[1288]*y[IDX_SiSI] -
        k[2139]*y[IDX_EM];
    data[3436] = 0.0 - k[1242]*y[IDX_HeII] - k[1243]*y[IDX_HeII] -
        k[1244]*y[IDX_HeII];
    data[3437] = 0.0 - k[1245]*y[IDX_HeII] - k[1246]*y[IDX_HeII];
    data[3438] = 0.0 - k[1249]*y[IDX_HeII];
    data[3439] = 0.0 - k[1247]*y[IDX_HeII] - k[1248]*y[IDX_HeII];
    data[3440] = 0.0 - k[215]*y[IDX_HeII] - k[1250]*y[IDX_HeII];
    data[3441] = 0.0 - k[1251]*y[IDX_HeII];
    data[3442] = 0.0 - k[1256]*y[IDX_HeII];
    data[3443] = 0.0 - k[1252]*y[IDX_HeII] - k[1253]*y[IDX_HeII];
    data[3444] = 0.0 - k[216]*y[IDX_HeII] - k[1254]*y[IDX_HeII] -
        k[1255]*y[IDX_HeII];
    data[3445] = 0.0 - k[1257]*y[IDX_HeII] - k[1258]*y[IDX_HeII];
    data[3446] = 0.0 - k[1259]*y[IDX_HeII] - k[1260]*y[IDX_HeII];
    data[3447] = 0.0 - k[217]*y[IDX_HeII] - k[1261]*y[IDX_HeII];
    data[3448] = 0.0 - k[1262]*y[IDX_HeII] - k[1263]*y[IDX_HeII];
    data[3449] = 0.0 - k[1264]*y[IDX_HeII] - k[1265]*y[IDX_HeII] -
        k[1266]*y[IDX_HeII] - k[1267]*y[IDX_HeII];
    data[3450] = 0.0 - k[1268]*y[IDX_HeII];
    data[3451] = 0.0 - k[1269]*y[IDX_HeII];
    data[3452] = 0.0 - k[219]*y[IDX_HeII];
    data[3453] = 0.0 - k[1276]*y[IDX_HeII] - k[1277]*y[IDX_HeII];
    data[3454] = 0.0 - k[1274]*y[IDX_HeII];
    data[3455] = 0.0 - k[1275]*y[IDX_HeII];
    data[3456] = 0.0 - k[1284]*y[IDX_HeII];
    data[3457] = 0.0 - k[1278]*y[IDX_HeII] - k[1279]*y[IDX_HeII];
    data[3458] = 0.0 - k[1280]*y[IDX_HeII] - k[1281]*y[IDX_HeII];
    data[3459] = 0.0 - k[1282]*y[IDX_HeII] - k[1283]*y[IDX_HeII];
    data[3460] = 0.0 - k[1285]*y[IDX_HeII] - k[1286]*y[IDX_HeII];
    data[3461] = 0.0 - k[1287]*y[IDX_HeII] - k[1288]*y[IDX_HeII];
    data[3462] = 0.0 - k[1272]*y[IDX_HeII] - k[1273]*y[IDX_HeII];
    data[3463] = 0.0 - k[218]*y[IDX_HeII] - k[1270]*y[IDX_HeII] -
        k[1271]*y[IDX_HeII];
    data[3464] = 0.0 - k[563]*y[IDX_HeHII];
    data[3465] = 0.0 - k[1106]*y[IDX_HeHII];
    data[3466] = 0.0 + k[2111]*y[IDX_HeI];
    data[3467] = 0.0 - k[951]*y[IDX_HeHII];
    data[3468] = 0.0 + k[926]*y[IDX_HeI];
    data[3469] = 0.0 + k[1236]*y[IDX_HeII];
    data[3470] = 0.0 + k[926]*y[IDX_H2II] + k[2111]*y[IDX_HII];
    data[3471] = 0.0 + k[1236]*y[IDX_HCOI];
    data[3472] = 0.0 - k[563]*y[IDX_EM] - k[951]*y[IDX_H2I] -
        k[1106]*y[IDX_HI];
    data[3473] = 0.0 + k[2335] + k[2336] + k[2337] + k[2338];
    data[3474] = 0.0 + k[1600]*y[IDX_NH2I] + k[1788]*y[IDX_HNCOI];
    data[3475] = 0.0 - k[627]*y[IDX_HNCI];
    data[3476] = 0.0 - k[1579]*y[IDX_HNCI];
    data[3477] = 0.0 - k[664]*y[IDX_HNCI];
    data[3478] = 0.0 - k[669]*y[IDX_HNCI];
    data[3479] = 0.0 + k[848]*y[IDX_HCNHII];
    data[3480] = 0.0 - k[727]*y[IDX_HNCI];
    data[3481] = 0.0 + k[762]*y[IDX_HCNHII] + k[1802]*y[IDX_NI];
    data[3482] = 0.0 + k[1131]*y[IDX_HCNHII];
    data[3483] = 0.0 + k[485]*y[IDX_EM];
    data[3484] = 0.0 - k[833]*y[IDX_HNCI];
    data[3485] = 0.0 - k[1707]*y[IDX_HNCI];
    data[3486] = 0.0 + k[485]*y[IDX_CH3CNHII] + k[541]*y[IDX_HCNHII];
    data[3487] = 0.0 - k[1754]*y[IDX_HNCI];
    data[3488] = 0.0 - k[1]*y[IDX_HNCI];
    data[3489] = 0.0 + k[1133]*y[IDX_HCNHII];
    data[3490] = 0.0 - k[1165]*y[IDX_HNCI];
    data[3491] = 0.0 - k[986]*y[IDX_HNCI];
    data[3492] = 0.0 + k[1135]*y[IDX_HCNHII];
    data[3493] = 0.0 - k[1050]*y[IDX_HNCI];
    data[3494] = 0.0 - k[1166]*y[IDX_HNCI];
    data[3495] = 0.0 - k[1090]*y[IDX_HNCI];
    data[3496] = 0.0 - k[1167]*y[IDX_HNCI];
    data[3497] = 0.0 - k[1116]*y[IDX_HNCI];
    data[3498] = 0.0 + k[541]*y[IDX_EM] + k[762]*y[IDX_CH2I] +
        k[848]*y[IDX_CHI] + k[1131]*y[IDX_CH3CNI] + k[1133]*y[IDX_H2COI] +
        k[1135]*y[IDX_H2SI] + k[1405]*y[IDX_NH2I] + k[1432]*y[IDX_NH3I];
    data[3499] = 0.0 - k[1168]*y[IDX_HNCI];
    data[3500] = 0.0 - k[1242]*y[IDX_HNCI] - k[1243]*y[IDX_HNCI] -
        k[1244]*y[IDX_HNCI];
    data[3501] = 0.0 - k[1]*y[IDX_HII] - k[414] - k[627]*y[IDX_CII] -
        k[664]*y[IDX_C2HII] - k[669]*y[IDX_C2H2II] - k[727]*y[IDX_CHII] -
        k[833]*y[IDX_CH5II] - k[986]*y[IDX_H2OII] - k[1050]*y[IDX_H3II] -
        k[1090]*y[IDX_H3OII] - k[1116]*y[IDX_HCNII] - k[1165]*y[IDX_H2COII] -
        k[1166]*y[IDX_H3COII] - k[1167]*y[IDX_H3SII] - k[1168]*y[IDX_HCOII] -
        k[1169]*y[IDX_HNOII] - k[1170]*y[IDX_HSII] - k[1171]*y[IDX_N2HII] -
        k[1172]*y[IDX_O2HII] - k[1242]*y[IDX_HeII] - k[1243]*y[IDX_HeII] -
        k[1244]*y[IDX_HeII] - k[1362]*y[IDX_NHII] - k[1387]*y[IDX_NH2II] -
        k[1528]*y[IDX_OHII] - k[1579]*y[IDX_C2HI] - k[1707]*y[IDX_CNI] -
        k[1754]*y[IDX_HI] - k[2036] - k[2297];
    data[3502] = 0.0 + k[1788]*y[IDX_CI];
    data[3503] = 0.0 - k[1169]*y[IDX_HNCI];
    data[3504] = 0.0 - k[1170]*y[IDX_HNCI];
    data[3505] = 0.0 + k[1802]*y[IDX_CH2I];
    data[3506] = 0.0 - k[1171]*y[IDX_HNCI];
    data[3507] = 0.0 - k[1362]*y[IDX_HNCI];
    data[3508] = 0.0 + k[1405]*y[IDX_HCNHII] + k[1600]*y[IDX_CI];
    data[3509] = 0.0 - k[1387]*y[IDX_HNCI];
    data[3510] = 0.0 + k[1432]*y[IDX_HCNHII];
    data[3511] = 0.0 - k[1172]*y[IDX_HNCI];
    data[3512] = 0.0 - k[1528]*y[IDX_HNCI];
    data[3513] = 0.0 + k[2423] + k[2424] + k[2425] + k[2426];
    data[3514] = 0.0 - k[1788]*y[IDX_HNCOI];
    data[3515] = 0.0 + k[1631]*y[IDX_NOI];
    data[3516] = 0.0 - k[900]*y[IDX_HNCOI];
    data[3517] = 0.0 - k[415] - k[900]*y[IDX_HII] - k[1788]*y[IDX_CI] -
        k[2037] - k[2157];
    data[3518] = 0.0 + k[1631]*y[IDX_CH2I];
    data[3519] = 0.0 + k[2371] + k[2372] + k[2373] + k[2374];
    data[3520] = 0.0 - k[1680]*y[IDX_HNOI];
    data[3521] = 0.0 - k[1626]*y[IDX_HNOI];
    data[3522] = 0.0 - k[1655]*y[IDX_HNOI] + k[1658]*y[IDX_NO2I];
    data[3523] = 0.0 - k[1708]*y[IDX_HNOI];
    data[3524] = 0.0 - k[1716]*y[IDX_HNOI];
    data[3525] = 0.0 + k[510]*y[IDX_H2NOII];
    data[3526] = 0.0 - k[1755]*y[IDX_HNOI] - k[1756]*y[IDX_HNOI] -
        k[1757]*y[IDX_HNOI];
    data[3527] = 0.0 - k[901]*y[IDX_HNOI];
    data[3528] = 0.0 + k[510]*y[IDX_EM];
    data[3529] = 0.0 - k[1051]*y[IDX_HNOI];
    data[3530] = 0.0 - k[1782]*y[IDX_HNOI] + k[1783]*y[IDX_NOI];
    data[3531] = 0.0 - k[1245]*y[IDX_HNOI] - k[1246]*y[IDX_HNOI];
    data[3532] = 0.0 - k[416] - k[901]*y[IDX_HII] - k[1051]*y[IDX_H3II] -
        k[1245]*y[IDX_HeII] - k[1246]*y[IDX_HeII] - k[1626]*y[IDX_CH2I] -
        k[1655]*y[IDX_CH3I] - k[1680]*y[IDX_CHI] - k[1708]*y[IDX_CNI] -
        k[1716]*y[IDX_COI] - k[1755]*y[IDX_HI] - k[1756]*y[IDX_HI] -
        k[1757]*y[IDX_HI] - k[1782]*y[IDX_HCOI] - k[1815]*y[IDX_NI] -
        k[1896]*y[IDX_OI] - k[1897]*y[IDX_OI] - k[1898]*y[IDX_OI] -
        k[1942]*y[IDX_OHI] - k[2038] - k[2271];
    data[3533] = 0.0 + k[300]*y[IDX_NOI];
    data[3534] = 0.0 - k[1815]*y[IDX_HNOI];
    data[3535] = 0.0 + k[1846]*y[IDX_NO2I] + k[1849]*y[IDX_O2I] +
        k[1854]*y[IDX_OHI];
    data[3536] = 0.0 + k[1902]*y[IDX_OI];
    data[3537] = 0.0 + k[300]*y[IDX_HNOII] + k[1783]*y[IDX_HCOI];
    data[3538] = 0.0 + k[1658]*y[IDX_CH3I] + k[1846]*y[IDX_NHI];
    data[3539] = 0.0 - k[1896]*y[IDX_HNOI] - k[1897]*y[IDX_HNOI] -
        k[1898]*y[IDX_HNOI] + k[1902]*y[IDX_NH2I];
    data[3540] = 0.0 + k[1849]*y[IDX_NHI];
    data[3541] = 0.0 + k[1854]*y[IDX_NHI] - k[1942]*y[IDX_HNOI];
    data[3542] = 0.0 - k[696]*y[IDX_HNOII];
    data[3543] = 0.0 - k[654]*y[IDX_HNOII];
    data[3544] = 0.0 - k[681]*y[IDX_HNOII];
    data[3545] = 0.0 - k[850]*y[IDX_HNOII];
    data[3546] = 0.0 - k[764]*y[IDX_HNOII];
    data[3547] = 0.0 - k[813]*y[IDX_HNOII];
    data[3548] = 0.0 - k[869]*y[IDX_HNOII];
    data[3549] = 0.0 - k[876]*y[IDX_HNOII];
    data[3550] = 0.0 - k[1173]*y[IDX_HNOII] + k[1351]*y[IDX_NHII];
    data[3551] = 0.0 - k[549]*y[IDX_HNOII];
    data[3552] = 0.0 + k[930]*y[IDX_NOI];
    data[3553] = 0.0 - k[971]*y[IDX_HNOII];
    data[3554] = 0.0 - k[1005]*y[IDX_HNOII] + k[1357]*y[IDX_NHII];
    data[3555] = 0.0 + k[1333]*y[IDX_NI];
    data[3556] = 0.0 + k[1060]*y[IDX_NOI];
    data[3557] = 0.0 - k[1125]*y[IDX_HNOII];
    data[3558] = 0.0 - k[1159]*y[IDX_HNOII];
    data[3559] = 0.0 - k[1169]*y[IDX_HNOII];
    data[3560] = 0.0 - k[300]*y[IDX_NOI] - k[549]*y[IDX_EM] -
        k[654]*y[IDX_C2I] - k[681]*y[IDX_C2HI] - k[696]*y[IDX_CI] -
        k[764]*y[IDX_CH2I] - k[813]*y[IDX_CH4I] - k[850]*y[IDX_CHI] -
        k[869]*y[IDX_CNI] - k[876]*y[IDX_COI] - k[971]*y[IDX_H2COI] -
        k[1005]*y[IDX_H2OI] - k[1125]*y[IDX_HCNI] - k[1159]*y[IDX_HCOI] -
        k[1169]*y[IDX_HNCI] - k[1173]*y[IDX_CO2I] - k[1174]*y[IDX_SI] -
        k[1319]*y[IDX_N2I] - k[1407]*y[IDX_NH2I] - k[1436]*y[IDX_NH3I] -
        k[1453]*y[IDX_NHI] - k[1544]*y[IDX_OHI] - k[2272];
    data[3561] = 0.0 + k[1333]*y[IDX_H2OII];
    data[3562] = 0.0 - k[1319]*y[IDX_HNOII];
    data[3563] = 0.0 - k[1453]*y[IDX_HNOII] + k[1458]*y[IDX_O2II];
    data[3564] = 0.0 + k[1351]*y[IDX_CO2I] + k[1357]*y[IDX_H2OI];
    data[3565] = 0.0 - k[1407]*y[IDX_HNOII];
    data[3566] = 0.0 + k[1391]*y[IDX_O2I] + k[1507]*y[IDX_OI];
    data[3567] = 0.0 - k[1436]*y[IDX_HNOII];
    data[3568] = 0.0 + k[1508]*y[IDX_OI];
    data[3569] = 0.0 - k[300]*y[IDX_HNOII] + k[930]*y[IDX_H2II] +
        k[1060]*y[IDX_H3II] + k[1462]*y[IDX_O2HII] + k[1531]*y[IDX_OHII];
    data[3570] = 0.0 + k[1507]*y[IDX_NH2II] + k[1508]*y[IDX_NH3II];
    data[3571] = 0.0 + k[1391]*y[IDX_NH2II];
    data[3572] = 0.0 + k[1458]*y[IDX_NHI];
    data[3573] = 0.0 + k[1462]*y[IDX_NOI];
    data[3574] = 0.0 - k[1544]*y[IDX_HNOII];
    data[3575] = 0.0 + k[1531]*y[IDX_NOI];
    data[3576] = 0.0 - k[1174]*y[IDX_HNOII];
    data[3577] = 0.0 - k[550]*y[IDX_HNSII];
    data[3578] = 0.0 + k[1061]*y[IDX_NSI];
    data[3579] = 0.0 + k[1148]*y[IDX_NSI];
    data[3580] = 0.0 - k[550]*y[IDX_EM] - k[2291];
    data[3581] = 0.0 + k[1392]*y[IDX_SI];
    data[3582] = 0.0 + k[1061]*y[IDX_H3II] + k[1148]*y[IDX_HCOII];
    data[3583] = 0.0 + k[1392]*y[IDX_NH2II];
    data[3584] = 0.0 + k[621]*y[IDX_H2OI];
    data[3585] = 0.0 + k[1038]*y[IDX_H3II];
    data[3586] = 0.0 + k[942]*y[IDX_H2I];
    data[3587] = 0.0 - k[551]*y[IDX_HOCII];
    data[3588] = 0.0 - k[5]*y[IDX_HOCII] + k[942]*y[IDX_COII];
    data[3589] = 0.0 + k[621]*y[IDX_CII];
    data[3590] = 0.0 + k[1038]*y[IDX_COI];
    data[3591] = 0.0 - k[5]*y[IDX_H2I] - k[551]*y[IDX_EM] - k[2168];
    data[3592] = 0.0 + k[737]*y[IDX_OCSI];
    data[3593] = 0.0 + k[788]*y[IDX_SOI];
    data[3594] = 0.0 + k[802]*y[IDX_OCSI];
    data[3595] = 0.0 - k[552]*y[IDX_HOCSII] - k[553]*y[IDX_HOCSII];
    data[3596] = 0.0 - k[1006]*y[IDX_HOCSII];
    data[3597] = 0.0 + k[1065]*y[IDX_OCSI];
    data[3598] = 0.0 + k[1149]*y[IDX_OCSI];
    data[3599] = 0.0 - k[552]*y[IDX_EM] - k[553]*y[IDX_EM] -
        k[1006]*y[IDX_H2OI] - k[2285];
    data[3600] = 0.0 + k[737]*y[IDX_CHII] + k[802]*y[IDX_CH4II] +
        k[1065]*y[IDX_H3II] + k[1149]*y[IDX_HCOII];
    data[3601] = 0.0 + k[788]*y[IDX_CH3II];
    data[3602] = 0.0 - k[1595]*y[IDX_HSI] - k[1596]*y[IDX_HSI];
    data[3603] = 0.0 - k[628]*y[IDX_HSI];
    data[3604] = 0.0 + k[1698]*y[IDX_SI] + k[1699]*y[IDX_SOI];
    data[3605] = 0.0 + k[1653]*y[IDX_H2SI];
    data[3606] = 0.0 - k[780]*y[IDX_HSI];
    data[3607] = 0.0 + k[1673]*y[IDX_SI];
    data[3608] = 0.0 + k[872]*y[IDX_H2SI];
    data[3609] = 0.0 + k[1935]*y[IDX_OHI];
    data[3610] = 0.0 + k[515]*y[IDX_H2SII] + k[518]*y[IDX_H2S2II] +
        k[518]*y[IDX_H2S2II] + k[533]*y[IDX_H3SII] + k[534]*y[IDX_H3SII] +
        k[555]*y[IDX_HS2II] + k[561]*y[IDX_HSiSII];
    data[3611] = 0.0 + k[1109]*y[IDX_SiSII] + k[1749]*y[IDX_H2SI] -
        k[1758]*y[IDX_HSI] + k[1766]*y[IDX_NSI] + k[1775]*y[IDX_OCSI] +
        k[1777]*y[IDX_S2I] + k[1778]*y[IDX_SOI];
    data[3612] = 0.0 - k[128]*y[IDX_HSI] - k[902]*y[IDX_HSI];
    data[3613] = 0.0 - k[1727]*y[IDX_HSI] + k[1735]*y[IDX_SI];
    data[3614] = 0.0 + k[975]*y[IDX_SII];
    data[3615] = 0.0 + k[1000]*y[IDX_H2SII];
    data[3616] = 0.0 + k[982]*y[IDX_H2SI];
    data[3617] = 0.0 + k[872]*y[IDX_COII] + k[982]*y[IDX_H2OII] +
        k[1017]*y[IDX_H2SII] + k[1301]*y[IDX_NII] + k[1383]*y[IDX_NH2II] +
        k[1415]*y[IDX_NH3II] + k[1653]*y[IDX_CH3I] + k[1749]*y[IDX_HI] +
        k[1888]*y[IDX_OI] + k[1938]*y[IDX_OHI] + k[2021];
    data[3618] = 0.0 + k[515]*y[IDX_EM] + k[1000]*y[IDX_H2OI] +
        k[1017]*y[IDX_H2SI] + k[1426]*y[IDX_NH3I];
    data[3619] = 0.0 + k[402] + k[402] + k[1224]*y[IDX_HeII] + k[2019] +
        k[2019];
    data[3620] = 0.0 + k[518]*y[IDX_EM] + k[518]*y[IDX_EM];
    data[3621] = 0.0 - k[1053]*y[IDX_HSI];
    data[3622] = 0.0 + k[533]*y[IDX_EM] + k[534]*y[IDX_EM];
    data[3623] = 0.0 + k[1952]*y[IDX_SI];
    data[3624] = 0.0 - k[1147]*y[IDX_HSI];
    data[3625] = 0.0 + k[1894]*y[IDX_OI];
    data[3626] = 0.0 + k[1224]*y[IDX_H2S2I] + k[1247]*y[IDX_HS2I] -
        k[1249]*y[IDX_HSI];
    data[3627] = 0.0 - k[128]*y[IDX_HII] - k[418] - k[628]*y[IDX_CII] -
        k[780]*y[IDX_CH3II] - k[902]*y[IDX_HII] - k[1053]*y[IDX_H3II] -
        k[1147]*y[IDX_HCOII] - k[1249]*y[IDX_HeII] - k[1595]*y[IDX_CI] -
        k[1596]*y[IDX_CI] - k[1727]*y[IDX_H2I] - k[1758]*y[IDX_HI] -
        k[1789]*y[IDX_HSI] - k[1789]*y[IDX_HSI] - k[1789]*y[IDX_HSI] -
        k[1789]*y[IDX_HSI] - k[1816]*y[IDX_NI] - k[1817]*y[IDX_NI] -
        k[1899]*y[IDX_OI] - k[1900]*y[IDX_OI] - k[1953]*y[IDX_SI] - k[2043] -
        k[2254];
    data[3628] = 0.0 + k[225]*y[IDX_MgI] + k[288]*y[IDX_NH3I] +
        k[301]*y[IDX_NOI] + k[347]*y[IDX_SI] + k[351]*y[IDX_SiI];
    data[3629] = 0.0 + k[417] + k[1247]*y[IDX_HeII] + k[2042];
    data[3630] = 0.0 + k[555]*y[IDX_EM];
    data[3631] = 0.0 + k[561]*y[IDX_EM];
    data[3632] = 0.0 + k[225]*y[IDX_HSII];
    data[3633] = 0.0 - k[1816]*y[IDX_HSI] - k[1817]*y[IDX_HSI];
    data[3634] = 0.0 + k[1301]*y[IDX_H2SI];
    data[3635] = 0.0 + k[1856]*y[IDX_SI];
    data[3636] = 0.0 + k[1383]*y[IDX_H2SI];
    data[3637] = 0.0 + k[288]*y[IDX_HSII] + k[1426]*y[IDX_H2SII];
    data[3638] = 0.0 + k[1415]*y[IDX_H2SI];
    data[3639] = 0.0 + k[301]*y[IDX_HSII];
    data[3640] = 0.0 + k[1766]*y[IDX_HI];
    data[3641] = 0.0 + k[1888]*y[IDX_H2SI] + k[1894]*y[IDX_HCSI] -
        k[1899]*y[IDX_HSI] - k[1900]*y[IDX_HSI];
    data[3642] = 0.0 + k[1775]*y[IDX_HI];
    data[3643] = 0.0 + k[1935]*y[IDX_CSI] + k[1938]*y[IDX_H2SI];
    data[3644] = 0.0 + k[347]*y[IDX_HSII] + k[1673]*y[IDX_CH4I] +
        k[1698]*y[IDX_CHI] + k[1735]*y[IDX_H2I] + k[1856]*y[IDX_NHI] +
        k[1952]*y[IDX_HCOI] - k[1953]*y[IDX_HSI];
    data[3645] = 0.0 + k[975]*y[IDX_H2COI];
    data[3646] = 0.0 + k[1777]*y[IDX_HI];
    data[3647] = 0.0 + k[351]*y[IDX_HSII];
    data[3648] = 0.0 + k[1109]*y[IDX_HI];
    data[3649] = 0.0 + k[1699]*y[IDX_CHI] + k[1778]*y[IDX_HI];
    data[3650] = 0.0 - k[697]*y[IDX_HSII];
    data[3651] = 0.0 - k[851]*y[IDX_HSII];
    data[3652] = 0.0 + k[740]*y[IDX_SI];
    data[3653] = 0.0 - k[814]*y[IDX_HSII];
    data[3654] = 0.0 + k[835]*y[IDX_SI];
    data[3655] = 0.0 - k[554]*y[IDX_HSII];
    data[3656] = 0.0 + k[1103]*y[IDX_H2SII] - k[1105]*y[IDX_HSII];
    data[3657] = 0.0 + k[128]*y[IDX_HSI] + k[894]*y[IDX_H2SI] +
        k[904]*y[IDX_OCSI];
    data[3658] = 0.0 - k[949]*y[IDX_HSII] + k[961]*y[IDX_SII] -
        k[2116]*y[IDX_HSII];
    data[3659] = 0.0 + k[923]*y[IDX_H2SI];
    data[3660] = 0.0 + k[968]*y[IDX_SI];
    data[3661] = 0.0 - k[1007]*y[IDX_HSII];
    data[3662] = 0.0 + k[987]*y[IDX_SI];
    data[3663] = 0.0 + k[894]*y[IDX_HII] + k[923]*y[IDX_H2II] -
        k[1175]*y[IDX_HSII] - k[1176]*y[IDX_HSII] + k[1226]*y[IDX_HeII] +
        k[1300]*y[IDX_NII] + k[1315]*y[IDX_N2II] + k[1382]*y[IDX_NH2II] +
        k[1472]*y[IDX_OII];
    data[3664] = 0.0 + k[1103]*y[IDX_HI] + k[1498]*y[IDX_OI];
    data[3665] = 0.0 + k[1224]*y[IDX_HeII];
    data[3666] = 0.0 + k[1068]*y[IDX_SI];
    data[3667] = 0.0 - k[1126]*y[IDX_HSII];
    data[3668] = 0.0 + k[1117]*y[IDX_SI];
    data[3669] = 0.0 + k[1163]*y[IDX_SII];
    data[3670] = 0.0 + k[1151]*y[IDX_SI];
    data[3671] = 0.0 + k[1224]*y[IDX_H2S2I] + k[1226]*y[IDX_H2SI];
    data[3672] = 0.0 - k[1170]*y[IDX_HSII];
    data[3673] = 0.0 + k[1174]*y[IDX_SI];
    data[3674] = 0.0 + k[128]*y[IDX_HII];
    data[3675] = 0.0 - k[225]*y[IDX_MgI] - k[288]*y[IDX_NH3I] -
        k[301]*y[IDX_NOI] - k[347]*y[IDX_SI] - k[351]*y[IDX_SiI] -
        k[554]*y[IDX_EM] - k[697]*y[IDX_CI] - k[814]*y[IDX_CH4I] -
        k[851]*y[IDX_CHI] - k[949]*y[IDX_H2I] - k[1007]*y[IDX_H2OI] -
        k[1105]*y[IDX_HI] - k[1126]*y[IDX_HCNI] - k[1170]*y[IDX_HNCI] -
        k[1175]*y[IDX_H2SI] - k[1176]*y[IDX_H2SI] - k[1336]*y[IDX_NI] -
        k[1437]*y[IDX_NH3I] - k[1503]*y[IDX_OI] - k[1504]*y[IDX_OI] - k[2039] -
        k[2040] - k[2116]*y[IDX_H2I] - k[2265];
    data[3676] = 0.0 - k[225]*y[IDX_HSII];
    data[3677] = 0.0 - k[1336]*y[IDX_HSII];
    data[3678] = 0.0 + k[1300]*y[IDX_H2SI];
    data[3679] = 0.0 + k[1315]*y[IDX_H2SI];
    data[3680] = 0.0 + k[1324]*y[IDX_SI];
    data[3681] = 0.0 + k[1372]*y[IDX_SI];
    data[3682] = 0.0 + k[1382]*y[IDX_H2SI] + k[1393]*y[IDX_SI];
    data[3683] = 0.0 - k[288]*y[IDX_HSII] - k[1437]*y[IDX_HSII];
    data[3684] = 0.0 - k[301]*y[IDX_HSII];
    data[3685] = 0.0 + k[1498]*y[IDX_H2SII] - k[1503]*y[IDX_HSII] -
        k[1504]*y[IDX_HSII];
    data[3686] = 0.0 + k[1472]*y[IDX_H2SI];
    data[3687] = 0.0 + k[1554]*y[IDX_SI];
    data[3688] = 0.0 + k[904]*y[IDX_HII];
    data[3689] = 0.0 + k[1533]*y[IDX_SI];
    data[3690] = 0.0 - k[347]*y[IDX_HSII] + k[740]*y[IDX_CHII] +
        k[835]*y[IDX_CH5II] + k[968]*y[IDX_H2COII] + k[987]*y[IDX_H2OII] +
        k[1068]*y[IDX_H3II] + k[1117]*y[IDX_HCNII] + k[1151]*y[IDX_HCOII] +
        k[1174]*y[IDX_HNOII] + k[1324]*y[IDX_N2HII] + k[1372]*y[IDX_NHII] +
        k[1393]*y[IDX_NH2II] + k[1533]*y[IDX_OHII] + k[1554]*y[IDX_O2HII];
    data[3691] = 0.0 + k[961]*y[IDX_H2I] + k[1163]*y[IDX_HCOI];
    data[3692] = 0.0 - k[351]*y[IDX_HSII];
    data[3693] = 0.0 + k[517]*y[IDX_H2S2II];
    data[3694] = 0.0 - k[127]*y[IDX_HS2I];
    data[3695] = 0.0 + k[517]*y[IDX_EM];
    data[3696] = 0.0 - k[1052]*y[IDX_HS2I];
    data[3697] = 0.0 - k[1091]*y[IDX_HS2I];
    data[3698] = 0.0 - k[1146]*y[IDX_HS2I];
    data[3699] = 0.0 - k[1247]*y[IDX_HS2I] - k[1248]*y[IDX_HS2I];
    data[3700] = 0.0 - k[127]*y[IDX_HII] - k[417] - k[1052]*y[IDX_H3II] -
        k[1091]*y[IDX_H3OII] - k[1146]*y[IDX_HCOII] - k[1247]*y[IDX_HeII] -
        k[1248]*y[IDX_HeII] - k[2041] - k[2042] - k[2204];
    data[3701] = 0.0 - k[555]*y[IDX_HS2II] - k[556]*y[IDX_HS2II];
    data[3702] = 0.0 + k[127]*y[IDX_HS2I];
    data[3703] = 0.0 - k[1019]*y[IDX_HS2II] + k[1176]*y[IDX_HSII] +
        k[1550]*y[IDX_SII];
    data[3704] = 0.0 + k[1225]*y[IDX_HeII];
    data[3705] = 0.0 + k[1067]*y[IDX_S2I];
    data[3706] = 0.0 + k[1092]*y[IDX_S2I];
    data[3707] = 0.0 + k[1150]*y[IDX_S2I];
    data[3708] = 0.0 + k[1225]*y[IDX_H2S2I];
    data[3709] = 0.0 + k[1176]*y[IDX_H2SI];
    data[3710] = 0.0 + k[127]*y[IDX_HII] + k[2041];
    data[3711] = 0.0 - k[555]*y[IDX_EM] - k[556]*y[IDX_EM] -
        k[1019]*y[IDX_H2SI] - k[2205];
    data[3712] = 0.0 + k[1550]*y[IDX_H2SI];
    data[3713] = 0.0 + k[1067]*y[IDX_H3II] + k[1092]*y[IDX_H3OII] +
        k[1150]*y[IDX_HCOII];
    data[3714] = 0.0 - k[561]*y[IDX_HSiSII] - k[562]*y[IDX_HSiSII];
    data[3715] = 0.0 - k[1009]*y[IDX_HSiSII];
    data[3716] = 0.0 - k[1020]*y[IDX_HSiSII];
    data[3717] = 0.0 + k[1077]*y[IDX_SiSI];
    data[3718] = 0.0 - k[1127]*y[IDX_HSiSII];
    data[3719] = 0.0 + k[1157]*y[IDX_SiSI];
    data[3720] = 0.0 - k[561]*y[IDX_EM] - k[562]*y[IDX_EM] -
        k[1009]*y[IDX_H2OI] - k[1020]*y[IDX_H2SI] - k[1127]*y[IDX_HCNI] -
        k[1439]*y[IDX_NH3I] - k[2191];
    data[3721] = 0.0 - k[1439]*y[IDX_HSiSII];
    data[3722] = 0.0 + k[1568]*y[IDX_SiH2II];
    data[3723] = 0.0 + k[1568]*y[IDX_SI];
    data[3724] = 0.0 + k[1077]*y[IDX_H3II] + k[1157]*y[IDX_HCOII];
    data[3725] = 0.0 - k[557]*y[IDX_HSOII];
    data[3726] = 0.0 + k[988]*y[IDX_SI];
    data[3727] = 0.0 + k[1070]*y[IDX_SOI];
    data[3728] = 0.0 + k[1152]*y[IDX_SOI];
    data[3729] = 0.0 - k[557]*y[IDX_EM] - k[2283];
    data[3730] = 0.0 + k[988]*y[IDX_H2OII];
    data[3731] = 0.0 + k[1070]*y[IDX_H3II] + k[1152]*y[IDX_HCOII];
    data[3732] = 0.0 - k[558]*y[IDX_HSO2II] - k[559]*y[IDX_HSO2II] -
        k[560]*y[IDX_HSO2II];
    data[3733] = 0.0 + k[962]*y[IDX_SO2II];
    data[3734] = 0.0 - k[1008]*y[IDX_HSO2II];
    data[3735] = 0.0 + k[989]*y[IDX_SO2I];
    data[3736] = 0.0 + k[1069]*y[IDX_SO2I];
    data[3737] = 0.0 - k[558]*y[IDX_EM] - k[559]*y[IDX_EM] -
        k[560]*y[IDX_EM] - k[1008]*y[IDX_H2OI] - k[1438]*y[IDX_NH3I] - k[2295];
    data[3738] = 0.0 - k[1438]*y[IDX_HSO2II];
    data[3739] = 0.0 + k[989]*y[IDX_H2OII] + k[1069]*y[IDX_H3II];
    data[3740] = 0.0 + k[962]*y[IDX_H2I];
    data[3741] = 0.0 + k[2319] + k[2320] + k[2321] + k[2322];
    data[3742] = 0.0 - k[19]*y[IDX_MgI];
    data[3743] = 0.0 - k[220]*y[IDX_MgI];
    data[3744] = 0.0 - k[56]*y[IDX_MgI];
    data[3745] = 0.0 - k[73]*y[IDX_MgI];
    data[3746] = 0.0 - k[834]*y[IDX_MgI];
    data[3747] = 0.0 - k[221]*y[IDX_MgI];
    data[3748] = 0.0 + k[2140]*y[IDX_MgII];
    data[3749] = 0.0 - k[129]*y[IDX_MgI];
    data[3750] = 0.0 - k[222]*y[IDX_MgI];
    data[3751] = 0.0 - k[181]*y[IDX_MgI];
    data[3752] = 0.0 - k[223]*y[IDX_MgI];
    data[3753] = 0.0 - k[1054]*y[IDX_MgI];
    data[3754] = 0.0 - k[224]*y[IDX_MgI];
    data[3755] = 0.0 - k[225]*y[IDX_MgI];
    data[3756] = 0.0 - k[19]*y[IDX_CII] - k[56]*y[IDX_CHII] -
        k[73]*y[IDX_CH3II] - k[129]*y[IDX_HII] - k[181]*y[IDX_H2OII] -
        k[220]*y[IDX_C2H2II] - k[221]*y[IDX_CSII] - k[222]*y[IDX_H2COII] -
        k[223]*y[IDX_H2SII] - k[224]*y[IDX_HCOII] - k[225]*y[IDX_HSII] -
        k[226]*y[IDX_N2II] - k[227]*y[IDX_NOII] - k[228]*y[IDX_O2II] -
        k[229]*y[IDX_SII] - k[230]*y[IDX_SOII] - k[231]*y[IDX_SiII] -
        k[232]*y[IDX_SiOII] - k[244]*y[IDX_NII] - k[279]*y[IDX_NH3II] - k[420] -
        k[834]*y[IDX_CH5II] - k[1054]*y[IDX_H3II] - k[2044] - k[2287];
    data[3757] = 0.0 + k[2140]*y[IDX_EM];
    data[3758] = 0.0 - k[244]*y[IDX_MgI];
    data[3759] = 0.0 - k[226]*y[IDX_MgI];
    data[3760] = 0.0 - k[279]*y[IDX_MgI];
    data[3761] = 0.0 - k[227]*y[IDX_MgI];
    data[3762] = 0.0 - k[228]*y[IDX_MgI];
    data[3763] = 0.0 - k[229]*y[IDX_MgI];
    data[3764] = 0.0 - k[231]*y[IDX_MgI];
    data[3765] = 0.0 - k[232]*y[IDX_MgI];
    data[3766] = 0.0 - k[230]*y[IDX_MgI];
    data[3767] = 0.0 + k[19]*y[IDX_MgI];
    data[3768] = 0.0 + k[220]*y[IDX_MgI];
    data[3769] = 0.0 + k[56]*y[IDX_MgI];
    data[3770] = 0.0 + k[73]*y[IDX_MgI];
    data[3771] = 0.0 + k[834]*y[IDX_MgI];
    data[3772] = 0.0 + k[221]*y[IDX_MgI];
    data[3773] = 0.0 - k[2140]*y[IDX_MgII];
    data[3774] = 0.0 + k[129]*y[IDX_MgI];
    data[3775] = 0.0 + k[222]*y[IDX_MgI];
    data[3776] = 0.0 + k[181]*y[IDX_MgI];
    data[3777] = 0.0 + k[223]*y[IDX_MgI];
    data[3778] = 0.0 + k[1054]*y[IDX_MgI];
    data[3779] = 0.0 + k[224]*y[IDX_MgI];
    data[3780] = 0.0 + k[225]*y[IDX_MgI];
    data[3781] = 0.0 + k[19]*y[IDX_CII] + k[56]*y[IDX_CHII] +
        k[73]*y[IDX_CH3II] + k[129]*y[IDX_HII] + k[181]*y[IDX_H2OII] +
        k[220]*y[IDX_C2H2II] + k[221]*y[IDX_CSII] + k[222]*y[IDX_H2COII] +
        k[223]*y[IDX_H2SII] + k[224]*y[IDX_HCOII] + k[225]*y[IDX_HSII] +
        k[226]*y[IDX_N2II] + k[227]*y[IDX_NOII] + k[228]*y[IDX_O2II] +
        k[229]*y[IDX_SII] + k[230]*y[IDX_SOII] + k[231]*y[IDX_SiII] +
        k[232]*y[IDX_SiOII] + k[244]*y[IDX_NII] + k[279]*y[IDX_NH3II] + k[420] +
        k[834]*y[IDX_CH5II] + k[1054]*y[IDX_H3II] + k[2044];
    data[3782] = 0.0 - k[2140]*y[IDX_EM] - k[2286];
    data[3783] = 0.0 + k[244]*y[IDX_MgI];
    data[3784] = 0.0 + k[226]*y[IDX_MgI];
    data[3785] = 0.0 + k[279]*y[IDX_MgI];
    data[3786] = 0.0 + k[227]*y[IDX_MgI];
    data[3787] = 0.0 + k[228]*y[IDX_MgI];
    data[3788] = 0.0 + k[229]*y[IDX_MgI];
    data[3789] = 0.0 + k[231]*y[IDX_MgI];
    data[3790] = 0.0 + k[232]*y[IDX_MgI];
    data[3791] = 0.0 + k[230]*y[IDX_MgI];
    data[3792] = 0.0 + k[699]*y[IDX_NHII] + k[1590]*y[IDX_CNI] +
        k[1597]*y[IDX_N2I] + k[1603]*y[IDX_NHI] + k[1605]*y[IDX_NOI] +
        k[1606]*y[IDX_NSI] - k[2102]*y[IDX_NI];
    data[3793] = 0.0 + k[632]*y[IDX_NSI] - k[2097]*y[IDX_NI];
    data[3794] = 0.0 + k[233]*y[IDX_NII] + k[1345]*y[IDX_NHII] -
        k[1790]*y[IDX_NI];
    data[3795] = 0.0 - k[1325]*y[IDX_NI] + k[1444]*y[IDX_NHI];
    data[3796] = 0.0 + k[234]*y[IDX_NII] + k[1348]*y[IDX_NHII] -
        k[1795]*y[IDX_NI];
    data[3797] = 0.0 - k[1326]*y[IDX_NI] - k[1327]*y[IDX_NI];
    data[3798] = 0.0 - k[1328]*y[IDX_NI] - k[1329]*y[IDX_NI] -
        k[1330]*y[IDX_NI];
    data[3799] = 0.0 - k[1791]*y[IDX_NI];
    data[3800] = 0.0 - k[1792]*y[IDX_NI];
    data[3801] = 0.0 - k[1793]*y[IDX_NI] - k[1794]*y[IDX_NI];
    data[3802] = 0.0 + k[375] - k[1796]*y[IDX_NI] + k[1972];
    data[3803] = 0.0 + k[468]*y[IDX_EM];
    data[3804] = 0.0 + k[470]*y[IDX_EM];
    data[3805] = 0.0 - k[1797]*y[IDX_NI];
    data[3806] = 0.0 - k[1798]*y[IDX_NI];
    data[3807] = 0.0 - k[1799]*y[IDX_NI];
    data[3808] = 0.0 - k[1800]*y[IDX_NI];
    data[3809] = 0.0 + k[87]*y[IDX_NII] + k[854]*y[IDX_NHII] +
        k[1681]*y[IDX_N2I] - k[1682]*y[IDX_NI] - k[1683]*y[IDX_NI] +
        k[1685]*y[IDX_NOI];
    data[3810] = 0.0 - k[728]*y[IDX_NI];
    data[3811] = 0.0 + k[235]*y[IDX_NII] + k[766]*y[IDX_NHII] +
        k[1629]*y[IDX_NOI] - k[1801]*y[IDX_NI] - k[1802]*y[IDX_NI] -
        k[1803]*y[IDX_NI];
    data[3812] = 0.0 - k[1331]*y[IDX_NI];
    data[3813] = 0.0 - k[1804]*y[IDX_NI] - k[1805]*y[IDX_NI] -
        k[1806]*y[IDX_NI];
    data[3814] = 0.0 + k[236]*y[IDX_NII] + k[1293]*y[IDX_NII];
    data[3815] = 0.0 + k[237]*y[IDX_NII] + k[392] + k[1208]*y[IDX_HeII] +
        k[1349]*y[IDX_NHII] + k[1590]*y[IDX_CI] + k[1711]*y[IDX_NOI] -
        k[1807]*y[IDX_NI] + k[1840]*y[IDX_NHI] + k[1880]*y[IDX_OI] + k[2001];
    data[3816] = 0.0 + k[498]*y[IDX_EM] - k[1332]*y[IDX_NI];
    data[3817] = 0.0 + k[238]*y[IDX_NII] + k[1353]*y[IDX_NHII];
    data[3818] = 0.0 + k[1448]*y[IDX_NHI];
    data[3819] = 0.0 + k[1350]*y[IDX_NHII] - k[1808]*y[IDX_NI];
    data[3820] = 0.0 - k[1809]*y[IDX_NI];
    data[3821] = 0.0 + k[468]*y[IDX_C2NII] + k[470]*y[IDX_C2N2II] +
        k[498]*y[IDX_CNII] + k[564]*y[IDX_N2II] + k[564]*y[IDX_N2II] +
        k[566]*y[IDX_N2HII] + k[567]*y[IDX_NHII] + k[568]*y[IDX_NH2II] +
        k[575]*y[IDX_NOII] + k[576]*y[IDX_NSII] + k[2141]*y[IDX_NII];
    data[3822] = 0.0 + k[1762]*y[IDX_NHI] + k[1765]*y[IDX_NOI] +
        k[1766]*y[IDX_NSI];
    data[3823] = 0.0 + k[954]*y[IDX_NHII] - k[1728]*y[IDX_NI];
    data[3824] = 0.0 - k[928]*y[IDX_NI];
    data[3825] = 0.0 - k[1810]*y[IDX_NI];
    data[3826] = 0.0 + k[239]*y[IDX_NII] + k[1354]*y[IDX_NHII];
    data[3827] = 0.0 + k[1449]*y[IDX_NHI];
    data[3828] = 0.0 + k[240]*y[IDX_NII] + k[1356]*y[IDX_NHII];
    data[3829] = 0.0 - k[1333]*y[IDX_NI] - k[1334]*y[IDX_NI] +
        k[1450]*y[IDX_NHI];
    data[3830] = 0.0 + k[241]*y[IDX_NII];
    data[3831] = 0.0 - k[1335]*y[IDX_NI];
    data[3832] = 0.0 + k[242]*y[IDX_NII] + k[1233]*y[IDX_HeII] +
        k[1234]*y[IDX_HeII] + k[1360]*y[IDX_NHII] + k[1474]*y[IDX_OII];
    data[3833] = 0.0 + k[243]*y[IDX_NII] + k[1361]*y[IDX_NHII] -
        k[1811]*y[IDX_NI] - k[1812]*y[IDX_NI] - k[1813]*y[IDX_NI];
    data[3834] = 0.0 - k[1814]*y[IDX_NI];
    data[3835] = 0.0 + k[1208]*y[IDX_CNI] + k[1233]*y[IDX_HCNI] +
        k[1234]*y[IDX_HCNI] + k[1243]*y[IDX_HNCI] + k[1250]*y[IDX_N2I] +
        k[1257]*y[IDX_NOI] + k[1259]*y[IDX_NSI];
    data[3836] = 0.0 + k[1243]*y[IDX_HeII] + k[1362]*y[IDX_NHII];
    data[3837] = 0.0 - k[1815]*y[IDX_NI];
    data[3838] = 0.0 - k[1816]*y[IDX_NI] - k[1817]*y[IDX_NI];
    data[3839] = 0.0 - k[1336]*y[IDX_NI];
    data[3840] = 0.0 + k[244]*y[IDX_NII];
    data[3841] = 0.0 - k[259]*y[IDX_N2II] - k[364] - k[422] -
        k[728]*y[IDX_CHII] - k[928]*y[IDX_H2II] - k[1325]*y[IDX_C2II] -
        k[1326]*y[IDX_C2HII] - k[1327]*y[IDX_C2HII] - k[1328]*y[IDX_C2H2II] -
        k[1329]*y[IDX_C2H2II] - k[1330]*y[IDX_C2H2II] - k[1331]*y[IDX_CH2II] -
        k[1332]*y[IDX_CNII] - k[1333]*y[IDX_H2OII] - k[1334]*y[IDX_H2OII] -
        k[1335]*y[IDX_H2SII] - k[1336]*y[IDX_HSII] - k[1337]*y[IDX_NHII] -
        k[1338]*y[IDX_NH2II] - k[1339]*y[IDX_O2II] - k[1340]*y[IDX_OHII] -
        k[1341]*y[IDX_SOII] - k[1342]*y[IDX_SiCII] - k[1343]*y[IDX_SiOII] -
        k[1344]*y[IDX_SiOII] - k[1682]*y[IDX_CHI] - k[1683]*y[IDX_CHI] -
        k[1728]*y[IDX_H2I] - k[1790]*y[IDX_C2I] - k[1791]*y[IDX_C2H3I] -
        k[1792]*y[IDX_C2H4I] - k[1793]*y[IDX_C2H5I] - k[1794]*y[IDX_C2H5I] -
        k[1795]*y[IDX_C2HI] - k[1796]*y[IDX_C2NI] - k[1797]*y[IDX_C3H2I] -
        k[1798]*y[IDX_C3NI] - k[1799]*y[IDX_C4HI] - k[1800]*y[IDX_C4NI] -
        k[1801]*y[IDX_CH2I] - k[1802]*y[IDX_CH2I] - k[1803]*y[IDX_CH2I] -
        k[1804]*y[IDX_CH3I] - k[1805]*y[IDX_CH3I] - k[1806]*y[IDX_CH3I] -
        k[1807]*y[IDX_CNI] - k[1808]*y[IDX_CO2I] - k[1809]*y[IDX_CSI] -
        k[1810]*y[IDX_H2CNI] - k[1811]*y[IDX_HCOI] - k[1812]*y[IDX_HCOI] -
        k[1813]*y[IDX_HCOI] - k[1814]*y[IDX_HCSI] - k[1815]*y[IDX_HNOI] -
        k[1816]*y[IDX_HSI] - k[1817]*y[IDX_HSI] - k[1818]*y[IDX_NCCNI] -
        k[1819]*y[IDX_NHI] - k[1820]*y[IDX_NO2I] - k[1821]*y[IDX_NO2I] -
        k[1822]*y[IDX_NO2I] - k[1823]*y[IDX_NOI] - k[1824]*y[IDX_NSI] -
        k[1825]*y[IDX_O2I] - k[1826]*y[IDX_O2HI] - k[1827]*y[IDX_OHI] -
        k[1828]*y[IDX_OHI] - k[1829]*y[IDX_S2I] - k[1830]*y[IDX_SOI] -
        k[1831]*y[IDX_SOI] - k[1832]*y[IDX_SiCI] - k[2097]*y[IDX_CII] -
        k[2102]*y[IDX_CI] - k[2127]*y[IDX_NII] - k[2251];
    data[3842] = 0.0 + k[87]*y[IDX_CHI] + k[233]*y[IDX_C2I] +
        k[234]*y[IDX_C2HI] + k[235]*y[IDX_CH2I] + k[236]*y[IDX_CH4I] +
        k[237]*y[IDX_CNI] + k[238]*y[IDX_COI] + k[239]*y[IDX_H2COI] +
        k[240]*y[IDX_H2OI] + k[241]*y[IDX_H2SI] + k[242]*y[IDX_HCNI] +
        k[243]*y[IDX_HCOI] + k[244]*y[IDX_MgI] + k[245]*y[IDX_NH2I] +
        k[246]*y[IDX_NH3I] + k[247]*y[IDX_NHI] + k[248]*y[IDX_NOI] +
        k[249]*y[IDX_O2I] + k[250]*y[IDX_OCSI] + k[251]*y[IDX_OHI] +
        k[1293]*y[IDX_CH4I] + k[1305]*y[IDX_NCCNI] + k[1313]*y[IDX_OCSI] -
        k[2127]*y[IDX_NI] + k[2141]*y[IDX_EM];
    data[3843] = 0.0 + k[421] + k[421] + k[1250]*y[IDX_HeII] +
        k[1363]*y[IDX_NHII] + k[1477]*y[IDX_OII] + k[1597]*y[IDX_CI] +
        k[1681]*y[IDX_CHI] + k[1901]*y[IDX_OI] + k[2045] + k[2045];
    data[3844] = 0.0 - k[259]*y[IDX_NI] + k[564]*y[IDX_EM] +
        k[564]*y[IDX_EM] + k[1505]*y[IDX_OI];
    data[3845] = 0.0 + k[566]*y[IDX_EM];
    data[3846] = 0.0 + k[1305]*y[IDX_NII] - k[1818]*y[IDX_NI];
    data[3847] = 0.0 + k[247]*y[IDX_NII] + k[429] + k[1366]*y[IDX_NHII] +
        k[1444]*y[IDX_C2II] + k[1448]*y[IDX_COII] + k[1449]*y[IDX_H2COII] +
        k[1450]*y[IDX_H2OII] + k[1455]*y[IDX_NH2II] + k[1456]*y[IDX_NH3II] +
        k[1603]*y[IDX_CI] + k[1762]*y[IDX_HI] - k[1819]*y[IDX_NI] +
        k[1840]*y[IDX_CNI] + k[1845]*y[IDX_NHI] + k[1845]*y[IDX_NHI] +
        k[1852]*y[IDX_OI] + k[1853]*y[IDX_OHI] + k[1856]*y[IDX_SI] + k[2054];
    data[3848] = 0.0 + k[567]*y[IDX_EM] + k[699]*y[IDX_CI] +
        k[766]*y[IDX_CH2I] + k[854]*y[IDX_CHI] + k[954]*y[IDX_H2I] -
        k[1337]*y[IDX_NI] + k[1345]*y[IDX_C2I] + k[1348]*y[IDX_C2HI] +
        k[1349]*y[IDX_CNI] + k[1350]*y[IDX_CO2I] + k[1353]*y[IDX_COI] +
        k[1354]*y[IDX_H2COI] + k[1356]*y[IDX_H2OI] + k[1360]*y[IDX_HCNI] +
        k[1361]*y[IDX_HCOI] + k[1362]*y[IDX_HNCI] + k[1363]*y[IDX_N2I] +
        k[1364]*y[IDX_NH2I] + k[1365]*y[IDX_NH3I] + k[1366]*y[IDX_NHI] +
        k[1369]*y[IDX_O2I] + k[1370]*y[IDX_OI] + k[1371]*y[IDX_OHI] +
        k[1372]*y[IDX_SI] + k[2048];
    data[3849] = 0.0 + k[245]*y[IDX_NII] + k[1364]*y[IDX_NHII];
    data[3850] = 0.0 + k[568]*y[IDX_EM] - k[1338]*y[IDX_NI] +
        k[1455]*y[IDX_NHI];
    data[3851] = 0.0 + k[246]*y[IDX_NII] + k[1365]*y[IDX_NHII];
    data[3852] = 0.0 + k[1456]*y[IDX_NHI];
    data[3853] = 0.0 + k[248]*y[IDX_NII] + k[433] + k[1257]*y[IDX_HeII] +
        k[1605]*y[IDX_CI] + k[1629]*y[IDX_CH2I] + k[1685]*y[IDX_CHI] +
        k[1711]*y[IDX_CNI] + k[1765]*y[IDX_HI] - k[1823]*y[IDX_NI] +
        k[1862]*y[IDX_SI] + k[1906]*y[IDX_OI] + k[1958]*y[IDX_SiI] + k[2058];
    data[3854] = 0.0 + k[575]*y[IDX_EM];
    data[3855] = 0.0 - k[1820]*y[IDX_NI] - k[1821]*y[IDX_NI] -
        k[1822]*y[IDX_NI];
    data[3856] = 0.0 + k[434] + k[632]*y[IDX_CII] + k[1259]*y[IDX_HeII] +
        k[1606]*y[IDX_CI] + k[1766]*y[IDX_HI] - k[1824]*y[IDX_NI] +
        k[1908]*y[IDX_OI] + k[2059];
    data[3857] = 0.0 + k[576]*y[IDX_EM];
    data[3858] = 0.0 + k[1370]*y[IDX_NHII] + k[1505]*y[IDX_N2II] +
        k[1852]*y[IDX_NHI] + k[1880]*y[IDX_CNI] + k[1901]*y[IDX_N2I] +
        k[1906]*y[IDX_NOI] + k[1908]*y[IDX_NSI];
    data[3859] = 0.0 + k[1474]*y[IDX_HCNI] + k[1477]*y[IDX_N2I];
    data[3860] = 0.0 + k[249]*y[IDX_NII] + k[1369]*y[IDX_NHII] -
        k[1825]*y[IDX_NI];
    data[3861] = 0.0 - k[1339]*y[IDX_NI];
    data[3862] = 0.0 - k[1826]*y[IDX_NI];
    data[3863] = 0.0 + k[250]*y[IDX_NII] + k[1313]*y[IDX_NII];
    data[3864] = 0.0 + k[251]*y[IDX_NII] + k[1371]*y[IDX_NHII] -
        k[1827]*y[IDX_NI] - k[1828]*y[IDX_NI] + k[1853]*y[IDX_NHI];
    data[3865] = 0.0 - k[1340]*y[IDX_NI];
    data[3866] = 0.0 + k[1372]*y[IDX_NHII] + k[1856]*y[IDX_NHI] +
        k[1862]*y[IDX_NOI];
    data[3867] = 0.0 - k[1829]*y[IDX_NI];
    data[3868] = 0.0 + k[1958]*y[IDX_NOI];
    data[3869] = 0.0 - k[1832]*y[IDX_NI];
    data[3870] = 0.0 - k[1342]*y[IDX_NI];
    data[3871] = 0.0 - k[1343]*y[IDX_NI] - k[1344]*y[IDX_NI];
    data[3872] = 0.0 - k[1830]*y[IDX_NI] - k[1831]*y[IDX_NI];
    data[3873] = 0.0 - k[1341]*y[IDX_NI];
    data[3874] = 0.0 - k[233]*y[IDX_NII];
    data[3875] = 0.0 - k[234]*y[IDX_NII];
    data[3876] = 0.0 - k[87]*y[IDX_NII] - k[852]*y[IDX_NII];
    data[3877] = 0.0 - k[235]*y[IDX_NII];
    data[3878] = 0.0 - k[1289]*y[IDX_NII] - k[1290]*y[IDX_NII] -
        k[1291]*y[IDX_NII] - k[1292]*y[IDX_NII];
    data[3879] = 0.0 - k[236]*y[IDX_NII] - k[1293]*y[IDX_NII] -
        k[1294]*y[IDX_NII] - k[1295]*y[IDX_NII];
    data[3880] = 0.0 - k[237]*y[IDX_NII] + k[1207]*y[IDX_HeII];
    data[3881] = 0.0 - k[238]*y[IDX_NII] - k[1297]*y[IDX_NII];
    data[3882] = 0.0 - k[1296]*y[IDX_NII];
    data[3883] = 0.0 - k[2141]*y[IDX_NII];
    data[3884] = 0.0 - k[952]*y[IDX_NII];
    data[3885] = 0.0 - k[239]*y[IDX_NII] - k[1298]*y[IDX_NII] -
        k[1299]*y[IDX_NII];
    data[3886] = 0.0 - k[240]*y[IDX_NII];
    data[3887] = 0.0 - k[241]*y[IDX_NII] - k[1300]*y[IDX_NII] -
        k[1301]*y[IDX_NII] - k[1302]*y[IDX_NII];
    data[3888] = 0.0 - k[242]*y[IDX_NII] + k[1232]*y[IDX_HeII];
    data[3889] = 0.0 - k[243]*y[IDX_NII] - k[1303]*y[IDX_NII];
    data[3890] = 0.0 + k[1207]*y[IDX_CNI] + k[1232]*y[IDX_HCNI] +
        k[1250]*y[IDX_N2I] + k[1252]*y[IDX_NH2I] + k[1256]*y[IDX_NHI] +
        k[1258]*y[IDX_NOI] + k[1260]*y[IDX_NSI];
    data[3891] = 0.0 - k[244]*y[IDX_NII];
    data[3892] = 0.0 + k[259]*y[IDX_N2II] + k[364] + k[422] -
        k[2127]*y[IDX_NII];
    data[3893] = 0.0 - k[87]*y[IDX_CHI] - k[233]*y[IDX_C2I] -
        k[234]*y[IDX_C2HI] - k[235]*y[IDX_CH2I] - k[236]*y[IDX_CH4I] -
        k[237]*y[IDX_CNI] - k[238]*y[IDX_COI] - k[239]*y[IDX_H2COI] -
        k[240]*y[IDX_H2OI] - k[241]*y[IDX_H2SI] - k[242]*y[IDX_HCNI] -
        k[243]*y[IDX_HCOI] - k[244]*y[IDX_MgI] - k[245]*y[IDX_NH2I] -
        k[246]*y[IDX_NH3I] - k[247]*y[IDX_NHI] - k[248]*y[IDX_NOI] -
        k[249]*y[IDX_O2I] - k[250]*y[IDX_OCSI] - k[251]*y[IDX_OHI] -
        k[852]*y[IDX_CHI] - k[952]*y[IDX_H2I] - k[1289]*y[IDX_CH3OHI] -
        k[1290]*y[IDX_CH3OHI] - k[1291]*y[IDX_CH3OHI] - k[1292]*y[IDX_CH3OHI] -
        k[1293]*y[IDX_CH4I] - k[1294]*y[IDX_CH4I] - k[1295]*y[IDX_CH4I] -
        k[1296]*y[IDX_CO2I] - k[1297]*y[IDX_COI] - k[1298]*y[IDX_H2COI] -
        k[1299]*y[IDX_H2COI] - k[1300]*y[IDX_H2SI] - k[1301]*y[IDX_H2SI] -
        k[1302]*y[IDX_H2SI] - k[1303]*y[IDX_HCOI] - k[1304]*y[IDX_NCCNI] -
        k[1305]*y[IDX_NCCNI] - k[1306]*y[IDX_NH3I] - k[1307]*y[IDX_NH3I] -
        k[1308]*y[IDX_NHI] - k[1309]*y[IDX_NOI] - k[1310]*y[IDX_O2I] -
        k[1311]*y[IDX_O2I] - k[1312]*y[IDX_OCSI] - k[1313]*y[IDX_OCSI] -
        k[2127]*y[IDX_NI] - k[2141]*y[IDX_EM] - k[2226];
    data[3894] = 0.0 + k[1250]*y[IDX_HeII];
    data[3895] = 0.0 + k[259]*y[IDX_NI];
    data[3896] = 0.0 - k[1304]*y[IDX_NII] - k[1305]*y[IDX_NII];
    data[3897] = 0.0 - k[247]*y[IDX_NII] + k[1256]*y[IDX_HeII] -
        k[1308]*y[IDX_NII];
    data[3898] = 0.0 - k[245]*y[IDX_NII] + k[1252]*y[IDX_HeII];
    data[3899] = 0.0 - k[246]*y[IDX_NII] - k[1306]*y[IDX_NII] -
        k[1307]*y[IDX_NII];
    data[3900] = 0.0 - k[248]*y[IDX_NII] + k[1258]*y[IDX_HeII] -
        k[1309]*y[IDX_NII];
    data[3901] = 0.0 + k[1260]*y[IDX_HeII];
    data[3902] = 0.0 - k[249]*y[IDX_NII] - k[1310]*y[IDX_NII] -
        k[1311]*y[IDX_NII];
    data[3903] = 0.0 - k[250]*y[IDX_NII] - k[1312]*y[IDX_NII] -
        k[1313]*y[IDX_NII];
    data[3904] = 0.0 - k[251]*y[IDX_NII];
    data[3905] = 0.0 + k[2347] + k[2348] + k[2349] + k[2350];
    data[3906] = 0.0 + k[53]*y[IDX_N2II] + k[698]*y[IDX_N2HII] -
        k[1597]*y[IDX_N2I];
    data[3907] = 0.0 + k[38]*y[IDX_N2II] + k[655]*y[IDX_N2HII];
    data[3908] = 0.0 + k[49]*y[IDX_N2II] + k[682]*y[IDX_N2HII];
    data[3909] = 0.0 + k[88]*y[IDX_N2II] + k[853]*y[IDX_N2HII] -
        k[1681]*y[IDX_N2I];
    data[3910] = 0.0 + k[67]*y[IDX_N2II] + k[765]*y[IDX_N2HII] -
        k[1627]*y[IDX_N2I];
    data[3911] = 0.0 + k[1321]*y[IDX_N2HII];
    data[3912] = 0.0 + k[815]*y[IDX_N2II] + k[816]*y[IDX_N2II] +
        k[817]*y[IDX_N2HII];
    data[3913] = 0.0 + k[100]*y[IDX_N2II] + k[1703]*y[IDX_CNI] +
        k[1703]*y[IDX_CNI] + k[1710]*y[IDX_NOI] + k[1807]*y[IDX_NI];
    data[3914] = 0.0 + k[107]*y[IDX_N2II] + k[877]*y[IDX_N2HII];
    data[3915] = 0.0 + k[1322]*y[IDX_N2HII];
    data[3916] = 0.0 + k[565]*y[IDX_N2HII];
    data[3917] = 0.0 - k[927]*y[IDX_N2I];
    data[3918] = 0.0 + k[252]*y[IDX_N2II] + k[1314]*y[IDX_N2II] +
        k[1323]*y[IDX_N2HII];
    data[3919] = 0.0 + k[189]*y[IDX_N2II] + k[1011]*y[IDX_N2HII];
    data[3920] = 0.0 + k[253]*y[IDX_N2II] + k[1315]*y[IDX_N2II] +
        k[1316]*y[IDX_N2II];
    data[3921] = 0.0 - k[1055]*y[IDX_N2I];
    data[3922] = 0.0 + k[201]*y[IDX_N2II] + k[1128]*y[IDX_N2HII];
    data[3923] = 0.0 + k[254]*y[IDX_N2II] + k[1160]*y[IDX_N2HII];
    data[3924] = 0.0 - k[215]*y[IDX_N2I] - k[1250]*y[IDX_N2I];
    data[3925] = 0.0 + k[1171]*y[IDX_N2HII];
    data[3926] = 0.0 - k[1319]*y[IDX_N2I];
    data[3927] = 0.0 + k[226]*y[IDX_N2II];
    data[3928] = 0.0 + k[259]*y[IDX_N2II] + k[1807]*y[IDX_CNI] +
        k[1818]*y[IDX_NCCNI] + k[1819]*y[IDX_NHI] + k[1820]*y[IDX_NO2I] +
        k[1822]*y[IDX_NO2I] + k[1823]*y[IDX_NOI] + k[1824]*y[IDX_NSI];
    data[3929] = 0.0 + k[1304]*y[IDX_NCCNI];
    data[3930] = 0.0 - k[215]*y[IDX_HeII] - k[421] - k[927]*y[IDX_H2II] -
        k[1055]*y[IDX_H3II] - k[1250]*y[IDX_HeII] - k[1319]*y[IDX_HNOII] -
        k[1320]*y[IDX_O2HII] - k[1363]*y[IDX_NHII] - k[1477]*y[IDX_OII] -
        k[1529]*y[IDX_OHII] - k[1597]*y[IDX_CI] - k[1627]*y[IDX_CH2I] -
        k[1681]*y[IDX_CHI] - k[1901]*y[IDX_OI] - k[2045] - k[2219];
    data[3931] = 0.0 + k[38]*y[IDX_C2I] + k[49]*y[IDX_C2HI] +
        k[53]*y[IDX_CI] + k[67]*y[IDX_CH2I] + k[88]*y[IDX_CHI] +
        k[100]*y[IDX_CNI] + k[107]*y[IDX_COI] + k[189]*y[IDX_H2OI] +
        k[201]*y[IDX_HCNI] + k[226]*y[IDX_MgI] + k[252]*y[IDX_H2COI] +
        k[253]*y[IDX_H2SI] + k[254]*y[IDX_HCOI] + k[255]*y[IDX_NOI] +
        k[256]*y[IDX_O2I] + k[257]*y[IDX_OCSI] + k[258]*y[IDX_SI] +
        k[259]*y[IDX_NI] + k[275]*y[IDX_NH2I] + k[289]*y[IDX_NH3I] +
        k[296]*y[IDX_NHI] + k[328]*y[IDX_OI] + k[342]*y[IDX_OHI] +
        k[815]*y[IDX_CH4I] + k[816]*y[IDX_CH4I] + k[1314]*y[IDX_H2COI] +
        k[1315]*y[IDX_H2SI] + k[1316]*y[IDX_H2SI] + k[1318]*y[IDX_OCSI];
    data[3932] = 0.0 + k[565]*y[IDX_EM] + k[655]*y[IDX_C2I] +
        k[682]*y[IDX_C2HI] + k[698]*y[IDX_CI] + k[765]*y[IDX_CH2I] +
        k[817]*y[IDX_CH4I] + k[853]*y[IDX_CHI] + k[877]*y[IDX_COI] +
        k[1011]*y[IDX_H2OI] + k[1128]*y[IDX_HCNI] + k[1160]*y[IDX_HCOI] +
        k[1171]*y[IDX_HNCI] + k[1321]*y[IDX_CH3CNI] + k[1322]*y[IDX_CO2I] +
        k[1323]*y[IDX_H2COI] + k[1324]*y[IDX_SI] + k[1408]*y[IDX_NH2I] +
        k[1440]*y[IDX_NH3I] + k[1454]*y[IDX_NHI] + k[1506]*y[IDX_OI] +
        k[1545]*y[IDX_OHI];
    data[3933] = 0.0 + k[1304]*y[IDX_NII] + k[1818]*y[IDX_NI];
    data[3934] = 0.0 + k[296]*y[IDX_N2II] + k[1454]*y[IDX_N2HII] +
        k[1819]*y[IDX_NI] + k[1843]*y[IDX_NHI] + k[1843]*y[IDX_NHI] +
        k[1844]*y[IDX_NHI] + k[1844]*y[IDX_NHI] + k[1847]*y[IDX_NOI] +
        k[1848]*y[IDX_NOI];
    data[3935] = 0.0 - k[1363]*y[IDX_N2I];
    data[3936] = 0.0 + k[275]*y[IDX_N2II] + k[1408]*y[IDX_N2HII] +
        k[1834]*y[IDX_NOI] + k[1835]*y[IDX_NOI];
    data[3937] = 0.0 + k[289]*y[IDX_N2II] + k[1440]*y[IDX_N2HII];
    data[3938] = 0.0 + k[255]*y[IDX_N2II] + k[1710]*y[IDX_CNI] +
        k[1823]*y[IDX_NI] + k[1834]*y[IDX_NH2I] + k[1835]*y[IDX_NH2I] +
        k[1847]*y[IDX_NHI] + k[1848]*y[IDX_NHI] + k[1858]*y[IDX_NOI] +
        k[1858]*y[IDX_NOI] + k[1860]*y[IDX_OCNI];
    data[3939] = 0.0 + k[1820]*y[IDX_NI] + k[1822]*y[IDX_NI];
    data[3940] = 0.0 + k[1824]*y[IDX_NI];
    data[3941] = 0.0 + k[328]*y[IDX_N2II] + k[1506]*y[IDX_N2HII] -
        k[1901]*y[IDX_N2I];
    data[3942] = 0.0 - k[1477]*y[IDX_N2I];
    data[3943] = 0.0 + k[256]*y[IDX_N2II];
    data[3944] = 0.0 - k[1320]*y[IDX_N2I];
    data[3945] = 0.0 + k[1860]*y[IDX_NOI];
    data[3946] = 0.0 + k[257]*y[IDX_N2II] + k[1318]*y[IDX_N2II];
    data[3947] = 0.0 + k[342]*y[IDX_N2II] + k[1545]*y[IDX_N2HII];
    data[3948] = 0.0 - k[1529]*y[IDX_N2I];
    data[3949] = 0.0 + k[258]*y[IDX_N2II] + k[1324]*y[IDX_N2HII];
    data[3950] = 0.0 - k[53]*y[IDX_N2II];
    data[3951] = 0.0 - k[38]*y[IDX_N2II];
    data[3952] = 0.0 - k[49]*y[IDX_N2II];
    data[3953] = 0.0 - k[88]*y[IDX_N2II];
    data[3954] = 0.0 - k[67]*y[IDX_N2II];
    data[3955] = 0.0 - k[815]*y[IDX_N2II] - k[816]*y[IDX_N2II];
    data[3956] = 0.0 - k[100]*y[IDX_N2II];
    data[3957] = 0.0 + k[1332]*y[IDX_NI];
    data[3958] = 0.0 - k[107]*y[IDX_N2II];
    data[3959] = 0.0 - k[564]*y[IDX_N2II];
    data[3960] = 0.0 - k[953]*y[IDX_N2II];
    data[3961] = 0.0 - k[252]*y[IDX_N2II] - k[1314]*y[IDX_N2II];
    data[3962] = 0.0 - k[189]*y[IDX_N2II] - k[1010]*y[IDX_N2II];
    data[3963] = 0.0 - k[253]*y[IDX_N2II] - k[1315]*y[IDX_N2II] -
        k[1316]*y[IDX_N2II];
    data[3964] = 0.0 - k[201]*y[IDX_N2II];
    data[3965] = 0.0 - k[254]*y[IDX_N2II] - k[1317]*y[IDX_N2II];
    data[3966] = 0.0 + k[215]*y[IDX_N2I];
    data[3967] = 0.0 - k[226]*y[IDX_N2II];
    data[3968] = 0.0 - k[259]*y[IDX_N2II] + k[1332]*y[IDX_CNII] +
        k[1337]*y[IDX_NHII] + k[2127]*y[IDX_NII];
    data[3969] = 0.0 + k[1308]*y[IDX_NHI] + k[1309]*y[IDX_NOI] +
        k[2127]*y[IDX_NI];
    data[3970] = 0.0 + k[215]*y[IDX_HeII];
    data[3971] = 0.0 - k[38]*y[IDX_C2I] - k[49]*y[IDX_C2HI] -
        k[53]*y[IDX_CI] - k[67]*y[IDX_CH2I] - k[88]*y[IDX_CHI] -
        k[100]*y[IDX_CNI] - k[107]*y[IDX_COI] - k[189]*y[IDX_H2OI] -
        k[201]*y[IDX_HCNI] - k[226]*y[IDX_MgI] - k[252]*y[IDX_H2COI] -
        k[253]*y[IDX_H2SI] - k[254]*y[IDX_HCOI] - k[255]*y[IDX_NOI] -
        k[256]*y[IDX_O2I] - k[257]*y[IDX_OCSI] - k[258]*y[IDX_SI] -
        k[259]*y[IDX_NI] - k[275]*y[IDX_NH2I] - k[289]*y[IDX_NH3I] -
        k[296]*y[IDX_NHI] - k[328]*y[IDX_OI] - k[342]*y[IDX_OHI] -
        k[564]*y[IDX_EM] - k[815]*y[IDX_CH4I] - k[816]*y[IDX_CH4I] -
        k[953]*y[IDX_H2I] - k[1010]*y[IDX_H2OI] - k[1314]*y[IDX_H2COI] -
        k[1315]*y[IDX_H2SI] - k[1316]*y[IDX_H2SI] - k[1317]*y[IDX_HCOI] -
        k[1318]*y[IDX_OCSI] - k[1505]*y[IDX_OI] - k[2230];
    data[3972] = 0.0 - k[296]*y[IDX_N2II] + k[1308]*y[IDX_NII];
    data[3973] = 0.0 + k[1337]*y[IDX_NI];
    data[3974] = 0.0 - k[275]*y[IDX_N2II];
    data[3975] = 0.0 - k[289]*y[IDX_N2II];
    data[3976] = 0.0 - k[255]*y[IDX_N2II] + k[1309]*y[IDX_NII];
    data[3977] = 0.0 - k[328]*y[IDX_N2II] - k[1505]*y[IDX_N2II];
    data[3978] = 0.0 - k[256]*y[IDX_N2II];
    data[3979] = 0.0 - k[257]*y[IDX_N2II] - k[1318]*y[IDX_N2II];
    data[3980] = 0.0 - k[342]*y[IDX_N2II];
    data[3981] = 0.0 - k[258]*y[IDX_N2II];
    data[3982] = 0.0 - k[698]*y[IDX_N2HII];
    data[3983] = 0.0 - k[655]*y[IDX_N2HII];
    data[3984] = 0.0 - k[682]*y[IDX_N2HII];
    data[3985] = 0.0 - k[853]*y[IDX_N2HII];
    data[3986] = 0.0 - k[765]*y[IDX_N2HII];
    data[3987] = 0.0 - k[1321]*y[IDX_N2HII];
    data[3988] = 0.0 - k[817]*y[IDX_N2HII];
    data[3989] = 0.0 - k[877]*y[IDX_N2HII];
    data[3990] = 0.0 - k[1322]*y[IDX_N2HII];
    data[3991] = 0.0 - k[565]*y[IDX_N2HII] - k[566]*y[IDX_N2HII];
    data[3992] = 0.0 + k[953]*y[IDX_N2II];
    data[3993] = 0.0 + k[927]*y[IDX_N2I];
    data[3994] = 0.0 - k[1323]*y[IDX_N2HII];
    data[3995] = 0.0 + k[1010]*y[IDX_N2II] - k[1011]*y[IDX_N2HII];
    data[3996] = 0.0 + k[1055]*y[IDX_N2I];
    data[3997] = 0.0 - k[1128]*y[IDX_N2HII];
    data[3998] = 0.0 - k[1160]*y[IDX_N2HII] + k[1317]*y[IDX_N2II];
    data[3999] = 0.0 - k[1171]*y[IDX_N2HII];
    data[4000] = 0.0 + k[1319]*y[IDX_N2I];
    data[4001] = 0.0 + k[1338]*y[IDX_NH2II];
    data[4002] = 0.0 + k[1306]*y[IDX_NH3I];
    data[4003] = 0.0 + k[927]*y[IDX_H2II] + k[1055]*y[IDX_H3II] +
        k[1319]*y[IDX_HNOII] + k[1320]*y[IDX_O2HII] + k[1363]*y[IDX_NHII] +
        k[1529]*y[IDX_OHII];
    data[4004] = 0.0 + k[953]*y[IDX_H2I] + k[1010]*y[IDX_H2OI] +
        k[1317]*y[IDX_HCOI];
    data[4005] = 0.0 - k[565]*y[IDX_EM] - k[566]*y[IDX_EM] -
        k[655]*y[IDX_C2I] - k[682]*y[IDX_C2HI] - k[698]*y[IDX_CI] -
        k[765]*y[IDX_CH2I] - k[817]*y[IDX_CH4I] - k[853]*y[IDX_CHI] -
        k[877]*y[IDX_COI] - k[1011]*y[IDX_H2OI] - k[1128]*y[IDX_HCNI] -
        k[1160]*y[IDX_HCOI] - k[1171]*y[IDX_HNCI] - k[1321]*y[IDX_CH3CNI] -
        k[1322]*y[IDX_CO2I] - k[1323]*y[IDX_H2COI] - k[1324]*y[IDX_SI] -
        k[1408]*y[IDX_NH2I] - k[1440]*y[IDX_NH3I] - k[1454]*y[IDX_NHI] -
        k[1506]*y[IDX_OI] - k[1545]*y[IDX_OHI] - k[2290];
    data[4006] = 0.0 - k[1454]*y[IDX_N2HII];
    data[4007] = 0.0 + k[1363]*y[IDX_N2I] + k[1367]*y[IDX_NOI];
    data[4008] = 0.0 - k[1408]*y[IDX_N2HII];
    data[4009] = 0.0 + k[1338]*y[IDX_NI];
    data[4010] = 0.0 + k[1306]*y[IDX_NII] - k[1440]*y[IDX_N2HII];
    data[4011] = 0.0 + k[1367]*y[IDX_NHII];
    data[4012] = 0.0 - k[1506]*y[IDX_N2HII];
    data[4013] = 0.0 + k[1320]*y[IDX_N2I];
    data[4014] = 0.0 - k[1545]*y[IDX_N2HII];
    data[4015] = 0.0 + k[1529]*y[IDX_N2I];
    data[4016] = 0.0 - k[1324]*y[IDX_N2HII];
    data[4017] = 0.0 + k[2471] + k[2472] + k[2473] + k[2474];
    data[4018] = 0.0 - k[1598]*y[IDX_NCCNI];
    data[4019] = 0.0 - k[20]*y[IDX_NCCNI];
    data[4020] = 0.0 - k[1580]*y[IDX_NCCNI];
    data[4021] = 0.0 + k[675]*y[IDX_C2N2II];
    data[4022] = 0.0 + k[675]*y[IDX_C2H2I] + k[1118]*y[IDX_HCNI];
    data[4023] = 0.0 + k[1705]*y[IDX_HCNI] + k[1707]*y[IDX_HNCI];
    data[4024] = 0.0 - k[1759]*y[IDX_NCCNI];
    data[4025] = 0.0 + k[1118]*y[IDX_C2N2II] + k[1705]*y[IDX_CNI];
    data[4026] = 0.0 - k[1251]*y[IDX_NCCNI];
    data[4027] = 0.0 + k[1707]*y[IDX_CNI];
    data[4028] = 0.0 - k[1818]*y[IDX_NCCNI];
    data[4029] = 0.0 - k[1304]*y[IDX_NCCNI] - k[1305]*y[IDX_NCCNI];
    data[4030] = 0.0 - k[20]*y[IDX_CII] - k[423] - k[1251]*y[IDX_HeII] -
        k[1304]*y[IDX_NII] - k[1305]*y[IDX_NII] - k[1580]*y[IDX_C2HI] -
        k[1598]*y[IDX_CI] - k[1759]*y[IDX_HI] - k[1818]*y[IDX_NI] -
        k[1943]*y[IDX_OHI] - k[2046] - k[2047] - k[2164];
    data[4031] = 0.0 - k[1943]*y[IDX_NCCNI];
    data[4032] = 0.0 + k[1601]*y[IDX_NH2I] - k[1602]*y[IDX_NHI] -
        k[1603]*y[IDX_NHI];
    data[4033] = 0.0 - k[631]*y[IDX_NHI];
    data[4034] = 0.0 + k[1374]*y[IDX_NH2II] + k[1412]*y[IDX_NH3II];
    data[4035] = 0.0 - k[1444]*y[IDX_NHI] - k[1445]*y[IDX_NHI];
    data[4036] = 0.0 + k[1375]*y[IDX_NH2II];
    data[4037] = 0.0 + k[1791]*y[IDX_NI];
    data[4038] = 0.0 + k[1793]*y[IDX_NI];
    data[4039] = 0.0 + k[855]*y[IDX_NH2II] + k[1683]*y[IDX_NI];
    data[4040] = 0.0 - k[731]*y[IDX_NHI];
    data[4041] = 0.0 + k[767]*y[IDX_NH2II] + k[1627]*y[IDX_N2I] +
        k[1803]*y[IDX_NI];
    data[4042] = 0.0 + k[1656]*y[IDX_NH2I];
    data[4043] = 0.0 - k[1446]*y[IDX_NHI];
    data[4044] = 0.0 + k[1289]*y[IDX_NII] + k[1290]*y[IDX_NII];
    data[4045] = 0.0 - k[1839]*y[IDX_NHI];
    data[4046] = 0.0 - k[1447]*y[IDX_NHI];
    data[4047] = 0.0 - k[1840]*y[IDX_NHI];
    data[4048] = 0.0 - k[294]*y[IDX_NHI] + k[996]*y[IDX_H2OI];
    data[4049] = 0.0 + k[1716]*y[IDX_HNOI];
    data[4050] = 0.0 - k[295]*y[IDX_NHI] + k[1398]*y[IDX_NH2I] -
        k[1448]*y[IDX_NHI];
    data[4051] = 0.0 + k[566]*y[IDX_N2HII] + k[569]*y[IDX_NH2II] +
        k[571]*y[IDX_NH3II];
    data[4052] = 0.0 + k[1757]*y[IDX_HNOI] + k[1760]*y[IDX_NH2I] -
        k[1762]*y[IDX_NHI] + k[1764]*y[IDX_NOI] + k[1767]*y[IDX_NSI] +
        k[1773]*y[IDX_OCNI];
    data[4053] = 0.0 - k[132]*y[IDX_NHI];
    data[4054] = 0.0 + k[1728]*y[IDX_NI] - k[1730]*y[IDX_NHI];
    data[4055] = 0.0 - k[168]*y[IDX_NHI] - k[929]*y[IDX_NHI];
    data[4056] = 0.0 + k[1810]*y[IDX_NI];
    data[4057] = 0.0 + k[260]*y[IDX_NHII] + k[1298]*y[IDX_NII] +
        k[1376]*y[IDX_NH2II];
    data[4058] = 0.0 - k[1449]*y[IDX_NHI];
    data[4059] = 0.0 + k[261]*y[IDX_NHII] + k[996]*y[IDX_CNII] +
        k[1378]*y[IDX_NH2II] - k[1841]*y[IDX_NHI];
    data[4060] = 0.0 - k[1450]*y[IDX_NHI];
    data[4061] = 0.0 + k[1300]*y[IDX_NII] + k[1302]*y[IDX_NII] +
        k[1381]*y[IDX_NH2II];
    data[4062] = 0.0 - k[1058]*y[IDX_NHI];
    data[4063] = 0.0 + k[1385]*y[IDX_NH2II] + k[1890]*y[IDX_OI];
    data[4064] = 0.0 - k[1451]*y[IDX_NHI];
    data[4065] = 0.0 + k[1386]*y[IDX_NH2II] + k[1811]*y[IDX_NI];
    data[4066] = 0.0 - k[1452]*y[IDX_NHI];
    data[4067] = 0.0 - k[1256]*y[IDX_NHI];
    data[4068] = 0.0 + k[1387]*y[IDX_NH2II];
    data[4069] = 0.0 + k[415] + k[2037];
    data[4070] = 0.0 + k[1716]*y[IDX_COI] + k[1757]*y[IDX_HI] +
        k[1815]*y[IDX_NI] + k[1898]*y[IDX_OI];
    data[4071] = 0.0 - k[1453]*y[IDX_NHI];
    data[4072] = 0.0 + k[1817]*y[IDX_NI];
    data[4073] = 0.0 + k[1683]*y[IDX_CHI] + k[1728]*y[IDX_H2I] +
        k[1791]*y[IDX_C2H3I] + k[1793]*y[IDX_C2H5I] + k[1803]*y[IDX_CH2I] +
        k[1810]*y[IDX_H2CNI] + k[1811]*y[IDX_HCOI] + k[1815]*y[IDX_HNOI] +
        k[1817]*y[IDX_HSI] - k[1819]*y[IDX_NHI] + k[1826]*y[IDX_O2HI] +
        k[1828]*y[IDX_OHI];
    data[4074] = 0.0 - k[247]*y[IDX_NHI] + k[1289]*y[IDX_CH3OHI] +
        k[1290]*y[IDX_CH3OHI] + k[1298]*y[IDX_H2COI] + k[1300]*y[IDX_H2SI] +
        k[1302]*y[IDX_H2SI] + k[1307]*y[IDX_NH3I] - k[1308]*y[IDX_NHI];
    data[4075] = 0.0 + k[1627]*y[IDX_CH2I];
    data[4076] = 0.0 - k[296]*y[IDX_NHI];
    data[4077] = 0.0 + k[566]*y[IDX_EM] - k[1454]*y[IDX_NHI];
    data[4078] = 0.0 - k[132]*y[IDX_HII] - k[168]*y[IDX_H2II] -
        k[247]*y[IDX_NII] - k[294]*y[IDX_CNII] - k[295]*y[IDX_COII] -
        k[296]*y[IDX_N2II] - k[297]*y[IDX_OII] - k[429] - k[430] -
        k[631]*y[IDX_CII] - k[731]*y[IDX_CHII] - k[929]*y[IDX_H2II] -
        k[1058]*y[IDX_H3II] - k[1256]*y[IDX_HeII] - k[1308]*y[IDX_NII] -
        k[1366]*y[IDX_NHII] - k[1444]*y[IDX_C2II] - k[1445]*y[IDX_C2II] -
        k[1446]*y[IDX_CH3II] - k[1447]*y[IDX_CH5II] - k[1448]*y[IDX_COII] -
        k[1449]*y[IDX_H2COII] - k[1450]*y[IDX_H2OII] - k[1451]*y[IDX_HCNII] -
        k[1452]*y[IDX_HCOII] - k[1453]*y[IDX_HNOII] - k[1454]*y[IDX_N2HII] -
        k[1455]*y[IDX_NH2II] - k[1456]*y[IDX_NH3II] - k[1457]*y[IDX_OII] -
        k[1458]*y[IDX_O2II] - k[1459]*y[IDX_O2HII] - k[1460]*y[IDX_OHII] -
        k[1461]*y[IDX_SII] - k[1602]*y[IDX_CI] - k[1603]*y[IDX_CI] -
        k[1730]*y[IDX_H2I] - k[1762]*y[IDX_HI] - k[1819]*y[IDX_NI] -
        k[1839]*y[IDX_CH4I] - k[1840]*y[IDX_CNI] - k[1841]*y[IDX_H2OI] -
        k[1842]*y[IDX_NH3I] - k[1843]*y[IDX_NHI] - k[1843]*y[IDX_NHI] -
        k[1843]*y[IDX_NHI] - k[1843]*y[IDX_NHI] - k[1844]*y[IDX_NHI] -
        k[1844]*y[IDX_NHI] - k[1844]*y[IDX_NHI] - k[1844]*y[IDX_NHI] -
        k[1845]*y[IDX_NHI] - k[1845]*y[IDX_NHI] - k[1845]*y[IDX_NHI] -
        k[1845]*y[IDX_NHI] - k[1846]*y[IDX_NO2I] - k[1847]*y[IDX_NOI] -
        k[1848]*y[IDX_NOI] - k[1849]*y[IDX_O2I] - k[1850]*y[IDX_O2I] -
        k[1851]*y[IDX_OI] - k[1852]*y[IDX_OI] - k[1853]*y[IDX_OHI] -
        k[1854]*y[IDX_OHI] - k[1855]*y[IDX_OHI] - k[1856]*y[IDX_SI] -
        k[1857]*y[IDX_SI] - k[2054] - k[2055] - k[2222];
    data[4079] = 0.0 + k[260]*y[IDX_H2COI] + k[261]*y[IDX_H2OI] +
        k[262]*y[IDX_NH3I] + k[263]*y[IDX_NOI] + k[264]*y[IDX_O2I] +
        k[265]*y[IDX_SI] - k[1366]*y[IDX_NHI];
    data[4080] = 0.0 + k[425] + k[1388]*y[IDX_NH2II] + k[1398]*y[IDX_COII] +
        k[1409]*y[IDX_NH3II] + k[1601]*y[IDX_CI] + k[1656]*y[IDX_CH3I] +
        k[1760]*y[IDX_HI] + k[1836]*y[IDX_OHI] + k[1903]*y[IDX_OI] + k[2050];
    data[4081] = 0.0 + k[569]*y[IDX_EM] + k[767]*y[IDX_CH2I] +
        k[855]*y[IDX_CHI] + k[1374]*y[IDX_C2I] + k[1375]*y[IDX_C2HI] +
        k[1376]*y[IDX_H2COI] + k[1378]*y[IDX_H2OI] + k[1381]*y[IDX_H2SI] +
        k[1385]*y[IDX_HCNI] + k[1386]*y[IDX_HCOI] + k[1387]*y[IDX_HNCI] +
        k[1388]*y[IDX_NH2I] + k[1389]*y[IDX_NH3I] + k[1393]*y[IDX_SI] -
        k[1455]*y[IDX_NHI];
    data[4082] = 0.0 + k[262]*y[IDX_NHII] + k[428] + k[1307]*y[IDX_NII] +
        k[1389]*y[IDX_NH2II] - k[1842]*y[IDX_NHI] + k[2053];
    data[4083] = 0.0 + k[571]*y[IDX_EM] + k[1409]*y[IDX_NH2I] +
        k[1412]*y[IDX_C2I] - k[1456]*y[IDX_NHI];
    data[4084] = 0.0 + k[263]*y[IDX_NHII] + k[1764]*y[IDX_HI] -
        k[1847]*y[IDX_NHI] - k[1848]*y[IDX_NHI];
    data[4085] = 0.0 - k[1846]*y[IDX_NHI];
    data[4086] = 0.0 + k[1767]*y[IDX_HI];
    data[4087] = 0.0 - k[1851]*y[IDX_NHI] - k[1852]*y[IDX_NHI] +
        k[1890]*y[IDX_HCNI] + k[1898]*y[IDX_HNOI] + k[1903]*y[IDX_NH2I];
    data[4088] = 0.0 - k[297]*y[IDX_NHI] - k[1457]*y[IDX_NHI];
    data[4089] = 0.0 + k[264]*y[IDX_NHII] - k[1849]*y[IDX_NHI] -
        k[1850]*y[IDX_NHI];
    data[4090] = 0.0 - k[1458]*y[IDX_NHI];
    data[4091] = 0.0 + k[1826]*y[IDX_NI];
    data[4092] = 0.0 - k[1459]*y[IDX_NHI];
    data[4093] = 0.0 + k[1773]*y[IDX_HI];
    data[4094] = 0.0 + k[1828]*y[IDX_NI] + k[1836]*y[IDX_NH2I] -
        k[1853]*y[IDX_NHI] - k[1854]*y[IDX_NHI] - k[1855]*y[IDX_NHI];
    data[4095] = 0.0 - k[1460]*y[IDX_NHI];
    data[4096] = 0.0 + k[265]*y[IDX_NHII] + k[1393]*y[IDX_NH2II] -
        k[1856]*y[IDX_NHI] - k[1857]*y[IDX_NHI];
    data[4097] = 0.0 - k[1461]*y[IDX_NHI];
    data[4098] = 0.0 - k[699]*y[IDX_NHII];
    data[4099] = 0.0 - k[1345]*y[IDX_NHII] - k[1346]*y[IDX_NHII] -
        k[1347]*y[IDX_NHII];
    data[4100] = 0.0 - k[1348]*y[IDX_NHII];
    data[4101] = 0.0 - k[854]*y[IDX_NHII];
    data[4102] = 0.0 - k[766]*y[IDX_NHII];
    data[4103] = 0.0 - k[1349]*y[IDX_NHII];
    data[4104] = 0.0 + k[294]*y[IDX_NHI];
    data[4105] = 0.0 - k[1353]*y[IDX_NHII];
    data[4106] = 0.0 + k[295]*y[IDX_NHI];
    data[4107] = 0.0 - k[1350]*y[IDX_NHII] - k[1351]*y[IDX_NHII] -
        k[1352]*y[IDX_NHII];
    data[4108] = 0.0 - k[567]*y[IDX_NHII];
    data[4109] = 0.0 + k[132]*y[IDX_NHI];
    data[4110] = 0.0 + k[952]*y[IDX_NII] - k[954]*y[IDX_NHII] -
        k[955]*y[IDX_NHII];
    data[4111] = 0.0 + k[168]*y[IDX_NHI] + k[928]*y[IDX_NI];
    data[4112] = 0.0 - k[260]*y[IDX_NHII] - k[1354]*y[IDX_NHII] -
        k[1355]*y[IDX_NHII];
    data[4113] = 0.0 - k[261]*y[IDX_NHII] - k[1356]*y[IDX_NHII] -
        k[1357]*y[IDX_NHII] - k[1358]*y[IDX_NHII] - k[1359]*y[IDX_NHII];
    data[4114] = 0.0 + k[1301]*y[IDX_NII];
    data[4115] = 0.0 - k[1360]*y[IDX_NHII];
    data[4116] = 0.0 + k[1303]*y[IDX_NII] - k[1361]*y[IDX_NHII];
    data[4117] = 0.0 + k[1244]*y[IDX_HNCI] + k[1253]*y[IDX_NH2I] +
        k[1254]*y[IDX_NH3I];
    data[4118] = 0.0 + k[1244]*y[IDX_HeII] - k[1362]*y[IDX_NHII];
    data[4119] = 0.0 + k[928]*y[IDX_H2II] - k[1337]*y[IDX_NHII];
    data[4120] = 0.0 + k[247]*y[IDX_NHI] + k[952]*y[IDX_H2I] +
        k[1301]*y[IDX_H2SI] + k[1303]*y[IDX_HCOI];
    data[4121] = 0.0 - k[1363]*y[IDX_NHII];
    data[4122] = 0.0 + k[296]*y[IDX_NHI];
    data[4123] = 0.0 + k[132]*y[IDX_HII] + k[168]*y[IDX_H2II] +
        k[247]*y[IDX_NII] + k[294]*y[IDX_CNII] + k[295]*y[IDX_COII] +
        k[296]*y[IDX_N2II] + k[297]*y[IDX_OII] + k[430] - k[1366]*y[IDX_NHII] +
        k[2055];
    data[4124] = 0.0 - k[260]*y[IDX_H2COI] - k[261]*y[IDX_H2OI] -
        k[262]*y[IDX_NH3I] - k[263]*y[IDX_NOI] - k[264]*y[IDX_O2I] -
        k[265]*y[IDX_SI] - k[567]*y[IDX_EM] - k[699]*y[IDX_CI] -
        k[766]*y[IDX_CH2I] - k[854]*y[IDX_CHI] - k[954]*y[IDX_H2I] -
        k[955]*y[IDX_H2I] - k[1337]*y[IDX_NI] - k[1345]*y[IDX_C2I] -
        k[1346]*y[IDX_C2I] - k[1347]*y[IDX_C2I] - k[1348]*y[IDX_C2HI] -
        k[1349]*y[IDX_CNI] - k[1350]*y[IDX_CO2I] - k[1351]*y[IDX_CO2I] -
        k[1352]*y[IDX_CO2I] - k[1353]*y[IDX_COI] - k[1354]*y[IDX_H2COI] -
        k[1355]*y[IDX_H2COI] - k[1356]*y[IDX_H2OI] - k[1357]*y[IDX_H2OI] -
        k[1358]*y[IDX_H2OI] - k[1359]*y[IDX_H2OI] - k[1360]*y[IDX_HCNI] -
        k[1361]*y[IDX_HCOI] - k[1362]*y[IDX_HNCI] - k[1363]*y[IDX_N2I] -
        k[1364]*y[IDX_NH2I] - k[1365]*y[IDX_NH3I] - k[1366]*y[IDX_NHI] -
        k[1367]*y[IDX_NOI] - k[1368]*y[IDX_O2I] - k[1369]*y[IDX_O2I] -
        k[1370]*y[IDX_OI] - k[1371]*y[IDX_OHI] - k[1372]*y[IDX_SI] -
        k[1373]*y[IDX_SI] - k[2048] - k[2232];
    data[4125] = 0.0 + k[1253]*y[IDX_HeII] - k[1364]*y[IDX_NHII];
    data[4126] = 0.0 - k[262]*y[IDX_NHII] + k[1254]*y[IDX_HeII] -
        k[1365]*y[IDX_NHII];
    data[4127] = 0.0 - k[263]*y[IDX_NHII] - k[1367]*y[IDX_NHII];
    data[4128] = 0.0 - k[1370]*y[IDX_NHII];
    data[4129] = 0.0 + k[297]*y[IDX_NHI];
    data[4130] = 0.0 - k[264]*y[IDX_NHII] - k[1368]*y[IDX_NHII] -
        k[1369]*y[IDX_NHII];
    data[4131] = 0.0 - k[1371]*y[IDX_NHII];
    data[4132] = 0.0 - k[265]*y[IDX_NHII] - k[1372]*y[IDX_NHII] -
        k[1373]*y[IDX_NHII];
    data[4133] = 0.0 - k[1599]*y[IDX_NH2I] - k[1600]*y[IDX_NH2I] -
        k[1601]*y[IDX_NH2I];
    data[4134] = 0.0 - k[629]*y[IDX_NH2I];
    data[4135] = 0.0 - k[271]*y[IDX_NH2I] - k[1394]*y[IDX_NH2I];
    data[4136] = 0.0 - k[1395]*y[IDX_NH2I];
    data[4137] = 0.0 - k[1396]*y[IDX_NH2I];
    data[4138] = 0.0 + k[89]*y[IDX_NH2II];
    data[4139] = 0.0 - k[729]*y[IDX_NH2I];
    data[4140] = 0.0 + k[68]*y[IDX_NH2II] + k[768]*y[IDX_NH3II];
    data[4141] = 0.0 - k[1656]*y[IDX_NH2I] + k[1657]*y[IDX_NH3I];
    data[4142] = 0.0 - k[1833]*y[IDX_NH2I] + k[1839]*y[IDX_NHI];
    data[4143] = 0.0 - k[1397]*y[IDX_NH2I];
    data[4144] = 0.0 + k[1838]*y[IDX_NH3I];
    data[4145] = 0.0 - k[272]*y[IDX_NH2I];
    data[4146] = 0.0 - k[273]*y[IDX_NH2I] - k[1398]*y[IDX_NH2I] +
        k[1423]*y[IDX_NH3I];
    data[4147] = 0.0 + k[570]*y[IDX_NH3II] + k[572]*y[IDX_NH4II] +
        k[573]*y[IDX_NH4II];
    data[4148] = 0.0 + k[1755]*y[IDX_HNOI] - k[1760]*y[IDX_NH2I] +
        k[1761]*y[IDX_NH3I];
    data[4149] = 0.0 - k[130]*y[IDX_NH2I];
    data[4150] = 0.0 - k[1729]*y[IDX_NH2I] + k[1730]*y[IDX_NHI];
    data[4151] = 0.0 - k[166]*y[IDX_NH2I];
    data[4152] = 0.0 + k[1355]*y[IDX_NHII];
    data[4153] = 0.0 - k[1399]*y[IDX_NH2I];
    data[4154] = 0.0 + k[1841]*y[IDX_NHI];
    data[4155] = 0.0 - k[274]*y[IDX_NH2I] - k[1400]*y[IDX_NH2I];
    data[4156] = 0.0 + k[266]*y[IDX_NH2II];
    data[4157] = 0.0 - k[1056]*y[IDX_NH2I];
    data[4158] = 0.0 - k[1401]*y[IDX_NH2I];
    data[4159] = 0.0 - k[1402]*y[IDX_NH2I];
    data[4160] = 0.0 + k[1940]*y[IDX_OHI];
    data[4161] = 0.0 - k[1403]*y[IDX_NH2I] + k[1430]*y[IDX_NH3I];
    data[4162] = 0.0 - k[1404]*y[IDX_NH2I] - k[1405]*y[IDX_NH2I];
    data[4163] = 0.0 + k[267]*y[IDX_NH2II];
    data[4164] = 0.0 - k[1406]*y[IDX_NH2I];
    data[4165] = 0.0 - k[1252]*y[IDX_NH2I] - k[1253]*y[IDX_NH2I];
    data[4166] = 0.0 + k[1755]*y[IDX_HI];
    data[4167] = 0.0 - k[1407]*y[IDX_NH2I];
    data[4168] = 0.0 - k[245]*y[IDX_NH2I];
    data[4169] = 0.0 - k[275]*y[IDX_NH2I];
    data[4170] = 0.0 - k[1408]*y[IDX_NH2I];
    data[4171] = 0.0 + k[1730]*y[IDX_H2I] + k[1839]*y[IDX_CH4I] +
        k[1841]*y[IDX_H2OI] + k[1842]*y[IDX_NH3I] + k[1842]*y[IDX_NH3I] +
        k[1845]*y[IDX_NHI] + k[1845]*y[IDX_NHI] + k[1855]*y[IDX_OHI];
    data[4172] = 0.0 + k[1355]*y[IDX_H2COI] - k[1364]*y[IDX_NH2I];
    data[4173] = 0.0 - k[130]*y[IDX_HII] - k[166]*y[IDX_H2II] -
        k[245]*y[IDX_NII] - k[271]*y[IDX_C2II] - k[272]*y[IDX_CNII] -
        k[273]*y[IDX_COII] - k[274]*y[IDX_H2OII] - k[275]*y[IDX_N2II] -
        k[276]*y[IDX_O2II] - k[277]*y[IDX_OHII] - k[315]*y[IDX_OII] - k[424] -
        k[425] - k[629]*y[IDX_CII] - k[729]*y[IDX_CHII] - k[1056]*y[IDX_H3II] -
        k[1252]*y[IDX_HeII] - k[1253]*y[IDX_HeII] - k[1364]*y[IDX_NHII] -
        k[1388]*y[IDX_NH2II] - k[1394]*y[IDX_C2II] - k[1395]*y[IDX_C2HII] -
        k[1396]*y[IDX_C2H2II] - k[1397]*y[IDX_CH5II] - k[1398]*y[IDX_COII] -
        k[1399]*y[IDX_H2COII] - k[1400]*y[IDX_H2OII] - k[1401]*y[IDX_H3COII] -
        k[1402]*y[IDX_H3OII] - k[1403]*y[IDX_HCNII] - k[1404]*y[IDX_HCNHII] -
        k[1405]*y[IDX_HCNHII] - k[1406]*y[IDX_HCOII] - k[1407]*y[IDX_HNOII] -
        k[1408]*y[IDX_N2HII] - k[1409]*y[IDX_NH3II] - k[1410]*y[IDX_O2HII] -
        k[1411]*y[IDX_OHII] - k[1599]*y[IDX_CI] - k[1600]*y[IDX_CI] -
        k[1601]*y[IDX_CI] - k[1656]*y[IDX_CH3I] - k[1729]*y[IDX_H2I] -
        k[1760]*y[IDX_HI] - k[1833]*y[IDX_CH4I] - k[1834]*y[IDX_NOI] -
        k[1835]*y[IDX_NOI] - k[1836]*y[IDX_OHI] - k[1837]*y[IDX_OHI] -
        k[1902]*y[IDX_OI] - k[1903]*y[IDX_OI] - k[2049] - k[2050] - k[2250];
    data[4174] = 0.0 + k[68]*y[IDX_CH2I] + k[89]*y[IDX_CHI] +
        k[266]*y[IDX_H2SI] + k[267]*y[IDX_HCOI] + k[268]*y[IDX_NH3I] +
        k[269]*y[IDX_NOI] + k[270]*y[IDX_SI] - k[1388]*y[IDX_NH2I];
    data[4175] = 0.0 + k[268]*y[IDX_NH2II] + k[426] + k[1417]*y[IDX_NH3II] +
        k[1423]*y[IDX_COII] + k[1430]*y[IDX_HCNII] + k[1657]*y[IDX_CH3I] +
        k[1761]*y[IDX_HI] + k[1838]*y[IDX_CNI] + k[1842]*y[IDX_NHI] +
        k[1842]*y[IDX_NHI] + k[1904]*y[IDX_OI] + k[1944]*y[IDX_OHI] + k[2051];
    data[4176] = 0.0 + k[570]*y[IDX_EM] + k[768]*y[IDX_CH2I] -
        k[1409]*y[IDX_NH2I] + k[1417]*y[IDX_NH3I];
    data[4177] = 0.0 + k[572]*y[IDX_EM] + k[573]*y[IDX_EM];
    data[4178] = 0.0 + k[269]*y[IDX_NH2II] - k[1834]*y[IDX_NH2I] -
        k[1835]*y[IDX_NH2I];
    data[4179] = 0.0 - k[1902]*y[IDX_NH2I] - k[1903]*y[IDX_NH2I] +
        k[1904]*y[IDX_NH3I];
    data[4180] = 0.0 - k[315]*y[IDX_NH2I];
    data[4181] = 0.0 - k[276]*y[IDX_NH2I];
    data[4182] = 0.0 - k[1410]*y[IDX_NH2I];
    data[4183] = 0.0 - k[1836]*y[IDX_NH2I] - k[1837]*y[IDX_NH2I] +
        k[1855]*y[IDX_NHI] + k[1940]*y[IDX_HCNI] + k[1944]*y[IDX_NH3I];
    data[4184] = 0.0 - k[277]*y[IDX_NH2I] - k[1411]*y[IDX_NH2I];
    data[4185] = 0.0 + k[270]*y[IDX_NH2II];
    data[4186] = 0.0 - k[1374]*y[IDX_NH2II];
    data[4187] = 0.0 + k[271]*y[IDX_NH2I];
    data[4188] = 0.0 - k[1375]*y[IDX_NH2II];
    data[4189] = 0.0 - k[89]*y[IDX_NH2II] - k[855]*y[IDX_NH2II];
    data[4190] = 0.0 - k[68]*y[IDX_NH2II] - k[767]*y[IDX_NH2II];
    data[4191] = 0.0 + k[1447]*y[IDX_NHI];
    data[4192] = 0.0 + k[272]*y[IDX_NH2I];
    data[4193] = 0.0 + k[273]*y[IDX_NH2I];
    data[4194] = 0.0 - k[568]*y[IDX_NH2II] - k[569]*y[IDX_NH2II];
    data[4195] = 0.0 + k[130]*y[IDX_NH2I] + k[900]*y[IDX_HNCOI];
    data[4196] = 0.0 + k[955]*y[IDX_NHII] - k[956]*y[IDX_NH2II];
    data[4197] = 0.0 + k[166]*y[IDX_NH2I] + k[929]*y[IDX_NHI];
    data[4198] = 0.0 - k[1376]*y[IDX_NH2II] - k[1377]*y[IDX_NH2II];
    data[4199] = 0.0 + k[1359]*y[IDX_NHII] - k[1378]*y[IDX_NH2II] -
        k[1379]*y[IDX_NH2II] - k[1380]*y[IDX_NH2II];
    data[4200] = 0.0 + k[274]*y[IDX_NH2I];
    data[4201] = 0.0 - k[266]*y[IDX_NH2II] - k[1381]*y[IDX_NH2II] -
        k[1382]*y[IDX_NH2II] - k[1383]*y[IDX_NH2II] - k[1384]*y[IDX_NH2II];
    data[4202] = 0.0 + k[1058]*y[IDX_NHI];
    data[4203] = 0.0 - k[1385]*y[IDX_NH2II];
    data[4204] = 0.0 + k[1451]*y[IDX_NHI];
    data[4205] = 0.0 - k[267]*y[IDX_NH2II] - k[1386]*y[IDX_NH2II];
    data[4206] = 0.0 + k[1452]*y[IDX_NHI];
    data[4207] = 0.0 + k[1255]*y[IDX_NH3I];
    data[4208] = 0.0 - k[1387]*y[IDX_NH2II];
    data[4209] = 0.0 + k[900]*y[IDX_HII];
    data[4210] = 0.0 + k[1453]*y[IDX_NHI];
    data[4211] = 0.0 - k[1338]*y[IDX_NH2II];
    data[4212] = 0.0 + k[245]*y[IDX_NH2I] + k[1307]*y[IDX_NH3I];
    data[4213] = 0.0 + k[275]*y[IDX_NH2I];
    data[4214] = 0.0 + k[1454]*y[IDX_NHI];
    data[4215] = 0.0 + k[929]*y[IDX_H2II] + k[1058]*y[IDX_H3II] +
        k[1366]*y[IDX_NHII] + k[1447]*y[IDX_CH5II] + k[1451]*y[IDX_HCNII] +
        k[1452]*y[IDX_HCOII] + k[1453]*y[IDX_HNOII] + k[1454]*y[IDX_N2HII] -
        k[1455]*y[IDX_NH2II] + k[1459]*y[IDX_O2HII] + k[1460]*y[IDX_OHII];
    data[4216] = 0.0 + k[955]*y[IDX_H2I] + k[1359]*y[IDX_H2OI] +
        k[1366]*y[IDX_NHI];
    data[4217] = 0.0 + k[130]*y[IDX_HII] + k[166]*y[IDX_H2II] +
        k[245]*y[IDX_NII] + k[271]*y[IDX_C2II] + k[272]*y[IDX_CNII] +
        k[273]*y[IDX_COII] + k[274]*y[IDX_H2OII] + k[275]*y[IDX_N2II] +
        k[276]*y[IDX_O2II] + k[277]*y[IDX_OHII] + k[315]*y[IDX_OII] + k[424] -
        k[1388]*y[IDX_NH2II] + k[2049];
    data[4218] = 0.0 - k[68]*y[IDX_CH2I] - k[89]*y[IDX_CHI] -
        k[266]*y[IDX_H2SI] - k[267]*y[IDX_HCOI] - k[268]*y[IDX_NH3I] -
        k[269]*y[IDX_NOI] - k[270]*y[IDX_SI] - k[568]*y[IDX_EM] -
        k[569]*y[IDX_EM] - k[767]*y[IDX_CH2I] - k[855]*y[IDX_CHI] -
        k[956]*y[IDX_H2I] - k[1338]*y[IDX_NI] - k[1374]*y[IDX_C2I] -
        k[1375]*y[IDX_C2HI] - k[1376]*y[IDX_H2COI] - k[1377]*y[IDX_H2COI] -
        k[1378]*y[IDX_H2OI] - k[1379]*y[IDX_H2OI] - k[1380]*y[IDX_H2OI] -
        k[1381]*y[IDX_H2SI] - k[1382]*y[IDX_H2SI] - k[1383]*y[IDX_H2SI] -
        k[1384]*y[IDX_H2SI] - k[1385]*y[IDX_HCNI] - k[1386]*y[IDX_HCOI] -
        k[1387]*y[IDX_HNCI] - k[1388]*y[IDX_NH2I] - k[1389]*y[IDX_NH3I] -
        k[1390]*y[IDX_O2I] - k[1391]*y[IDX_O2I] - k[1392]*y[IDX_SI] -
        k[1393]*y[IDX_SI] - k[1455]*y[IDX_NHI] - k[1507]*y[IDX_OI] - k[2238];
    data[4219] = 0.0 - k[268]*y[IDX_NH2II] + k[1255]*y[IDX_HeII] +
        k[1307]*y[IDX_NII] - k[1389]*y[IDX_NH2II];
    data[4220] = 0.0 - k[269]*y[IDX_NH2II];
    data[4221] = 0.0 - k[1507]*y[IDX_NH2II];
    data[4222] = 0.0 + k[315]*y[IDX_NH2I];
    data[4223] = 0.0 - k[1390]*y[IDX_NH2II] - k[1391]*y[IDX_NH2II];
    data[4224] = 0.0 + k[276]*y[IDX_NH2I];
    data[4225] = 0.0 + k[1459]*y[IDX_NHI];
    data[4226] = 0.0 + k[277]*y[IDX_NH2I] + k[1460]*y[IDX_NHI];
    data[4227] = 0.0 - k[270]*y[IDX_NH2II] - k[1392]*y[IDX_NH2II] -
        k[1393]*y[IDX_NH2II];
    data[4228] = 0.0 + k[2307] + k[2308] + k[2309] + k[2310];
    data[4229] = 0.0 - k[21]*y[IDX_NH3I] - k[630]*y[IDX_NH3I];
    data[4230] = 0.0 - k[1418]*y[IDX_NH3I];
    data[4231] = 0.0 - k[282]*y[IDX_NH3I] - k[1419]*y[IDX_NH3I];
    data[4232] = 0.0 - k[1420]*y[IDX_NH3I];
    data[4233] = 0.0 - k[1421]*y[IDX_NH3I];
    data[4234] = 0.0 - k[57]*y[IDX_NH3I] - k[730]*y[IDX_NH3I];
    data[4235] = 0.0 - k[748]*y[IDX_NH3I];
    data[4236] = 0.0 - k[1657]*y[IDX_NH3I];
    data[4237] = 0.0 - k[781]*y[IDX_NH3I];
    data[4238] = 0.0 - k[792]*y[IDX_NH3I];
    data[4239] = 0.0 + k[1833]*y[IDX_NH2I];
    data[4240] = 0.0 - k[78]*y[IDX_NH3I] - k[801]*y[IDX_NH3I];
    data[4241] = 0.0 - k[1422]*y[IDX_NH3I];
    data[4242] = 0.0 - k[1838]*y[IDX_NH3I];
    data[4243] = 0.0 - k[283]*y[IDX_NH3I] - k[1423]*y[IDX_NH3I];
    data[4244] = 0.0 + k[574]*y[IDX_NH4II];
    data[4245] = 0.0 - k[1761]*y[IDX_NH3I];
    data[4246] = 0.0 - k[131]*y[IDX_NH3I];
    data[4247] = 0.0 + k[1729]*y[IDX_NH2I];
    data[4248] = 0.0 - k[167]*y[IDX_NH3I];
    data[4249] = 0.0 - k[284]*y[IDX_NH3I] - k[1424]*y[IDX_NH3I];
    data[4250] = 0.0 - k[285]*y[IDX_NH3I] - k[1425]*y[IDX_NH3I];
    data[4251] = 0.0 + k[1382]*y[IDX_NH2II];
    data[4252] = 0.0 - k[286]*y[IDX_NH3I] - k[1426]*y[IDX_NH3I];
    data[4253] = 0.0 - k[1057]*y[IDX_NH3I];
    data[4254] = 0.0 - k[1427]*y[IDX_NH3I];
    data[4255] = 0.0 - k[1428]*y[IDX_NH3I];
    data[4256] = 0.0 - k[1429]*y[IDX_NH3I];
    data[4257] = 0.0 - k[287]*y[IDX_NH3I] - k[1430]*y[IDX_NH3I];
    data[4258] = 0.0 - k[1431]*y[IDX_NH3I] - k[1432]*y[IDX_NH3I];
    data[4259] = 0.0 + k[278]*y[IDX_NH3II];
    data[4260] = 0.0 - k[1433]*y[IDX_NH3I];
    data[4261] = 0.0 - k[1434]*y[IDX_NH3I];
    data[4262] = 0.0 - k[1435]*y[IDX_NH3I];
    data[4263] = 0.0 - k[216]*y[IDX_NH3I] - k[1254]*y[IDX_NH3I] -
        k[1255]*y[IDX_NH3I];
    data[4264] = 0.0 - k[1436]*y[IDX_NH3I];
    data[4265] = 0.0 - k[288]*y[IDX_NH3I] - k[1437]*y[IDX_NH3I];
    data[4266] = 0.0 - k[1439]*y[IDX_NH3I];
    data[4267] = 0.0 - k[1438]*y[IDX_NH3I];
    data[4268] = 0.0 + k[279]*y[IDX_NH3II];
    data[4269] = 0.0 - k[246]*y[IDX_NH3I] - k[1306]*y[IDX_NH3I] -
        k[1307]*y[IDX_NH3I];
    data[4270] = 0.0 - k[289]*y[IDX_NH3I];
    data[4271] = 0.0 - k[1440]*y[IDX_NH3I];
    data[4272] = 0.0 - k[1842]*y[IDX_NH3I];
    data[4273] = 0.0 - k[262]*y[IDX_NH3I] - k[1365]*y[IDX_NH3I];
    data[4274] = 0.0 + k[1729]*y[IDX_H2I] + k[1833]*y[IDX_CH4I] +
        k[1837]*y[IDX_OHI];
    data[4275] = 0.0 - k[268]*y[IDX_NH3I] + k[1382]*y[IDX_H2SI] -
        k[1389]*y[IDX_NH3I];
    data[4276] = 0.0 - k[21]*y[IDX_CII] - k[57]*y[IDX_CHII] -
        k[78]*y[IDX_CH4II] - k[131]*y[IDX_HII] - k[167]*y[IDX_H2II] -
        k[216]*y[IDX_HeII] - k[246]*y[IDX_NII] - k[262]*y[IDX_NHII] -
        k[268]*y[IDX_NH2II] - k[282]*y[IDX_C2H2II] - k[283]*y[IDX_COII] -
        k[284]*y[IDX_H2COII] - k[285]*y[IDX_H2OII] - k[286]*y[IDX_H2SII] -
        k[287]*y[IDX_HCNII] - k[288]*y[IDX_HSII] - k[289]*y[IDX_N2II] -
        k[290]*y[IDX_O2II] - k[291]*y[IDX_OCSII] - k[292]*y[IDX_SII] -
        k[293]*y[IDX_SOII] - k[316]*y[IDX_OII] - k[335]*y[IDX_OHII] - k[426] -
        k[427] - k[428] - k[630]*y[IDX_CII] - k[730]*y[IDX_CHII] -
        k[748]*y[IDX_CH2II] - k[781]*y[IDX_CH3II] - k[792]*y[IDX_CH3OH2II] -
        k[801]*y[IDX_CH4II] - k[1057]*y[IDX_H3II] - k[1254]*y[IDX_HeII] -
        k[1255]*y[IDX_HeII] - k[1306]*y[IDX_NII] - k[1307]*y[IDX_NII] -
        k[1365]*y[IDX_NHII] - k[1389]*y[IDX_NH2II] - k[1417]*y[IDX_NH3II] -
        k[1418]*y[IDX_C2HII] - k[1419]*y[IDX_C2H2II] - k[1420]*y[IDX_C2H5OH2II]
        - k[1421]*y[IDX_C2NII] - k[1422]*y[IDX_CH5II] - k[1423]*y[IDX_COII] -
        k[1424]*y[IDX_H2COII] - k[1425]*y[IDX_H2OII] - k[1426]*y[IDX_H2SII] -
        k[1427]*y[IDX_H3COII] - k[1428]*y[IDX_H3OII] - k[1429]*y[IDX_H3SII] -
        k[1430]*y[IDX_HCNII] - k[1431]*y[IDX_HCNHII] - k[1432]*y[IDX_HCNHII] -
        k[1433]*y[IDX_HCOII] - k[1434]*y[IDX_HCO2II] - k[1435]*y[IDX_HCSII] -
        k[1436]*y[IDX_HNOII] - k[1437]*y[IDX_HSII] - k[1438]*y[IDX_HSO2II] -
        k[1439]*y[IDX_HSiSII] - k[1440]*y[IDX_N2HII] - k[1441]*y[IDX_O2HII] -
        k[1442]*y[IDX_SiHII] - k[1443]*y[IDX_SiOHII] - k[1530]*y[IDX_OHII] -
        k[1657]*y[IDX_CH3I] - k[1761]*y[IDX_HI] - k[1838]*y[IDX_CNI] -
        k[1842]*y[IDX_NHI] - k[1904]*y[IDX_OI] - k[1944]*y[IDX_OHI] - k[2051] -
        k[2052] - k[2053] - k[2225];
    data[4277] = 0.0 + k[278]*y[IDX_HCOI] + k[279]*y[IDX_MgI] +
        k[280]*y[IDX_NOI] + k[281]*y[IDX_SiI] - k[1417]*y[IDX_NH3I];
    data[4278] = 0.0 + k[574]*y[IDX_EM];
    data[4279] = 0.0 + k[280]*y[IDX_NH3II];
    data[4280] = 0.0 - k[1904]*y[IDX_NH3I];
    data[4281] = 0.0 - k[316]*y[IDX_NH3I];
    data[4282] = 0.0 - k[290]*y[IDX_NH3I];
    data[4283] = 0.0 - k[1441]*y[IDX_NH3I];
    data[4284] = 0.0 - k[291]*y[IDX_NH3I];
    data[4285] = 0.0 + k[1837]*y[IDX_NH2I] - k[1944]*y[IDX_NH3I];
    data[4286] = 0.0 - k[335]*y[IDX_NH3I] - k[1530]*y[IDX_NH3I];
    data[4287] = 0.0 - k[292]*y[IDX_NH3I];
    data[4288] = 0.0 + k[281]*y[IDX_NH3II];
    data[4289] = 0.0 - k[1442]*y[IDX_NH3I];
    data[4290] = 0.0 - k[1443]*y[IDX_NH3I];
    data[4291] = 0.0 - k[293]*y[IDX_NH3I];
    data[4292] = 0.0 + k[21]*y[IDX_NH3I];
    data[4293] = 0.0 - k[1412]*y[IDX_NH3II];
    data[4294] = 0.0 + k[1395]*y[IDX_NH2I];
    data[4295] = 0.0 + k[282]*y[IDX_NH3I] + k[1396]*y[IDX_NH2I];
    data[4296] = 0.0 - k[856]*y[IDX_NH3II];
    data[4297] = 0.0 + k[57]*y[IDX_NH3I];
    data[4298] = 0.0 - k[768]*y[IDX_NH3II];
    data[4299] = 0.0 - k[818]*y[IDX_NH3II];
    data[4300] = 0.0 + k[78]*y[IDX_NH3I];
    data[4301] = 0.0 + k[1397]*y[IDX_NH2I];
    data[4302] = 0.0 + k[283]*y[IDX_NH3I];
    data[4303] = 0.0 - k[570]*y[IDX_NH3II] - k[571]*y[IDX_NH3II];
    data[4304] = 0.0 + k[131]*y[IDX_NH3I];
    data[4305] = 0.0 + k[956]*y[IDX_NH2II] - k[957]*y[IDX_NH3II];
    data[4306] = 0.0 + k[167]*y[IDX_NH3I];
    data[4307] = 0.0 + k[1377]*y[IDX_NH2II] - k[1413]*y[IDX_NH3II];
    data[4308] = 0.0 + k[284]*y[IDX_NH3I] + k[1399]*y[IDX_NH2I];
    data[4309] = 0.0 + k[1358]*y[IDX_NHII] + k[1379]*y[IDX_NH2II] -
        k[1414]*y[IDX_NH3II];
    data[4310] = 0.0 + k[285]*y[IDX_NH3I] + k[1400]*y[IDX_NH2I];
    data[4311] = 0.0 + k[1383]*y[IDX_NH2II] - k[1415]*y[IDX_NH3II];
    data[4312] = 0.0 + k[286]*y[IDX_NH3I];
    data[4313] = 0.0 + k[1056]*y[IDX_NH2I];
    data[4314] = 0.0 + k[1401]*y[IDX_NH2I];
    data[4315] = 0.0 + k[1402]*y[IDX_NH2I];
    data[4316] = 0.0 + k[287]*y[IDX_NH3I] + k[1403]*y[IDX_NH2I];
    data[4317] = 0.0 + k[1404]*y[IDX_NH2I] + k[1405]*y[IDX_NH2I];
    data[4318] = 0.0 - k[278]*y[IDX_NH3II] - k[1416]*y[IDX_NH3II];
    data[4319] = 0.0 + k[1406]*y[IDX_NH2I];
    data[4320] = 0.0 + k[216]*y[IDX_NH3I];
    data[4321] = 0.0 + k[1407]*y[IDX_NH2I];
    data[4322] = 0.0 + k[288]*y[IDX_NH3I];
    data[4323] = 0.0 - k[279]*y[IDX_NH3II];
    data[4324] = 0.0 + k[246]*y[IDX_NH3I];
    data[4325] = 0.0 + k[289]*y[IDX_NH3I];
    data[4326] = 0.0 + k[1408]*y[IDX_NH2I];
    data[4327] = 0.0 + k[1455]*y[IDX_NH2II] - k[1456]*y[IDX_NH3II];
    data[4328] = 0.0 + k[262]*y[IDX_NH3I] + k[1358]*y[IDX_H2OI] +
        k[1364]*y[IDX_NH2I];
    data[4329] = 0.0 + k[1056]*y[IDX_H3II] + k[1364]*y[IDX_NHII] +
        k[1388]*y[IDX_NH2II] + k[1395]*y[IDX_C2HII] + k[1396]*y[IDX_C2H2II] +
        k[1397]*y[IDX_CH5II] + k[1399]*y[IDX_H2COII] + k[1400]*y[IDX_H2OII] +
        k[1401]*y[IDX_H3COII] + k[1402]*y[IDX_H3OII] + k[1403]*y[IDX_HCNII] +
        k[1404]*y[IDX_HCNHII] + k[1405]*y[IDX_HCNHII] + k[1406]*y[IDX_HCOII] +
        k[1407]*y[IDX_HNOII] + k[1408]*y[IDX_N2HII] - k[1409]*y[IDX_NH3II] +
        k[1410]*y[IDX_O2HII] + k[1411]*y[IDX_OHII];
    data[4330] = 0.0 + k[268]*y[IDX_NH3I] + k[956]*y[IDX_H2I] +
        k[1377]*y[IDX_H2COI] + k[1379]*y[IDX_H2OI] + k[1383]*y[IDX_H2SI] +
        k[1388]*y[IDX_NH2I] + k[1455]*y[IDX_NHI];
    data[4331] = 0.0 + k[21]*y[IDX_CII] + k[57]*y[IDX_CHII] +
        k[78]*y[IDX_CH4II] + k[131]*y[IDX_HII] + k[167]*y[IDX_H2II] +
        k[216]*y[IDX_HeII] + k[246]*y[IDX_NII] + k[262]*y[IDX_NHII] +
        k[268]*y[IDX_NH2II] + k[282]*y[IDX_C2H2II] + k[283]*y[IDX_COII] +
        k[284]*y[IDX_H2COII] + k[285]*y[IDX_H2OII] + k[286]*y[IDX_H2SII] +
        k[287]*y[IDX_HCNII] + k[288]*y[IDX_HSII] + k[289]*y[IDX_N2II] +
        k[290]*y[IDX_O2II] + k[291]*y[IDX_OCSII] + k[292]*y[IDX_SII] +
        k[293]*y[IDX_SOII] + k[316]*y[IDX_OII] + k[335]*y[IDX_OHII] + k[427] -
        k[1417]*y[IDX_NH3II] + k[2052];
    data[4332] = 0.0 - k[278]*y[IDX_HCOI] - k[279]*y[IDX_MgI] -
        k[280]*y[IDX_NOI] - k[281]*y[IDX_SiI] - k[570]*y[IDX_EM] -
        k[571]*y[IDX_EM] - k[768]*y[IDX_CH2I] - k[818]*y[IDX_CH4I] -
        k[856]*y[IDX_CHI] - k[957]*y[IDX_H2I] - k[1409]*y[IDX_NH2I] -
        k[1412]*y[IDX_C2I] - k[1413]*y[IDX_H2COI] - k[1414]*y[IDX_H2OI] -
        k[1415]*y[IDX_H2SI] - k[1416]*y[IDX_HCOI] - k[1417]*y[IDX_NH3I] -
        k[1456]*y[IDX_NHI] - k[1508]*y[IDX_OI] - k[1546]*y[IDX_OHI] - k[2243];
    data[4333] = 0.0 - k[280]*y[IDX_NH3II];
    data[4334] = 0.0 - k[1508]*y[IDX_NH3II];
    data[4335] = 0.0 + k[316]*y[IDX_NH3I];
    data[4336] = 0.0 + k[290]*y[IDX_NH3I];
    data[4337] = 0.0 + k[1410]*y[IDX_NH2I];
    data[4338] = 0.0 + k[291]*y[IDX_NH3I];
    data[4339] = 0.0 - k[1546]*y[IDX_NH3II];
    data[4340] = 0.0 + k[335]*y[IDX_NH3I] + k[1411]*y[IDX_NH2I];
    data[4341] = 0.0 + k[292]*y[IDX_NH3I];
    data[4342] = 0.0 - k[281]*y[IDX_NH3II];
    data[4343] = 0.0 + k[293]*y[IDX_NH3I];
    data[4344] = 0.0 + k[1418]*y[IDX_NH3I];
    data[4345] = 0.0 + k[1419]*y[IDX_NH3I];
    data[4346] = 0.0 + k[1420]*y[IDX_NH3I];
    data[4347] = 0.0 + k[856]*y[IDX_NH3II];
    data[4348] = 0.0 + k[730]*y[IDX_NH3I];
    data[4349] = 0.0 + k[748]*y[IDX_NH3I];
    data[4350] = 0.0 + k[781]*y[IDX_NH3I];
    data[4351] = 0.0 + k[792]*y[IDX_NH3I];
    data[4352] = 0.0 + k[818]*y[IDX_NH3II];
    data[4353] = 0.0 + k[801]*y[IDX_NH3I];
    data[4354] = 0.0 + k[1422]*y[IDX_NH3I];
    data[4355] = 0.0 - k[572]*y[IDX_NH4II] - k[573]*y[IDX_NH4II] -
        k[574]*y[IDX_NH4II];
    data[4356] = 0.0 + k[957]*y[IDX_NH3II];
    data[4357] = 0.0 + k[1413]*y[IDX_NH3II];
    data[4358] = 0.0 + k[1424]*y[IDX_NH3I];
    data[4359] = 0.0 + k[1380]*y[IDX_NH2II] + k[1414]*y[IDX_NH3II];
    data[4360] = 0.0 + k[1425]*y[IDX_NH3I];
    data[4361] = 0.0 + k[1384]*y[IDX_NH2II] + k[1415]*y[IDX_NH3II];
    data[4362] = 0.0 + k[1426]*y[IDX_NH3I];
    data[4363] = 0.0 + k[1057]*y[IDX_NH3I];
    data[4364] = 0.0 + k[1427]*y[IDX_NH3I];
    data[4365] = 0.0 + k[1428]*y[IDX_NH3I];
    data[4366] = 0.0 + k[1429]*y[IDX_NH3I];
    data[4367] = 0.0 + k[1431]*y[IDX_NH3I] + k[1432]*y[IDX_NH3I];
    data[4368] = 0.0 + k[1416]*y[IDX_NH3II];
    data[4369] = 0.0 + k[1433]*y[IDX_NH3I];
    data[4370] = 0.0 + k[1434]*y[IDX_NH3I];
    data[4371] = 0.0 + k[1435]*y[IDX_NH3I];
    data[4372] = 0.0 + k[1436]*y[IDX_NH3I];
    data[4373] = 0.0 + k[1437]*y[IDX_NH3I];
    data[4374] = 0.0 + k[1439]*y[IDX_NH3I];
    data[4375] = 0.0 + k[1438]*y[IDX_NH3I];
    data[4376] = 0.0 + k[1440]*y[IDX_NH3I];
    data[4377] = 0.0 + k[1456]*y[IDX_NH3II];
    data[4378] = 0.0 + k[1365]*y[IDX_NH3I];
    data[4379] = 0.0 + k[1409]*y[IDX_NH3II];
    data[4380] = 0.0 + k[1380]*y[IDX_H2OI] + k[1384]*y[IDX_H2SI] +
        k[1389]*y[IDX_NH3I];
    data[4381] = 0.0 + k[730]*y[IDX_CHII] + k[748]*y[IDX_CH2II] +
        k[781]*y[IDX_CH3II] + k[792]*y[IDX_CH3OH2II] + k[801]*y[IDX_CH4II] +
        k[1057]*y[IDX_H3II] + k[1365]*y[IDX_NHII] + k[1389]*y[IDX_NH2II] +
        k[1417]*y[IDX_NH3II] + k[1418]*y[IDX_C2HII] + k[1419]*y[IDX_C2H2II] +
        k[1420]*y[IDX_C2H5OH2II] + k[1422]*y[IDX_CH5II] + k[1424]*y[IDX_H2COII]
        + k[1425]*y[IDX_H2OII] + k[1426]*y[IDX_H2SII] + k[1427]*y[IDX_H3COII] +
        k[1428]*y[IDX_H3OII] + k[1429]*y[IDX_H3SII] + k[1431]*y[IDX_HCNHII] +
        k[1432]*y[IDX_HCNHII] + k[1433]*y[IDX_HCOII] + k[1434]*y[IDX_HCO2II] +
        k[1435]*y[IDX_HCSII] + k[1436]*y[IDX_HNOII] + k[1437]*y[IDX_HSII] +
        k[1438]*y[IDX_HSO2II] + k[1439]*y[IDX_HSiSII] + k[1440]*y[IDX_N2HII] +
        k[1441]*y[IDX_O2HII] + k[1442]*y[IDX_SiHII] + k[1443]*y[IDX_SiOHII] +
        k[1530]*y[IDX_OHII];
    data[4382] = 0.0 + k[818]*y[IDX_CH4I] + k[856]*y[IDX_CHI] +
        k[957]*y[IDX_H2I] + k[1409]*y[IDX_NH2I] + k[1413]*y[IDX_H2COI] +
        k[1414]*y[IDX_H2OI] + k[1415]*y[IDX_H2SI] + k[1416]*y[IDX_HCOI] +
        k[1417]*y[IDX_NH3I] + k[1456]*y[IDX_NHI] + k[1546]*y[IDX_OHI];
    data[4383] = 0.0 - k[572]*y[IDX_EM] - k[573]*y[IDX_EM] -
        k[574]*y[IDX_EM] - k[2288];
    data[4384] = 0.0 + k[1441]*y[IDX_NH3I];
    data[4385] = 0.0 + k[1546]*y[IDX_NH3II];
    data[4386] = 0.0 + k[1530]*y[IDX_NH3I];
    data[4387] = 0.0 + k[1442]*y[IDX_NH3I];
    data[4388] = 0.0 + k[1443]*y[IDX_NH3I];
    data[4389] = 0.0 + k[2363] + k[2364] + k[2365] + k[2366];
    data[4390] = 0.0 + k[696]*y[IDX_HNOII] - k[1604]*y[IDX_NOI] -
        k[1605]*y[IDX_NOI];
    data[4391] = 0.0 - k[22]*y[IDX_NOI];
    data[4392] = 0.0 + k[654]*y[IDX_HNOII];
    data[4393] = 0.0 - k[34]*y[IDX_NOI];
    data[4394] = 0.0 + k[681]*y[IDX_HNOII];
    data[4395] = 0.0 - k[40]*y[IDX_NOI];
    data[4396] = 0.0 - k[1574]*y[IDX_NOI];
    data[4397] = 0.0 - k[45]*y[IDX_NOI];
    data[4398] = 0.0 + k[850]*y[IDX_HNOII] + k[1680]*y[IDX_HNOI] -
        k[1684]*y[IDX_NOI] - k[1685]*y[IDX_NOI] - k[1686]*y[IDX_NOI];
    data[4399] = 0.0 - k[58]*y[IDX_NOI];
    data[4400] = 0.0 + k[764]*y[IDX_HNOII] + k[1626]*y[IDX_HNOI] +
        k[1628]*y[IDX_NO2I] - k[1629]*y[IDX_NOI] - k[1630]*y[IDX_NOI] -
        k[1631]*y[IDX_NOI];
    data[4401] = 0.0 - k[61]*y[IDX_NOI];
    data[4402] = 0.0 + k[1655]*y[IDX_HNOI] - k[1659]*y[IDX_NOI];
    data[4403] = 0.0 - k[74]*y[IDX_NOI];
    data[4404] = 0.0 + k[1292]*y[IDX_NII];
    data[4405] = 0.0 + k[813]*y[IDX_HNOII];
    data[4406] = 0.0 + k[869]*y[IDX_HNOII] + k[1708]*y[IDX_HNOI] +
        k[1709]*y[IDX_NO2I] - k[1710]*y[IDX_NOI] - k[1711]*y[IDX_NOI] +
        k[1712]*y[IDX_O2I] + k[1881]*y[IDX_OI];
    data[4407] = 0.0 - k[97]*y[IDX_NOI];
    data[4408] = 0.0 + k[876]*y[IDX_HNOII] + k[1717]*y[IDX_NO2I];
    data[4409] = 0.0 - k[104]*y[IDX_NOI];
    data[4410] = 0.0 + k[1173]*y[IDX_HNOII] + k[1296]*y[IDX_NII] +
        k[1808]*y[IDX_NI];
    data[4411] = 0.0 + k[511]*y[IDX_H2NOII] + k[549]*y[IDX_HNOII];
    data[4412] = 0.0 + k[1756]*y[IDX_HNOI] + k[1763]*y[IDX_NO2I] -
        k[1764]*y[IDX_NOI] - k[1765]*y[IDX_NOI];
    data[4413] = 0.0 - k[133]*y[IDX_NOI];
    data[4414] = 0.0 - k[169]*y[IDX_NOI] - k[930]*y[IDX_NOI];
    data[4415] = 0.0 + k[971]*y[IDX_HNOII];
    data[4416] = 0.0 - k[298]*y[IDX_NOI];
    data[4417] = 0.0 + k[511]*y[IDX_EM];
    data[4418] = 0.0 + k[1005]*y[IDX_HNOII];
    data[4419] = 0.0 - k[182]*y[IDX_NOI];
    data[4420] = 0.0 - k[299]*y[IDX_NOI];
    data[4421] = 0.0 - k[1060]*y[IDX_NOI];
    data[4422] = 0.0 + k[1125]*y[IDX_HNOII];
    data[4423] = 0.0 - k[197]*y[IDX_NOI];
    data[4424] = 0.0 + k[1159]*y[IDX_HNOII] + k[1782]*y[IDX_HNOI] -
        k[1783]*y[IDX_NOI];
    data[4425] = 0.0 + k[1246]*y[IDX_HNOI] - k[1257]*y[IDX_NOI] -
        k[1258]*y[IDX_NOI];
    data[4426] = 0.0 + k[1169]*y[IDX_HNOII];
    data[4427] = 0.0 + k[416] + k[1246]*y[IDX_HeII] + k[1626]*y[IDX_CH2I] +
        k[1655]*y[IDX_CH3I] + k[1680]*y[IDX_CHI] + k[1708]*y[IDX_CNI] +
        k[1756]*y[IDX_HI] + k[1782]*y[IDX_HCOI] + k[1815]*y[IDX_NI] +
        k[1897]*y[IDX_OI] + k[1942]*y[IDX_OHI] + k[2038];
    data[4428] = 0.0 - k[300]*y[IDX_NOI] + k[549]*y[IDX_EM] +
        k[654]*y[IDX_C2I] + k[681]*y[IDX_C2HI] + k[696]*y[IDX_CI] +
        k[764]*y[IDX_CH2I] + k[813]*y[IDX_CH4I] + k[850]*y[IDX_CHI] +
        k[869]*y[IDX_CNI] + k[876]*y[IDX_COI] + k[971]*y[IDX_H2COI] +
        k[1005]*y[IDX_H2OI] + k[1125]*y[IDX_HCNI] + k[1159]*y[IDX_HCOI] +
        k[1169]*y[IDX_HNCI] + k[1173]*y[IDX_CO2I] + k[1174]*y[IDX_SI] +
        k[1319]*y[IDX_N2I] + k[1407]*y[IDX_NH2I] + k[1436]*y[IDX_NH3I] +
        k[1453]*y[IDX_NHI] + k[1544]*y[IDX_OHI];
    data[4429] = 0.0 - k[301]*y[IDX_NOI];
    data[4430] = 0.0 + k[227]*y[IDX_NOII];
    data[4431] = 0.0 + k[1344]*y[IDX_SiOII] + k[1808]*y[IDX_CO2I] +
        k[1815]*y[IDX_HNOI] + k[1821]*y[IDX_NO2I] + k[1821]*y[IDX_NO2I] -
        k[1823]*y[IDX_NOI] + k[1825]*y[IDX_O2I] + k[1827]*y[IDX_OHI] +
        k[1831]*y[IDX_SOI];
    data[4432] = 0.0 - k[248]*y[IDX_NOI] + k[1292]*y[IDX_CH3OHI] +
        k[1296]*y[IDX_CO2I] - k[1309]*y[IDX_NOI] + k[1311]*y[IDX_O2I] +
        k[1312]*y[IDX_OCSI];
    data[4433] = 0.0 + k[1319]*y[IDX_HNOII] + k[1901]*y[IDX_OI];
    data[4434] = 0.0 - k[255]*y[IDX_NOI];
    data[4435] = 0.0 + k[1453]*y[IDX_HNOII] + k[1846]*y[IDX_NO2I] -
        k[1847]*y[IDX_NOI] - k[1848]*y[IDX_NOI] + k[1850]*y[IDX_O2I] +
        k[1851]*y[IDX_OI];
    data[4436] = 0.0 - k[263]*y[IDX_NOI] - k[1367]*y[IDX_NOI];
    data[4437] = 0.0 + k[1407]*y[IDX_HNOII] - k[1834]*y[IDX_NOI] -
        k[1835]*y[IDX_NOI];
    data[4438] = 0.0 - k[269]*y[IDX_NOI];
    data[4439] = 0.0 + k[1436]*y[IDX_HNOII];
    data[4440] = 0.0 - k[280]*y[IDX_NOI];
    data[4441] = 0.0 - k[22]*y[IDX_CII] - k[34]*y[IDX_C2II] -
        k[40]*y[IDX_C2HII] - k[45]*y[IDX_C2H2II] - k[58]*y[IDX_CHII] -
        k[61]*y[IDX_CH2II] - k[74]*y[IDX_CH3II] - k[97]*y[IDX_CNII] -
        k[104]*y[IDX_COII] - k[133]*y[IDX_HII] - k[169]*y[IDX_H2II] -
        k[182]*y[IDX_H2OII] - k[197]*y[IDX_HCNII] - k[248]*y[IDX_NII] -
        k[255]*y[IDX_N2II] - k[263]*y[IDX_NHII] - k[269]*y[IDX_NH2II] -
        k[280]*y[IDX_NH3II] - k[298]*y[IDX_H2COII] - k[299]*y[IDX_H2SII] -
        k[300]*y[IDX_HNOII] - k[301]*y[IDX_HSII] - k[302]*y[IDX_O2II] -
        k[303]*y[IDX_SII] - k[304]*y[IDX_S2II] - k[305]*y[IDX_SiOII] -
        k[336]*y[IDX_OHII] - k[432] - k[433] - k[930]*y[IDX_H2II] -
        k[1060]*y[IDX_H3II] - k[1257]*y[IDX_HeII] - k[1258]*y[IDX_HeII] -
        k[1309]*y[IDX_NII] - k[1367]*y[IDX_NHII] - k[1462]*y[IDX_O2HII] -
        k[1531]*y[IDX_OHII] - k[1574]*y[IDX_C2H2I] - k[1604]*y[IDX_CI] -
        k[1605]*y[IDX_CI] - k[1629]*y[IDX_CH2I] - k[1630]*y[IDX_CH2I] -
        k[1631]*y[IDX_CH2I] - k[1659]*y[IDX_CH3I] - k[1684]*y[IDX_CHI] -
        k[1685]*y[IDX_CHI] - k[1686]*y[IDX_CHI] - k[1710]*y[IDX_CNI] -
        k[1711]*y[IDX_CNI] - k[1764]*y[IDX_HI] - k[1765]*y[IDX_HI] -
        k[1783]*y[IDX_HCOI] - k[1823]*y[IDX_NI] - k[1834]*y[IDX_NH2I] -
        k[1835]*y[IDX_NH2I] - k[1847]*y[IDX_NHI] - k[1848]*y[IDX_NHI] -
        k[1858]*y[IDX_NOI] - k[1858]*y[IDX_NOI] - k[1858]*y[IDX_NOI] -
        k[1858]*y[IDX_NOI] - k[1859]*y[IDX_O2I] - k[1860]*y[IDX_OCNI] -
        k[1861]*y[IDX_SI] - k[1862]*y[IDX_SI] - k[1906]*y[IDX_OI] -
        k[1945]*y[IDX_OHI] - k[1958]*y[IDX_SiI] - k[2057] - k[2058] - k[2212];
    data[4442] = 0.0 + k[227]*y[IDX_MgI] + k[352]*y[IDX_SiI];
    data[4443] = 0.0 + k[431] + k[1628]*y[IDX_CH2I] + k[1709]*y[IDX_CNI] +
        k[1717]*y[IDX_COI] + k[1763]*y[IDX_HI] + k[1821]*y[IDX_NI] +
        k[1821]*y[IDX_NI] + k[1846]*y[IDX_NHI] + k[1905]*y[IDX_OI] + k[2056];
    data[4444] = 0.0 + k[1907]*y[IDX_OI];
    data[4445] = 0.0 + k[1851]*y[IDX_NHI] + k[1881]*y[IDX_CNI] +
        k[1897]*y[IDX_HNOI] + k[1901]*y[IDX_N2I] + k[1905]*y[IDX_NO2I] -
        k[1906]*y[IDX_NOI] + k[1907]*y[IDX_NSI] + k[1910]*y[IDX_OCNI];
    data[4446] = 0.0 + k[1311]*y[IDX_NII] + k[1712]*y[IDX_CNI] +
        k[1825]*y[IDX_NI] + k[1850]*y[IDX_NHI] - k[1859]*y[IDX_NOI] +
        k[1863]*y[IDX_OCNI];
    data[4447] = 0.0 - k[302]*y[IDX_NOI];
    data[4448] = 0.0 - k[1462]*y[IDX_NOI];
    data[4449] = 0.0 - k[1860]*y[IDX_NOI] + k[1863]*y[IDX_O2I] +
        k[1910]*y[IDX_OI];
    data[4450] = 0.0 + k[1312]*y[IDX_NII];
    data[4451] = 0.0 + k[1544]*y[IDX_HNOII] + k[1827]*y[IDX_NI] +
        k[1942]*y[IDX_HNOI] - k[1945]*y[IDX_NOI];
    data[4452] = 0.0 - k[336]*y[IDX_NOI] - k[1531]*y[IDX_NOI];
    data[4453] = 0.0 + k[1174]*y[IDX_HNOII] - k[1861]*y[IDX_NOI] -
        k[1862]*y[IDX_NOI];
    data[4454] = 0.0 - k[303]*y[IDX_NOI];
    data[4455] = 0.0 - k[304]*y[IDX_NOI];
    data[4456] = 0.0 + k[352]*y[IDX_NOII] - k[1958]*y[IDX_NOI];
    data[4457] = 0.0 - k[305]*y[IDX_NOI] + k[1344]*y[IDX_NI];
    data[4458] = 0.0 + k[1831]*y[IDX_NI];
    data[4459] = 0.0 + k[22]*y[IDX_NOI];
    data[4460] = 0.0 + k[34]*y[IDX_NOI];
    data[4461] = 0.0 + k[40]*y[IDX_NOI];
    data[4462] = 0.0 + k[45]*y[IDX_NOI];
    data[4463] = 0.0 + k[58]*y[IDX_NOI];
    data[4464] = 0.0 + k[61]*y[IDX_NOI];
    data[4465] = 0.0 + k[74]*y[IDX_NOI];
    data[4466] = 0.0 + k[1291]*y[IDX_NII];
    data[4467] = 0.0 + k[1469]*y[IDX_OII];
    data[4468] = 0.0 + k[97]*y[IDX_NOI] + k[868]*y[IDX_O2I];
    data[4469] = 0.0 + k[1297]*y[IDX_NII];
    data[4470] = 0.0 + k[104]*y[IDX_NOI];
    data[4471] = 0.0 + k[1352]*y[IDX_NHII];
    data[4472] = 0.0 - k[575]*y[IDX_NOII];
    data[4473] = 0.0 + k[133]*y[IDX_NOI] + k[901]*y[IDX_HNOI] +
        k[903]*y[IDX_NO2I];
    data[4474] = 0.0 + k[169]*y[IDX_NOI];
    data[4475] = 0.0 + k[1299]*y[IDX_NII];
    data[4476] = 0.0 + k[298]*y[IDX_NOI];
    data[4477] = 0.0 + k[182]*y[IDX_NOI] + k[1334]*y[IDX_NI];
    data[4478] = 0.0 + k[299]*y[IDX_NOI];
    data[4479] = 0.0 + k[1059]*y[IDX_NO2I];
    data[4480] = 0.0 + k[1475]*y[IDX_OII];
    data[4481] = 0.0 + k[197]*y[IDX_NOI];
    data[4482] = 0.0 + k[1245]*y[IDX_HNOI];
    data[4483] = 0.0 + k[901]*y[IDX_HII] + k[1245]*y[IDX_HeII];
    data[4484] = 0.0 + k[300]*y[IDX_NOI];
    data[4485] = 0.0 + k[301]*y[IDX_NOI];
    data[4486] = 0.0 - k[227]*y[IDX_NOII];
    data[4487] = 0.0 + k[1334]*y[IDX_H2OII] + k[1339]*y[IDX_O2II] +
        k[1340]*y[IDX_OHII] + k[1343]*y[IDX_SiOII];
    data[4488] = 0.0 + k[248]*y[IDX_NOI] + k[1291]*y[IDX_CH3OHI] +
        k[1297]*y[IDX_COI] + k[1299]*y[IDX_H2COI] + k[1310]*y[IDX_O2I];
    data[4489] = 0.0 + k[1477]*y[IDX_OII];
    data[4490] = 0.0 + k[255]*y[IDX_NOI] + k[1505]*y[IDX_OI];
    data[4491] = 0.0 + k[1457]*y[IDX_OII];
    data[4492] = 0.0 + k[263]*y[IDX_NOI] + k[1352]*y[IDX_CO2I] +
        k[1368]*y[IDX_O2I];
    data[4493] = 0.0 + k[269]*y[IDX_NOI];
    data[4494] = 0.0 + k[280]*y[IDX_NOI];
    data[4495] = 0.0 + k[22]*y[IDX_CII] + k[34]*y[IDX_C2II] +
        k[40]*y[IDX_C2HII] + k[45]*y[IDX_C2H2II] + k[58]*y[IDX_CHII] +
        k[61]*y[IDX_CH2II] + k[74]*y[IDX_CH3II] + k[97]*y[IDX_CNII] +
        k[104]*y[IDX_COII] + k[133]*y[IDX_HII] + k[169]*y[IDX_H2II] +
        k[182]*y[IDX_H2OII] + k[197]*y[IDX_HCNII] + k[248]*y[IDX_NII] +
        k[255]*y[IDX_N2II] + k[263]*y[IDX_NHII] + k[269]*y[IDX_NH2II] +
        k[280]*y[IDX_NH3II] + k[298]*y[IDX_H2COII] + k[299]*y[IDX_H2SII] +
        k[300]*y[IDX_HNOII] + k[301]*y[IDX_HSII] + k[302]*y[IDX_O2II] +
        k[303]*y[IDX_SII] + k[304]*y[IDX_S2II] + k[305]*y[IDX_SiOII] +
        k[336]*y[IDX_OHII] + k[432] + k[2057];
    data[4496] = 0.0 - k[227]*y[IDX_MgI] - k[352]*y[IDX_SiI] -
        k[575]*y[IDX_EM] - k[2236];
    data[4497] = 0.0 + k[903]*y[IDX_HII] + k[1059]*y[IDX_H3II] +
        k[1478]*y[IDX_OII];
    data[4498] = 0.0 + k[1509]*y[IDX_OI];
    data[4499] = 0.0 + k[1505]*y[IDX_N2II] + k[1509]*y[IDX_NSII];
    data[4500] = 0.0 + k[1457]*y[IDX_NHI] + k[1469]*y[IDX_CNI] +
        k[1475]*y[IDX_HCNI] + k[1477]*y[IDX_N2I] + k[1478]*y[IDX_NO2I];
    data[4501] = 0.0 + k[868]*y[IDX_CNII] + k[1310]*y[IDX_NII] +
        k[1368]*y[IDX_NHII];
    data[4502] = 0.0 + k[302]*y[IDX_NOI] + k[1339]*y[IDX_NI];
    data[4503] = 0.0 + k[336]*y[IDX_NOI] + k[1340]*y[IDX_NI];
    data[4504] = 0.0 + k[303]*y[IDX_NOI];
    data[4505] = 0.0 + k[304]*y[IDX_NOI];
    data[4506] = 0.0 - k[352]*y[IDX_NOII];
    data[4507] = 0.0 + k[305]*y[IDX_NOI] + k[1343]*y[IDX_NI];
    data[4508] = 0.0 + k[2447] + k[2448] + k[2449] + k[2450];
    data[4509] = 0.0 - k[1628]*y[IDX_NO2I];
    data[4510] = 0.0 - k[1658]*y[IDX_NO2I];
    data[4511] = 0.0 - k[1709]*y[IDX_NO2I];
    data[4512] = 0.0 - k[1717]*y[IDX_NO2I];
    data[4513] = 0.0 - k[1763]*y[IDX_NO2I];
    data[4514] = 0.0 - k[903]*y[IDX_NO2I];
    data[4515] = 0.0 - k[1059]*y[IDX_NO2I];
    data[4516] = 0.0 + k[1896]*y[IDX_OI];
    data[4517] = 0.0 - k[1820]*y[IDX_NO2I] - k[1821]*y[IDX_NO2I] -
        k[1822]*y[IDX_NO2I];
    data[4518] = 0.0 - k[1846]*y[IDX_NO2I];
    data[4519] = 0.0 + k[1859]*y[IDX_O2I] + k[1945]*y[IDX_OHI];
    data[4520] = 0.0 - k[431] - k[903]*y[IDX_HII] - k[1059]*y[IDX_H3II] -
        k[1478]*y[IDX_OII] - k[1628]*y[IDX_CH2I] - k[1658]*y[IDX_CH3I] -
        k[1709]*y[IDX_CNI] - k[1717]*y[IDX_COI] - k[1763]*y[IDX_HI] -
        k[1820]*y[IDX_NI] - k[1821]*y[IDX_NI] - k[1822]*y[IDX_NI] -
        k[1846]*y[IDX_NHI] - k[1905]*y[IDX_OI] - k[2056] - k[2270];
    data[4521] = 0.0 + k[1896]*y[IDX_HNOI] - k[1905]*y[IDX_NO2I];
    data[4522] = 0.0 - k[1478]*y[IDX_NO2I];
    data[4523] = 0.0 + k[1859]*y[IDX_NOI] + k[1864]*y[IDX_OCNI];
    data[4524] = 0.0 + k[1864]*y[IDX_O2I];
    data[4525] = 0.0 + k[1945]*y[IDX_NOI];
    data[4526] = 0.0 + k[2451] + k[2452] + k[2453] + k[2454];
    data[4527] = 0.0 - k[1606]*y[IDX_NSI] - k[1607]*y[IDX_NSI];
    data[4528] = 0.0 - k[23]*y[IDX_NSI] - k[632]*y[IDX_NSI];
    data[4529] = 0.0 + k[1714]*y[IDX_SI];
    data[4530] = 0.0 + k[550]*y[IDX_HNSII];
    data[4531] = 0.0 - k[1766]*y[IDX_NSI] - k[1767]*y[IDX_NSI];
    data[4532] = 0.0 - k[134]*y[IDX_NSI];
    data[4533] = 0.0 - k[1061]*y[IDX_NSI];
    data[4534] = 0.0 - k[1148]*y[IDX_NSI];
    data[4535] = 0.0 - k[1259]*y[IDX_NSI] - k[1260]*y[IDX_NSI];
    data[4536] = 0.0 + k[550]*y[IDX_EM];
    data[4537] = 0.0 + k[1816]*y[IDX_NI];
    data[4538] = 0.0 + k[1816]*y[IDX_HSI] - k[1824]*y[IDX_NSI] +
        k[1829]*y[IDX_S2I] + k[1830]*y[IDX_SOI];
    data[4539] = 0.0 + k[1857]*y[IDX_SI];
    data[4540] = 0.0 + k[1861]*y[IDX_SI];
    data[4541] = 0.0 - k[23]*y[IDX_CII] - k[134]*y[IDX_HII] - k[434] -
        k[632]*y[IDX_CII] - k[1061]*y[IDX_H3II] - k[1148]*y[IDX_HCOII] -
        k[1259]*y[IDX_HeII] - k[1260]*y[IDX_HeII] - k[1606]*y[IDX_CI] -
        k[1607]*y[IDX_CI] - k[1766]*y[IDX_HI] - k[1767]*y[IDX_HI] -
        k[1824]*y[IDX_NI] - k[1907]*y[IDX_OI] - k[1908]*y[IDX_OI] - k[2059] -
        k[2262];
    data[4542] = 0.0 - k[1907]*y[IDX_NSI] - k[1908]*y[IDX_NSI];
    data[4543] = 0.0 + k[1714]*y[IDX_CNI] + k[1857]*y[IDX_NHI] +
        k[1861]*y[IDX_NOI];
    data[4544] = 0.0 + k[1829]*y[IDX_NI];
    data[4545] = 0.0 + k[1830]*y[IDX_NI];
    data[4546] = 0.0 + k[23]*y[IDX_NSI];
    data[4547] = 0.0 - k[576]*y[IDX_NSII];
    data[4548] = 0.0 + k[134]*y[IDX_NSI];
    data[4549] = 0.0 + k[1335]*y[IDX_NI];
    data[4550] = 0.0 + k[1336]*y[IDX_NI];
    data[4551] = 0.0 + k[1335]*y[IDX_H2SII] + k[1336]*y[IDX_HSII] +
        k[1341]*y[IDX_SOII];
    data[4552] = 0.0 + k[1461]*y[IDX_SII];
    data[4553] = 0.0 + k[1373]*y[IDX_SI];
    data[4554] = 0.0 + k[23]*y[IDX_CII] + k[134]*y[IDX_HII];
    data[4555] = 0.0 - k[576]*y[IDX_EM] - k[1509]*y[IDX_OI] - k[2276];
    data[4556] = 0.0 - k[1509]*y[IDX_NSII];
    data[4557] = 0.0 + k[1373]*y[IDX_NHII];
    data[4558] = 0.0 + k[1461]*y[IDX_NHI];
    data[4559] = 0.0 + k[1341]*y[IDX_NI];
    data[4560] = 0.0 + k[700]*y[IDX_O2II] + k[702]*y[IDX_OHII] +
        k[1591]*y[IDX_COI] + k[1604]*y[IDX_NOI] + k[1608]*y[IDX_O2I] +
        k[1612]*y[IDX_OHI] + k[1615]*y[IDX_SOI] - k[2104]*y[IDX_OI];
    data[4561] = 0.0 + k[633]*y[IDX_O2I] + k[639]*y[IDX_SOI] -
        k[2098]*y[IDX_OI];
    data[4562] = 0.0 + k[306]*y[IDX_OII] + k[1517]*y[IDX_OHII] -
        k[1867]*y[IDX_OI];
    data[4563] = 0.0 - k[1490]*y[IDX_OI];
    data[4564] = 0.0 + k[308]*y[IDX_OII] + k[1518]*y[IDX_OHII] -
        k[1875]*y[IDX_OI];
    data[4565] = 0.0 - k[1491]*y[IDX_OI];
    data[4566] = 0.0 + k[307]*y[IDX_OII] - k[1868]*y[IDX_OI];
    data[4567] = 0.0 - k[1492]*y[IDX_OI];
    data[4568] = 0.0 - k[1869]*y[IDX_OI];
    data[4569] = 0.0 - k[1870]*y[IDX_OI] - k[1871]*y[IDX_OI] -
        k[1872]*y[IDX_OI] - k[1873]*y[IDX_OI];
    data[4570] = 0.0 - k[1874]*y[IDX_OI];
    data[4571] = 0.0 - k[1876]*y[IDX_OI];
    data[4572] = 0.0 - k[1877]*y[IDX_OI];
    data[4573] = 0.0 - k[1878]*y[IDX_OI];
    data[4574] = 0.0 - k[0]*y[IDX_OI] + k[90]*y[IDX_OII] +
        k[858]*y[IDX_O2II] + k[860]*y[IDX_OHII] + k[1684]*y[IDX_NOI] +
        k[1688]*y[IDX_O2I] + k[1690]*y[IDX_O2I] - k[1693]*y[IDX_OI] -
        k[1694]*y[IDX_OI];
    data[4575] = 0.0 + k[733]*y[IDX_O2I] - k[735]*y[IDX_OI];
    data[4576] = 0.0 + k[69]*y[IDX_OII] + k[769]*y[IDX_O2II] +
        k[771]*y[IDX_OHII] + k[1635]*y[IDX_O2I] - k[1637]*y[IDX_OI] -
        k[1638]*y[IDX_OI] - k[1639]*y[IDX_OI] - k[1640]*y[IDX_OI] +
        k[1643]*y[IDX_OHI];
    data[4577] = 0.0 - k[750]*y[IDX_OI];
    data[4578] = 0.0 - k[1664]*y[IDX_OI] - k[1665]*y[IDX_OI] +
        k[1666]*y[IDX_OHI];
    data[4579] = 0.0 + k[782]*y[IDX_O2I] - k[783]*y[IDX_OI] -
        k[784]*y[IDX_OI];
    data[4580] = 0.0 + k[309]*y[IDX_OII] + k[819]*y[IDX_OHII] -
        k[1879]*y[IDX_OI];
    data[4581] = 0.0 - k[1493]*y[IDX_OI];
    data[4582] = 0.0 - k[1494]*y[IDX_OI] - k[1495]*y[IDX_OI];
    data[4583] = 0.0 + k[1519]*y[IDX_OHII] + k[1713]*y[IDX_O2I] -
        k[1880]*y[IDX_OI] - k[1881]*y[IDX_OI] + k[1932]*y[IDX_OHI];
    data[4584] = 0.0 - k[326]*y[IDX_OI];
    data[4585] = 0.0 + k[310]*y[IDX_OII] + k[394] + k[1213]*y[IDX_HeII] +
        k[1521]*y[IDX_OHII] + k[1591]*y[IDX_CI] + k[1718]*y[IDX_O2I] + k[2004];
    data[4586] = 0.0 - k[327]*y[IDX_OI] + k[499]*y[IDX_EM] +
        k[1539]*y[IDX_OHI] + k[2002];
    data[4587] = 0.0 + k[393] + k[891]*y[IDX_HII] + k[1209]*y[IDX_HeII] +
        k[1520]*y[IDX_OHII] - k[1882]*y[IDX_OI] + k[2003];
    data[4588] = 0.0 - k[1883]*y[IDX_OI] - k[1884]*y[IDX_OI];
    data[4589] = 0.0 + k[1485]*y[IDX_O2I] - k[1496]*y[IDX_OI];
    data[4590] = 0.0 + k[499]*y[IDX_COII] + k[502]*y[IDX_H2COII] +
        k[512]*y[IDX_H2OII] + k[513]*y[IDX_H2OII] + k[529]*y[IDX_H3OII] +
        k[544]*y[IDX_HCO2II] + k[559]*y[IDX_HSO2II] + k[575]*y[IDX_NOII] +
        k[577]*y[IDX_O2II] + k[577]*y[IDX_O2II] + k[580]*y[IDX_OCSII] +
        k[582]*y[IDX_OHII] + k[584]*y[IDX_SOII] + k[585]*y[IDX_SO2II] +
        k[585]*y[IDX_SO2II] + k[586]*y[IDX_SO2II] + k[602]*y[IDX_SiOII] +
        k[2142]*y[IDX_OII];
    data[4591] = 0.0 + k[12]*y[IDX_O2I] + k[12]*y[IDX_O2I] +
        k[13]*y[IDX_OHI] + k[196]*y[IDX_OII] + k[1752]*y[IDX_HCOI] +
        k[1755]*y[IDX_HNOI] + k[1764]*y[IDX_NOI] + k[1768]*y[IDX_O2I] +
        k[1769]*y[IDX_O2HI] + k[1772]*y[IDX_OCNI] + k[1776]*y[IDX_OHI] +
        k[1778]*y[IDX_SOI] - k[2124]*y[IDX_OI];
    data[4592] = 0.0 - k[136]*y[IDX_OI] + k[891]*y[IDX_CO2I];
    data[4593] = 0.0 + k[6]*y[IDX_O2I] + k[6]*y[IDX_O2I] + k[7]*y[IDX_OHI] -
        k[1733]*y[IDX_OI];
    data[4594] = 0.0 - k[932]*y[IDX_OI];
    data[4595] = 0.0 - k[1885]*y[IDX_OI];
    data[4596] = 0.0 + k[311]*y[IDX_OII] + k[1218]*y[IDX_HeII] +
        k[1522]*y[IDX_OHII] - k[1886]*y[IDX_OI];
    data[4597] = 0.0 + k[502]*y[IDX_EM];
    data[4598] = 0.0 + k[312]*y[IDX_OII] + k[1358]*y[IDX_NHII] +
        k[1380]*y[IDX_NH2II] + k[1523]*y[IDX_OHII] - k[1887]*y[IDX_OI];
    data[4599] = 0.0 + k[512]*y[IDX_EM] + k[513]*y[IDX_EM] -
        k[1497]*y[IDX_OI] + k[1540]*y[IDX_OHI];
    data[4600] = 0.0 + k[313]*y[IDX_OII] + k[1524]*y[IDX_OHII] -
        k[1888]*y[IDX_OI];
    data[4601] = 0.0 - k[1498]*y[IDX_OI] - k[1499]*y[IDX_OI];
    data[4602] = 0.0 - k[1063]*y[IDX_OI] - k[1064]*y[IDX_OI];
    data[4603] = 0.0 + k[529]*y[IDX_EM];
    data[4604] = 0.0 + k[1525]*y[IDX_OHII] - k[1889]*y[IDX_OI] -
        k[1890]*y[IDX_OI] - k[1891]*y[IDX_OI];
    data[4605] = 0.0 + k[314]*y[IDX_OII] + k[1237]*y[IDX_HeII] +
        k[1527]*y[IDX_OHII] + k[1752]*y[IDX_HI] + k[1812]*y[IDX_NI] -
        k[1892]*y[IDX_OI] - k[1893]*y[IDX_OI];
    data[4606] = 0.0 + k[544]*y[IDX_EM] - k[1500]*y[IDX_OI];
    data[4607] = 0.0 - k[1894]*y[IDX_OI] - k[1895]*y[IDX_OI];
    data[4608] = 0.0 - k[1501]*y[IDX_OI] - k[1502]*y[IDX_OI];
    data[4609] = 0.0 + k[1209]*y[IDX_CO2I] + k[1213]*y[IDX_COI] +
        k[1218]*y[IDX_H2COI] + k[1237]*y[IDX_HCOI] + k[1258]*y[IDX_NOI] +
        k[1261]*y[IDX_O2I] + k[1262]*y[IDX_OCNI] + k[1264]*y[IDX_OCSI] +
        k[1271]*y[IDX_SO2I] + k[1272]*y[IDX_SOI] + k[1285]*y[IDX_SiOI];
    data[4610] = 0.0 + k[1528]*y[IDX_OHII];
    data[4611] = 0.0 + k[1755]*y[IDX_HI] - k[1896]*y[IDX_OI] -
        k[1897]*y[IDX_OI] - k[1898]*y[IDX_OI];
    data[4612] = 0.0 - k[1899]*y[IDX_OI] - k[1900]*y[IDX_OI];
    data[4613] = 0.0 - k[1503]*y[IDX_OI] - k[1504]*y[IDX_OI];
    data[4614] = 0.0 + k[559]*y[IDX_EM];
    data[4615] = 0.0 + k[1339]*y[IDX_O2II] + k[1341]*y[IDX_SOII] +
        k[1812]*y[IDX_HCOI] + k[1820]*y[IDX_NO2I] + k[1820]*y[IDX_NO2I] +
        k[1823]*y[IDX_NOI] + k[1825]*y[IDX_O2I] + k[1828]*y[IDX_OHI] +
        k[1830]*y[IDX_SOI];
    data[4616] = 0.0 + k[1309]*y[IDX_NOI] + k[1310]*y[IDX_O2I];
    data[4617] = 0.0 + k[1529]*y[IDX_OHII] - k[1901]*y[IDX_OI];
    data[4618] = 0.0 - k[328]*y[IDX_OI] - k[1505]*y[IDX_OI];
    data[4619] = 0.0 - k[1506]*y[IDX_OI];
    data[4620] = 0.0 + k[297]*y[IDX_OII] + k[1458]*y[IDX_O2II] +
        k[1460]*y[IDX_OHII] + k[1847]*y[IDX_NOI] + k[1849]*y[IDX_O2I] -
        k[1851]*y[IDX_OI] - k[1852]*y[IDX_OI] + k[1855]*y[IDX_OHI];
    data[4621] = 0.0 + k[1358]*y[IDX_H2OI] + k[1367]*y[IDX_NOI] -
        k[1370]*y[IDX_OI];
    data[4622] = 0.0 + k[315]*y[IDX_OII] + k[1411]*y[IDX_OHII] +
        k[1837]*y[IDX_OHI] - k[1902]*y[IDX_OI] - k[1903]*y[IDX_OI];
    data[4623] = 0.0 + k[1380]*y[IDX_H2OI] + k[1390]*y[IDX_O2I] -
        k[1507]*y[IDX_OI];
    data[4624] = 0.0 + k[316]*y[IDX_OII] + k[1530]*y[IDX_OHII] -
        k[1904]*y[IDX_OI];
    data[4625] = 0.0 - k[1508]*y[IDX_OI] + k[1546]*y[IDX_OHI];
    data[4626] = 0.0 + k[433] + k[1258]*y[IDX_HeII] + k[1309]*y[IDX_NII] +
        k[1367]*y[IDX_NHII] + k[1531]*y[IDX_OHII] + k[1604]*y[IDX_CI] +
        k[1684]*y[IDX_CHI] + k[1764]*y[IDX_HI] + k[1823]*y[IDX_NI] +
        k[1847]*y[IDX_NHI] + k[1859]*y[IDX_O2I] + k[1861]*y[IDX_SI] -
        k[1906]*y[IDX_OI] + k[2058];
    data[4627] = 0.0 + k[575]*y[IDX_EM];
    data[4628] = 0.0 + k[431] + k[1820]*y[IDX_NI] + k[1820]*y[IDX_NI] -
        k[1905]*y[IDX_OI] + k[2056];
    data[4629] = 0.0 - k[1907]*y[IDX_OI] - k[1908]*y[IDX_OI];
    data[4630] = 0.0 - k[1509]*y[IDX_OI];
    data[4631] = 0.0 - k[0]*y[IDX_CHI] - k[136]*y[IDX_HII] -
        k[326]*y[IDX_CNII] - k[327]*y[IDX_COII] - k[328]*y[IDX_N2II] - k[365] -
        k[438] - k[735]*y[IDX_CHII] - k[750]*y[IDX_CH2II] - k[783]*y[IDX_CH3II]
        - k[784]*y[IDX_CH3II] - k[932]*y[IDX_H2II] - k[1063]*y[IDX_H3II] -
        k[1064]*y[IDX_H3II] - k[1370]*y[IDX_NHII] - k[1490]*y[IDX_C2II] -
        k[1491]*y[IDX_C2HII] - k[1492]*y[IDX_C2H2II] - k[1493]*y[IDX_CH4II] -
        k[1494]*y[IDX_CH5II] - k[1495]*y[IDX_CH5II] - k[1496]*y[IDX_CSII] -
        k[1497]*y[IDX_H2OII] - k[1498]*y[IDX_H2SII] - k[1499]*y[IDX_H2SII] -
        k[1500]*y[IDX_HCO2II] - k[1501]*y[IDX_HCSII] - k[1502]*y[IDX_HCSII] -
        k[1503]*y[IDX_HSII] - k[1504]*y[IDX_HSII] - k[1505]*y[IDX_N2II] -
        k[1506]*y[IDX_N2HII] - k[1507]*y[IDX_NH2II] - k[1508]*y[IDX_NH3II] -
        k[1509]*y[IDX_NSII] - k[1510]*y[IDX_O2HII] - k[1511]*y[IDX_OHII] -
        k[1512]*y[IDX_SiCII] - k[1513]*y[IDX_SiHII] - k[1514]*y[IDX_SiH2II] -
        k[1515]*y[IDX_SiH3II] - k[1516]*y[IDX_SiOII] - k[1637]*y[IDX_CH2I] -
        k[1638]*y[IDX_CH2I] - k[1639]*y[IDX_CH2I] - k[1640]*y[IDX_CH2I] -
        k[1664]*y[IDX_CH3I] - k[1665]*y[IDX_CH3I] - k[1693]*y[IDX_CHI] -
        k[1694]*y[IDX_CHI] - k[1733]*y[IDX_H2I] - k[1851]*y[IDX_NHI] -
        k[1852]*y[IDX_NHI] - k[1867]*y[IDX_C2I] - k[1868]*y[IDX_C2H2I] -
        k[1869]*y[IDX_C2H3I] - k[1870]*y[IDX_C2H4I] - k[1871]*y[IDX_C2H4I] -
        k[1872]*y[IDX_C2H4I] - k[1873]*y[IDX_C2H4I] - k[1874]*y[IDX_C2H5I] -
        k[1875]*y[IDX_C2HI] - k[1876]*y[IDX_C2NI] - k[1877]*y[IDX_C3NI] -
        k[1878]*y[IDX_C4NI] - k[1879]*y[IDX_CH4I] - k[1880]*y[IDX_CNI] -
        k[1881]*y[IDX_CNI] - k[1882]*y[IDX_CO2I] - k[1883]*y[IDX_CSI] -
        k[1884]*y[IDX_CSI] - k[1885]*y[IDX_H2CNI] - k[1886]*y[IDX_H2COI] -
        k[1887]*y[IDX_H2OI] - k[1888]*y[IDX_H2SI] - k[1889]*y[IDX_HCNI] -
        k[1890]*y[IDX_HCNI] - k[1891]*y[IDX_HCNI] - k[1892]*y[IDX_HCOI] -
        k[1893]*y[IDX_HCOI] - k[1894]*y[IDX_HCSI] - k[1895]*y[IDX_HCSI] -
        k[1896]*y[IDX_HNOI] - k[1897]*y[IDX_HNOI] - k[1898]*y[IDX_HNOI] -
        k[1899]*y[IDX_HSI] - k[1900]*y[IDX_HSI] - k[1901]*y[IDX_N2I] -
        k[1902]*y[IDX_NH2I] - k[1903]*y[IDX_NH2I] - k[1904]*y[IDX_NH3I] -
        k[1905]*y[IDX_NO2I] - k[1906]*y[IDX_NOI] - k[1907]*y[IDX_NSI] -
        k[1908]*y[IDX_NSI] - k[1909]*y[IDX_O2HI] - k[1910]*y[IDX_OCNI] -
        k[1911]*y[IDX_OCNI] - k[1912]*y[IDX_OCSI] - k[1913]*y[IDX_OCSI] -
        k[1914]*y[IDX_OHI] - k[1915]*y[IDX_S2I] - k[1916]*y[IDX_SO2I] -
        k[1917]*y[IDX_SOI] - k[1918]*y[IDX_SiC2I] - k[1919]*y[IDX_SiC3I] -
        k[1920]*y[IDX_SiCI] - k[1921]*y[IDX_SiCI] - k[1922]*y[IDX_SiH2I] -
        k[1923]*y[IDX_SiH2I] - k[1924]*y[IDX_SiH3I] - k[1925]*y[IDX_SiH4I] -
        k[1926]*y[IDX_SiHI] - k[2098]*y[IDX_CII] - k[2104]*y[IDX_CI] -
        k[2124]*y[IDX_HI] - k[2128]*y[IDX_OI] - k[2128]*y[IDX_OI] -
        k[2128]*y[IDX_OI] - k[2128]*y[IDX_OI] - k[2129]*y[IDX_SOI] -
        k[2130]*y[IDX_SiII] - k[2131]*y[IDX_SiI] - k[2252];
    data[4632] = 0.0 + k[69]*y[IDX_CH2I] + k[90]*y[IDX_CHI] +
        k[196]*y[IDX_HI] + k[297]*y[IDX_NHI] + k[306]*y[IDX_C2I] +
        k[307]*y[IDX_C2H2I] + k[308]*y[IDX_C2HI] + k[309]*y[IDX_CH4I] +
        k[310]*y[IDX_COI] + k[311]*y[IDX_H2COI] + k[312]*y[IDX_H2OI] +
        k[313]*y[IDX_H2SI] + k[314]*y[IDX_HCOI] + k[315]*y[IDX_NH2I] +
        k[316]*y[IDX_NH3I] + k[317]*y[IDX_O2I] + k[318]*y[IDX_OCSI] +
        k[319]*y[IDX_OHI] + k[320]*y[IDX_SO2I] + k[2142]*y[IDX_EM];
    data[4633] = 0.0 + k[6]*y[IDX_H2I] + k[6]*y[IDX_H2I] + k[12]*y[IDX_HI] +
        k[12]*y[IDX_HI] + k[317]*y[IDX_OII] + k[436] + k[436] +
        k[633]*y[IDX_CII] + k[733]*y[IDX_CHII] + k[782]*y[IDX_CH3II] +
        k[1261]*y[IDX_HeII] + k[1310]*y[IDX_NII] + k[1390]*y[IDX_NH2II] +
        k[1485]*y[IDX_CSII] + k[1486]*y[IDX_SII] + k[1608]*y[IDX_CI] +
        k[1635]*y[IDX_CH2I] + k[1688]*y[IDX_CHI] + k[1690]*y[IDX_CHI] +
        k[1713]*y[IDX_CNI] + k[1718]*y[IDX_COI] + k[1768]*y[IDX_HI] +
        k[1825]*y[IDX_NI] + k[1849]*y[IDX_NHI] + k[1859]*y[IDX_NOI] +
        k[1865]*y[IDX_SI] + k[1866]*y[IDX_SOI] + k[1959]*y[IDX_SiI] + k[2062] +
        k[2062];
    data[4634] = 0.0 + k[577]*y[IDX_EM] + k[577]*y[IDX_EM] +
        k[700]*y[IDX_CI] + k[769]*y[IDX_CH2I] + k[858]*y[IDX_CHI] +
        k[1339]*y[IDX_NI] + k[1458]*y[IDX_NHI] + k[1484]*y[IDX_SI] + k[2060];
    data[4635] = 0.0 + k[1769]*y[IDX_HI] - k[1909]*y[IDX_OI] + k[2064];
    data[4636] = 0.0 - k[1510]*y[IDX_OI];
    data[4637] = 0.0 + k[439] + k[1262]*y[IDX_HeII] + k[1772]*y[IDX_HI] -
        k[1910]*y[IDX_OI] - k[1911]*y[IDX_OI] + k[2065];
    data[4638] = 0.0 + k[318]*y[IDX_OII] + k[1264]*y[IDX_HeII] -
        k[1912]*y[IDX_OI] - k[1913]*y[IDX_OI];
    data[4639] = 0.0 + k[580]*y[IDX_EM];
    data[4640] = 0.0 + k[7]*y[IDX_H2I] + k[13]*y[IDX_HI] + k[319]*y[IDX_OII]
        + k[442] + k[1532]*y[IDX_OHII] + k[1539]*y[IDX_COII] +
        k[1540]*y[IDX_H2OII] + k[1546]*y[IDX_NH3II] + k[1612]*y[IDX_CI] +
        k[1643]*y[IDX_CH2I] + k[1666]*y[IDX_CH3I] + k[1776]*y[IDX_HI] +
        k[1828]*y[IDX_NI] + k[1837]*y[IDX_NH2I] + k[1855]*y[IDX_NHI] -
        k[1914]*y[IDX_OI] + k[1932]*y[IDX_CNI] + k[1947]*y[IDX_OHI] +
        k[1947]*y[IDX_OHI] + k[2069];
    data[4641] = 0.0 + k[582]*y[IDX_EM] + k[702]*y[IDX_CI] +
        k[771]*y[IDX_CH2I] + k[819]*y[IDX_CH4I] + k[860]*y[IDX_CHI] +
        k[1411]*y[IDX_NH2I] + k[1460]*y[IDX_NHI] - k[1511]*y[IDX_OI] +
        k[1517]*y[IDX_C2I] + k[1518]*y[IDX_C2HI] + k[1519]*y[IDX_CNI] +
        k[1520]*y[IDX_CO2I] + k[1521]*y[IDX_COI] + k[1522]*y[IDX_H2COI] +
        k[1523]*y[IDX_H2OI] + k[1524]*y[IDX_H2SI] + k[1525]*y[IDX_HCNI] +
        k[1527]*y[IDX_HCOI] + k[1528]*y[IDX_HNCI] + k[1529]*y[IDX_N2I] +
        k[1530]*y[IDX_NH3I] + k[1531]*y[IDX_NOI] + k[1532]*y[IDX_OHI] +
        k[1533]*y[IDX_SI] + k[1535]*y[IDX_SiI] + k[1536]*y[IDX_SiHI] +
        k[1537]*y[IDX_SiOI];
    data[4642] = 0.0 + k[1484]*y[IDX_O2II] + k[1533]*y[IDX_OHII] +
        k[1861]*y[IDX_NOI] + k[1865]*y[IDX_O2I] + k[1955]*y[IDX_SOI];
    data[4643] = 0.0 + k[1486]*y[IDX_O2I];
    data[4644] = 0.0 - k[1915]*y[IDX_OI];
    data[4645] = 0.0 + k[1535]*y[IDX_OHII] + k[1959]*y[IDX_O2I] -
        k[2131]*y[IDX_OI];
    data[4646] = 0.0 - k[2130]*y[IDX_OI];
    data[4647] = 0.0 - k[1920]*y[IDX_OI] - k[1921]*y[IDX_OI];
    data[4648] = 0.0 - k[1512]*y[IDX_OI];
    data[4649] = 0.0 - k[1918]*y[IDX_OI];
    data[4650] = 0.0 - k[1919]*y[IDX_OI];
    data[4651] = 0.0 + k[1536]*y[IDX_OHII] - k[1926]*y[IDX_OI];
    data[4652] = 0.0 - k[1513]*y[IDX_OI];
    data[4653] = 0.0 - k[1922]*y[IDX_OI] - k[1923]*y[IDX_OI];
    data[4654] = 0.0 - k[1514]*y[IDX_OI];
    data[4655] = 0.0 - k[1924]*y[IDX_OI];
    data[4656] = 0.0 - k[1515]*y[IDX_OI];
    data[4657] = 0.0 - k[1925]*y[IDX_OI];
    data[4658] = 0.0 + k[456] + k[1285]*y[IDX_HeII] + k[1537]*y[IDX_OHII] +
        k[2093];
    data[4659] = 0.0 + k[602]*y[IDX_EM] - k[1516]*y[IDX_OI] + k[2092];
    data[4660] = 0.0 + k[446] + k[639]*y[IDX_CII] + k[1272]*y[IDX_HeII] +
        k[1615]*y[IDX_CI] + k[1778]*y[IDX_HI] + k[1830]*y[IDX_NI] +
        k[1866]*y[IDX_O2I] - k[1917]*y[IDX_OI] + k[1955]*y[IDX_SI] + k[2075] -
        k[2129]*y[IDX_OI];
    data[4661] = 0.0 + k[584]*y[IDX_EM] + k[1341]*y[IDX_NI];
    data[4662] = 0.0 + k[320]*y[IDX_OII] + k[445] + k[1271]*y[IDX_HeII] -
        k[1916]*y[IDX_OI] + k[2074];
    data[4663] = 0.0 + k[585]*y[IDX_EM] + k[585]*y[IDX_EM] +
        k[586]*y[IDX_EM];
    data[4664] = 0.0 - k[2103]*y[IDX_OII];
    data[4665] = 0.0 + k[634]*y[IDX_O2I];
    data[4666] = 0.0 - k[306]*y[IDX_OII] - k[1463]*y[IDX_OII];
    data[4667] = 0.0 - k[308]*y[IDX_OII] - k[1465]*y[IDX_OII];
    data[4668] = 0.0 - k[307]*y[IDX_OII];
    data[4669] = 0.0 - k[1464]*y[IDX_OII];
    data[4670] = 0.0 - k[90]*y[IDX_OII] - k[857]*y[IDX_OII];
    data[4671] = 0.0 + k[734]*y[IDX_O2I];
    data[4672] = 0.0 - k[69]*y[IDX_OII];
    data[4673] = 0.0 - k[1466]*y[IDX_OII] - k[1467]*y[IDX_OII];
    data[4674] = 0.0 - k[309]*y[IDX_OII] - k[1468]*y[IDX_OII];
    data[4675] = 0.0 - k[1469]*y[IDX_OII];
    data[4676] = 0.0 + k[326]*y[IDX_OI];
    data[4677] = 0.0 - k[310]*y[IDX_OII];
    data[4678] = 0.0 + k[327]*y[IDX_OI];
    data[4679] = 0.0 + k[1210]*y[IDX_HeII] - k[1470]*y[IDX_OII];
    data[4680] = 0.0 - k[2142]*y[IDX_OII];
    data[4681] = 0.0 - k[196]*y[IDX_OII];
    data[4682] = 0.0 + k[136]*y[IDX_OI];
    data[4683] = 0.0 - k[958]*y[IDX_OII];
    data[4684] = 0.0 - k[311]*y[IDX_OII] - k[1471]*y[IDX_OII];
    data[4685] = 0.0 - k[312]*y[IDX_OII];
    data[4686] = 0.0 - k[313]*y[IDX_OII] - k[1472]*y[IDX_OII] -
        k[1473]*y[IDX_OII];
    data[4687] = 0.0 - k[1474]*y[IDX_OII] - k[1475]*y[IDX_OII];
    data[4688] = 0.0 - k[314]*y[IDX_OII] - k[1476]*y[IDX_OII];
    data[4689] = 0.0 + k[1210]*y[IDX_CO2I] + k[1257]*y[IDX_NOI] +
        k[1261]*y[IDX_O2I] + k[1263]*y[IDX_OCNI] + k[1265]*y[IDX_OCSI] +
        k[1268]*y[IDX_OHI] + k[1273]*y[IDX_SOI] + k[1286]*y[IDX_SiOI];
    data[4690] = 0.0 + k[1311]*y[IDX_O2I];
    data[4691] = 0.0 - k[1477]*y[IDX_OII];
    data[4692] = 0.0 + k[328]*y[IDX_OI];
    data[4693] = 0.0 - k[297]*y[IDX_OII] - k[1457]*y[IDX_OII];
    data[4694] = 0.0 - k[315]*y[IDX_OII];
    data[4695] = 0.0 - k[316]*y[IDX_OII];
    data[4696] = 0.0 + k[1257]*y[IDX_HeII];
    data[4697] = 0.0 - k[1478]*y[IDX_OII];
    data[4698] = 0.0 + k[136]*y[IDX_HII] + k[326]*y[IDX_CNII] +
        k[327]*y[IDX_COII] + k[328]*y[IDX_N2II] + k[365] + k[438];
    data[4699] = 0.0 - k[69]*y[IDX_CH2I] - k[90]*y[IDX_CHI] -
        k[196]*y[IDX_HI] - k[297]*y[IDX_NHI] - k[306]*y[IDX_C2I] -
        k[307]*y[IDX_C2H2I] - k[308]*y[IDX_C2HI] - k[309]*y[IDX_CH4I] -
        k[310]*y[IDX_COI] - k[311]*y[IDX_H2COI] - k[312]*y[IDX_H2OI] -
        k[313]*y[IDX_H2SI] - k[314]*y[IDX_HCOI] - k[315]*y[IDX_NH2I] -
        k[316]*y[IDX_NH3I] - k[317]*y[IDX_O2I] - k[318]*y[IDX_OCSI] -
        k[319]*y[IDX_OHI] - k[320]*y[IDX_SO2I] - k[857]*y[IDX_CHI] -
        k[958]*y[IDX_H2I] - k[1457]*y[IDX_NHI] - k[1463]*y[IDX_C2I] -
        k[1464]*y[IDX_C2H4I] - k[1465]*y[IDX_C2HI] - k[1466]*y[IDX_CH3OHI] -
        k[1467]*y[IDX_CH3OHI] - k[1468]*y[IDX_CH4I] - k[1469]*y[IDX_CNI] -
        k[1470]*y[IDX_CO2I] - k[1471]*y[IDX_H2COI] - k[1472]*y[IDX_H2SI] -
        k[1473]*y[IDX_H2SI] - k[1474]*y[IDX_HCNI] - k[1475]*y[IDX_HCNI] -
        k[1476]*y[IDX_HCOI] - k[1477]*y[IDX_N2I] - k[1478]*y[IDX_NO2I] -
        k[1479]*y[IDX_OCSI] - k[1480]*y[IDX_OHI] - k[1481]*y[IDX_SO2I] -
        k[2103]*y[IDX_CI] - k[2142]*y[IDX_EM] - k[2227];
    data[4700] = 0.0 - k[317]*y[IDX_OII] + k[634]*y[IDX_CII] +
        k[734]*y[IDX_CHII] + k[1261]*y[IDX_HeII] + k[1311]*y[IDX_NII];
    data[4701] = 0.0 + k[2060];
    data[4702] = 0.0 + k[1263]*y[IDX_HeII];
    data[4703] = 0.0 - k[318]*y[IDX_OII] + k[1265]*y[IDX_HeII] -
        k[1479]*y[IDX_OII];
    data[4704] = 0.0 - k[319]*y[IDX_OII] + k[1268]*y[IDX_HeII] -
        k[1480]*y[IDX_OII];
    data[4705] = 0.0 + k[2068];
    data[4706] = 0.0 + k[1286]*y[IDX_HeII];
    data[4707] = 0.0 + k[1273]*y[IDX_HeII];
    data[4708] = 0.0 - k[320]*y[IDX_OII] - k[1481]*y[IDX_OII];
    data[4709] = 0.0 + k[2375] + k[2376] + k[2377] + k[2378];
    data[4710] = 0.0 + k[54]*y[IDX_O2II] + k[701]*y[IDX_O2HII] -
        k[1608]*y[IDX_O2I];
    data[4711] = 0.0 - k[633]*y[IDX_O2I] - k[634]*y[IDX_O2I];
    data[4712] = 0.0 + k[39]*y[IDX_O2II] + k[657]*y[IDX_O2HII] -
        k[1572]*y[IDX_O2I];
    data[4713] = 0.0 - k[649]*y[IDX_O2I];
    data[4714] = 0.0 + k[683]*y[IDX_O2HII] - k[1581]*y[IDX_O2I];
    data[4715] = 0.0 + k[321]*y[IDX_O2II];
    data[4716] = 0.0 - k[1576]*y[IDX_O2I] - k[1577]*y[IDX_O2I];
    data[4717] = 0.0 + k[91]*y[IDX_O2II] + k[859]*y[IDX_O2HII] -
        k[1687]*y[IDX_O2I] - k[1688]*y[IDX_O2I] - k[1689]*y[IDX_O2I] -
        k[1690]*y[IDX_O2I] + k[1692]*y[IDX_O2HI];
    data[4718] = 0.0 - k[732]*y[IDX_O2I] - k[733]*y[IDX_O2I] -
        k[734]*y[IDX_O2I];
    data[4719] = 0.0 + k[70]*y[IDX_O2II] + k[770]*y[IDX_O2HII] -
        k[1632]*y[IDX_O2I] - k[1633]*y[IDX_O2I] - k[1634]*y[IDX_O2I] -
        k[1635]*y[IDX_O2I] - k[1636]*y[IDX_O2I];
    data[4720] = 0.0 - k[749]*y[IDX_O2I];
    data[4721] = 0.0 - k[1660]*y[IDX_O2I] - k[1661]*y[IDX_O2I] -
        k[1662]*y[IDX_O2I] + k[1663]*y[IDX_O2HI];
    data[4722] = 0.0 - k[782]*y[IDX_O2I];
    data[4723] = 0.0 + k[1483]*y[IDX_O2II];
    data[4724] = 0.0 - k[1671]*y[IDX_O2I];
    data[4725] = 0.0 - k[79]*y[IDX_O2I];
    data[4726] = 0.0 - k[324]*y[IDX_O2I];
    data[4727] = 0.0 + k[870]*y[IDX_O2HII] - k[1712]*y[IDX_O2I] -
        k[1713]*y[IDX_O2I];
    data[4728] = 0.0 - k[98]*y[IDX_O2I] - k[868]*y[IDX_O2I];
    data[4729] = 0.0 + k[878]*y[IDX_O2HII] - k[1718]*y[IDX_O2I];
    data[4730] = 0.0 - k[105]*y[IDX_O2I];
    data[4731] = 0.0 + k[1212]*y[IDX_HeII] + k[1489]*y[IDX_O2HII] +
        k[1882]*y[IDX_OI];
    data[4732] = 0.0 - k[1485]*y[IDX_O2I];
    data[4733] = 0.0 + k[578]*y[IDX_O2HII];
    data[4734] = 0.0 - k[12]*y[IDX_O2I] - k[1768]*y[IDX_O2I] +
        k[1770]*y[IDX_O2HI];
    data[4735] = 0.0 - k[135]*y[IDX_O2I];
    data[4736] = 0.0 - k[6]*y[IDX_O2I] + k[959]*y[IDX_O2HII] -
        k[1731]*y[IDX_O2I] - k[1732]*y[IDX_O2I];
    data[4737] = 0.0 - k[170]*y[IDX_O2I] - k[931]*y[IDX_O2I];
    data[4738] = 0.0 + k[174]*y[IDX_O2II] + k[972]*y[IDX_O2II] +
        k[973]*y[IDX_O2HII];
    data[4739] = 0.0 - k[967]*y[IDX_O2I];
    data[4740] = 0.0 + k[1012]*y[IDX_O2HII];
    data[4741] = 0.0 - k[183]*y[IDX_O2I];
    data[4742] = 0.0 + k[322]*y[IDX_O2II];
    data[4743] = 0.0 - k[1062]*y[IDX_O2I];
    data[4744] = 0.0 + k[1129]*y[IDX_O2HII];
    data[4745] = 0.0 - k[198]*y[IDX_O2I];
    data[4746] = 0.0 + k[204]*y[IDX_O2II] + k[1162]*y[IDX_O2HII] -
        k[1784]*y[IDX_O2I] - k[1785]*y[IDX_O2I] + k[1786]*y[IDX_O2HI];
    data[4747] = 0.0 + k[1500]*y[IDX_OI];
    data[4748] = 0.0 - k[217]*y[IDX_O2I] + k[1212]*y[IDX_CO2I] -
        k[1261]*y[IDX_O2I] + k[1270]*y[IDX_SO2I];
    data[4749] = 0.0 + k[1172]*y[IDX_O2HII];
    data[4750] = 0.0 + k[1898]*y[IDX_OI];
    data[4751] = 0.0 + k[228]*y[IDX_O2II];
    data[4752] = 0.0 + k[1822]*y[IDX_NO2I] - k[1825]*y[IDX_O2I] +
        k[1826]*y[IDX_O2HI];
    data[4753] = 0.0 - k[249]*y[IDX_O2I] - k[1310]*y[IDX_O2I] -
        k[1311]*y[IDX_O2I];
    data[4754] = 0.0 + k[1320]*y[IDX_O2HII];
    data[4755] = 0.0 - k[256]*y[IDX_O2I];
    data[4756] = 0.0 + k[1459]*y[IDX_O2HII] - k[1849]*y[IDX_O2I] -
        k[1850]*y[IDX_O2I];
    data[4757] = 0.0 - k[264]*y[IDX_O2I] - k[1368]*y[IDX_O2I] -
        k[1369]*y[IDX_O2I];
    data[4758] = 0.0 + k[276]*y[IDX_O2II] + k[1410]*y[IDX_O2HII];
    data[4759] = 0.0 - k[1390]*y[IDX_O2I] - k[1391]*y[IDX_O2I];
    data[4760] = 0.0 + k[290]*y[IDX_O2II] + k[1441]*y[IDX_O2HII];
    data[4761] = 0.0 + k[302]*y[IDX_O2II] + k[1462]*y[IDX_O2HII] +
        k[1858]*y[IDX_NOI] + k[1858]*y[IDX_NOI] - k[1859]*y[IDX_O2I] +
        k[1906]*y[IDX_OI];
    data[4762] = 0.0 + k[1478]*y[IDX_OII] + k[1822]*y[IDX_NI] +
        k[1905]*y[IDX_OI];
    data[4763] = 0.0 + k[1500]*y[IDX_HCO2II] + k[1510]*y[IDX_O2HII] +
        k[1516]*y[IDX_SiOII] + k[1882]*y[IDX_CO2I] + k[1898]*y[IDX_HNOI] +
        k[1905]*y[IDX_NO2I] + k[1906]*y[IDX_NOI] + k[1909]*y[IDX_O2HI] +
        k[1911]*y[IDX_OCNI] + k[1914]*y[IDX_OHI] + k[1916]*y[IDX_SO2I] +
        k[1917]*y[IDX_SOI] + k[2128]*y[IDX_OI] + k[2128]*y[IDX_OI];
    data[4764] = 0.0 - k[317]*y[IDX_O2I] + k[1478]*y[IDX_NO2I] +
        k[1481]*y[IDX_SO2I];
    data[4765] = 0.0 - k[6]*y[IDX_H2I] - k[12]*y[IDX_HI] -
        k[79]*y[IDX_CH4II] - k[98]*y[IDX_CNII] - k[105]*y[IDX_COII] -
        k[135]*y[IDX_HII] - k[170]*y[IDX_H2II] - k[183]*y[IDX_H2OII] -
        k[198]*y[IDX_HCNII] - k[217]*y[IDX_HeII] - k[249]*y[IDX_NII] -
        k[256]*y[IDX_N2II] - k[264]*y[IDX_NHII] - k[317]*y[IDX_OII] -
        k[324]*y[IDX_ClII] - k[325]*y[IDX_SO2II] - k[337]*y[IDX_OHII] - k[435] -
        k[436] - k[633]*y[IDX_CII] - k[634]*y[IDX_CII] - k[649]*y[IDX_C2II] -
        k[732]*y[IDX_CHII] - k[733]*y[IDX_CHII] - k[734]*y[IDX_CHII] -
        k[749]*y[IDX_CH2II] - k[782]*y[IDX_CH3II] - k[868]*y[IDX_CNII] -
        k[931]*y[IDX_H2II] - k[967]*y[IDX_H2COII] - k[1062]*y[IDX_H3II] -
        k[1261]*y[IDX_HeII] - k[1310]*y[IDX_NII] - k[1311]*y[IDX_NII] -
        k[1368]*y[IDX_NHII] - k[1369]*y[IDX_NHII] - k[1390]*y[IDX_NH2II] -
        k[1391]*y[IDX_NH2II] - k[1485]*y[IDX_CSII] - k[1486]*y[IDX_SII] -
        k[1487]*y[IDX_SiSII] - k[1488]*y[IDX_SiSII] - k[1567]*y[IDX_SiH2II] -
        k[1572]*y[IDX_C2I] - k[1576]*y[IDX_C2H3I] - k[1577]*y[IDX_C2H3I] -
        k[1581]*y[IDX_C2HI] - k[1608]*y[IDX_CI] - k[1632]*y[IDX_CH2I] -
        k[1633]*y[IDX_CH2I] - k[1634]*y[IDX_CH2I] - k[1635]*y[IDX_CH2I] -
        k[1636]*y[IDX_CH2I] - k[1660]*y[IDX_CH3I] - k[1661]*y[IDX_CH3I] -
        k[1662]*y[IDX_CH3I] - k[1671]*y[IDX_CH4I] - k[1687]*y[IDX_CHI] -
        k[1688]*y[IDX_CHI] - k[1689]*y[IDX_CHI] - k[1690]*y[IDX_CHI] -
        k[1712]*y[IDX_CNI] - k[1713]*y[IDX_CNI] - k[1718]*y[IDX_COI] -
        k[1731]*y[IDX_H2I] - k[1732]*y[IDX_H2I] - k[1768]*y[IDX_HI] -
        k[1784]*y[IDX_HCOI] - k[1785]*y[IDX_HCOI] - k[1825]*y[IDX_NI] -
        k[1849]*y[IDX_NHI] - k[1850]*y[IDX_NHI] - k[1859]*y[IDX_NOI] -
        k[1863]*y[IDX_OCNI] - k[1864]*y[IDX_OCNI] - k[1865]*y[IDX_SI] -
        k[1866]*y[IDX_SOI] - k[1959]*y[IDX_SiI] - k[2061] - k[2062] - k[2253];
    data[4766] = 0.0 + k[39]*y[IDX_C2I] + k[54]*y[IDX_CI] +
        k[70]*y[IDX_CH2I] + k[91]*y[IDX_CHI] + k[174]*y[IDX_H2COI] +
        k[204]*y[IDX_HCOI] + k[228]*y[IDX_MgI] + k[276]*y[IDX_NH2I] +
        k[290]*y[IDX_NH3I] + k[302]*y[IDX_NOI] + k[321]*y[IDX_C2H2I] +
        k[322]*y[IDX_H2SI] + k[323]*y[IDX_SI] + k[353]*y[IDX_SiI] +
        k[972]*y[IDX_H2COI] + k[1483]*y[IDX_CH3OHI];
    data[4767] = 0.0 + k[437] + k[1663]*y[IDX_CH3I] + k[1692]*y[IDX_CHI] +
        k[1770]*y[IDX_HI] + k[1786]*y[IDX_HCOI] + k[1826]*y[IDX_NI] +
        k[1909]*y[IDX_OI] + k[1946]*y[IDX_OHI] + k[2063];
    data[4768] = 0.0 + k[578]*y[IDX_EM] + k[657]*y[IDX_C2I] +
        k[683]*y[IDX_C2HI] + k[701]*y[IDX_CI] + k[770]*y[IDX_CH2I] +
        k[859]*y[IDX_CHI] + k[870]*y[IDX_CNI] + k[878]*y[IDX_COI] +
        k[959]*y[IDX_H2I] + k[973]*y[IDX_H2COI] + k[1012]*y[IDX_H2OI] +
        k[1129]*y[IDX_HCNI] + k[1162]*y[IDX_HCOI] + k[1172]*y[IDX_HNCI] +
        k[1320]*y[IDX_N2I] + k[1410]*y[IDX_NH2I] + k[1441]*y[IDX_NH3I] +
        k[1459]*y[IDX_NHI] + k[1462]*y[IDX_NOI] + k[1489]*y[IDX_CO2I] +
        k[1510]*y[IDX_OI] + k[1547]*y[IDX_OHI] + k[1554]*y[IDX_SI];
    data[4769] = 0.0 - k[1863]*y[IDX_O2I] - k[1864]*y[IDX_O2I] +
        k[1911]*y[IDX_OI];
    data[4770] = 0.0 + k[1547]*y[IDX_O2HII] + k[1914]*y[IDX_OI] +
        k[1946]*y[IDX_O2HI];
    data[4771] = 0.0 - k[337]*y[IDX_O2I];
    data[4772] = 0.0 + k[323]*y[IDX_O2II] + k[1554]*y[IDX_O2HII] -
        k[1865]*y[IDX_O2I];
    data[4773] = 0.0 - k[1486]*y[IDX_O2I];
    data[4774] = 0.0 + k[353]*y[IDX_O2II] - k[1959]*y[IDX_O2I];
    data[4775] = 0.0 - k[1567]*y[IDX_O2I];
    data[4776] = 0.0 + k[1516]*y[IDX_OI];
    data[4777] = 0.0 - k[1487]*y[IDX_O2I] - k[1488]*y[IDX_O2I];
    data[4778] = 0.0 - k[1866]*y[IDX_O2I] + k[1917]*y[IDX_OI];
    data[4779] = 0.0 + k[1270]*y[IDX_HeII] + k[1481]*y[IDX_OII] +
        k[1916]*y[IDX_OI];
    data[4780] = 0.0 - k[325]*y[IDX_O2I];
    data[4781] = 0.0 - k[54]*y[IDX_O2II] - k[700]*y[IDX_O2II];
    data[4782] = 0.0 - k[39]*y[IDX_O2II] - k[656]*y[IDX_O2II];
    data[4783] = 0.0 - k[321]*y[IDX_O2II] - k[1482]*y[IDX_O2II];
    data[4784] = 0.0 - k[91]*y[IDX_O2II] - k[858]*y[IDX_O2II];
    data[4785] = 0.0 - k[70]*y[IDX_O2II] - k[769]*y[IDX_O2II];
    data[4786] = 0.0 - k[1483]*y[IDX_O2II];
    data[4787] = 0.0 + k[79]*y[IDX_O2I];
    data[4788] = 0.0 + k[324]*y[IDX_O2I];
    data[4789] = 0.0 + k[98]*y[IDX_O2I];
    data[4790] = 0.0 + k[105]*y[IDX_O2I];
    data[4791] = 0.0 + k[1211]*y[IDX_HeII] + k[1470]*y[IDX_OII];
    data[4792] = 0.0 - k[577]*y[IDX_O2II];
    data[4793] = 0.0 + k[135]*y[IDX_O2I];
    data[4794] = 0.0 + k[170]*y[IDX_O2I];
    data[4795] = 0.0 - k[174]*y[IDX_O2II] - k[972]*y[IDX_O2II];
    data[4796] = 0.0 + k[183]*y[IDX_O2I] + k[1497]*y[IDX_OI];
    data[4797] = 0.0 - k[322]*y[IDX_O2II];
    data[4798] = 0.0 + k[198]*y[IDX_O2I];
    data[4799] = 0.0 - k[204]*y[IDX_O2II] - k[1161]*y[IDX_O2II];
    data[4800] = 0.0 + k[217]*y[IDX_O2I] + k[1211]*y[IDX_CO2I];
    data[4801] = 0.0 - k[228]*y[IDX_O2II];
    data[4802] = 0.0 - k[1339]*y[IDX_O2II];
    data[4803] = 0.0 + k[249]*y[IDX_O2I];
    data[4804] = 0.0 + k[256]*y[IDX_O2I];
    data[4805] = 0.0 - k[1458]*y[IDX_O2II];
    data[4806] = 0.0 + k[264]*y[IDX_O2I];
    data[4807] = 0.0 - k[276]*y[IDX_O2II];
    data[4808] = 0.0 - k[290]*y[IDX_O2II];
    data[4809] = 0.0 - k[302]*y[IDX_O2II];
    data[4810] = 0.0 + k[1497]*y[IDX_H2OII] + k[1511]*y[IDX_OHII];
    data[4811] = 0.0 + k[317]*y[IDX_O2I] + k[1470]*y[IDX_CO2I] +
        k[1480]*y[IDX_OHI];
    data[4812] = 0.0 + k[79]*y[IDX_CH4II] + k[98]*y[IDX_CNII] +
        k[105]*y[IDX_COII] + k[135]*y[IDX_HII] + k[170]*y[IDX_H2II] +
        k[183]*y[IDX_H2OII] + k[198]*y[IDX_HCNII] + k[217]*y[IDX_HeII] +
        k[249]*y[IDX_NII] + k[256]*y[IDX_N2II] + k[264]*y[IDX_NHII] +
        k[317]*y[IDX_OII] + k[324]*y[IDX_ClII] + k[325]*y[IDX_SO2II] +
        k[337]*y[IDX_OHII] + k[435] + k[2061];
    data[4813] = 0.0 - k[39]*y[IDX_C2I] - k[54]*y[IDX_CI] -
        k[70]*y[IDX_CH2I] - k[91]*y[IDX_CHI] - k[174]*y[IDX_H2COI] -
        k[204]*y[IDX_HCOI] - k[228]*y[IDX_MgI] - k[276]*y[IDX_NH2I] -
        k[290]*y[IDX_NH3I] - k[302]*y[IDX_NOI] - k[321]*y[IDX_C2H2I] -
        k[322]*y[IDX_H2SI] - k[323]*y[IDX_SI] - k[353]*y[IDX_SiI] -
        k[577]*y[IDX_EM] - k[656]*y[IDX_C2I] - k[700]*y[IDX_CI] -
        k[769]*y[IDX_CH2I] - k[858]*y[IDX_CHI] - k[972]*y[IDX_H2COI] -
        k[1161]*y[IDX_HCOI] - k[1339]*y[IDX_NI] - k[1458]*y[IDX_NHI] -
        k[1482]*y[IDX_C2H2I] - k[1483]*y[IDX_CH3OHI] - k[1484]*y[IDX_SI] -
        k[2060] - k[2229];
    data[4814] = 0.0 + k[1480]*y[IDX_OII];
    data[4815] = 0.0 + k[337]*y[IDX_O2I] + k[1511]*y[IDX_OI];
    data[4816] = 0.0 - k[323]*y[IDX_O2II] - k[1484]*y[IDX_O2II];
    data[4817] = 0.0 - k[353]*y[IDX_O2II];
    data[4818] = 0.0 + k[325]*y[IDX_O2I];
    data[4819] = 0.0 + k[2387] + k[2388] + k[2389] + k[2390];
    data[4820] = 0.0 + k[1577]*y[IDX_O2I];
    data[4821] = 0.0 - k[1691]*y[IDX_O2HI] - k[1692]*y[IDX_O2HI];
    data[4822] = 0.0 + k[1662]*y[IDX_O2I] - k[1663]*y[IDX_O2HI];
    data[4823] = 0.0 + k[1671]*y[IDX_O2I];
    data[4824] = 0.0 - k[1719]*y[IDX_O2HI];
    data[4825] = 0.0 - k[1769]*y[IDX_O2HI] - k[1770]*y[IDX_O2HI] -
        k[1771]*y[IDX_O2HI];
    data[4826] = 0.0 + k[1731]*y[IDX_O2I];
    data[4827] = 0.0 + k[967]*y[IDX_O2I];
    data[4828] = 0.0 + k[1785]*y[IDX_O2I] - k[1786]*y[IDX_O2HI];
    data[4829] = 0.0 - k[1826]*y[IDX_O2HI];
    data[4830] = 0.0 - k[1909]*y[IDX_O2HI];
    data[4831] = 0.0 + k[967]*y[IDX_H2COII] + k[1577]*y[IDX_C2H3I] +
        k[1662]*y[IDX_CH3I] + k[1671]*y[IDX_CH4I] + k[1731]*y[IDX_H2I] +
        k[1785]*y[IDX_HCOI];
    data[4832] = 0.0 - k[437] - k[1663]*y[IDX_CH3I] - k[1691]*y[IDX_CHI] -
        k[1692]*y[IDX_CHI] - k[1719]*y[IDX_COI] - k[1769]*y[IDX_HI] -
        k[1770]*y[IDX_HI] - k[1771]*y[IDX_HI] - k[1786]*y[IDX_HCOI] -
        k[1826]*y[IDX_NI] - k[1909]*y[IDX_OI] - k[1946]*y[IDX_OHI] - k[2063] -
        k[2064] - k[2294];
    data[4833] = 0.0 - k[1946]*y[IDX_O2HI];
    data[4834] = 0.0 - k[701]*y[IDX_O2HII];
    data[4835] = 0.0 - k[657]*y[IDX_O2HII];
    data[4836] = 0.0 - k[683]*y[IDX_O2HII];
    data[4837] = 0.0 - k[859]*y[IDX_O2HII];
    data[4838] = 0.0 - k[770]*y[IDX_O2HII];
    data[4839] = 0.0 - k[870]*y[IDX_O2HII];
    data[4840] = 0.0 - k[878]*y[IDX_O2HII];
    data[4841] = 0.0 - k[1489]*y[IDX_O2HII];
    data[4842] = 0.0 - k[578]*y[IDX_O2HII];
    data[4843] = 0.0 - k[959]*y[IDX_O2HII];
    data[4844] = 0.0 + k[931]*y[IDX_O2I];
    data[4845] = 0.0 - k[973]*y[IDX_O2HII];
    data[4846] = 0.0 - k[1012]*y[IDX_O2HII];
    data[4847] = 0.0 + k[1062]*y[IDX_O2I];
    data[4848] = 0.0 - k[1129]*y[IDX_O2HII];
    data[4849] = 0.0 + k[1161]*y[IDX_O2II] - k[1162]*y[IDX_O2HII];
    data[4850] = 0.0 - k[1172]*y[IDX_O2HII];
    data[4851] = 0.0 - k[1320]*y[IDX_O2HII];
    data[4852] = 0.0 - k[1459]*y[IDX_O2HII];
    data[4853] = 0.0 + k[1369]*y[IDX_O2I];
    data[4854] = 0.0 - k[1410]*y[IDX_O2HII];
    data[4855] = 0.0 - k[1441]*y[IDX_O2HII];
    data[4856] = 0.0 - k[1462]*y[IDX_O2HII];
    data[4857] = 0.0 - k[1510]*y[IDX_O2HII];
    data[4858] = 0.0 + k[931]*y[IDX_H2II] + k[1062]*y[IDX_H3II] +
        k[1369]*y[IDX_NHII];
    data[4859] = 0.0 + k[1161]*y[IDX_HCOI];
    data[4860] = 0.0 - k[578]*y[IDX_EM] - k[657]*y[IDX_C2I] -
        k[683]*y[IDX_C2HI] - k[701]*y[IDX_CI] - k[770]*y[IDX_CH2I] -
        k[859]*y[IDX_CHI] - k[870]*y[IDX_CNI] - k[878]*y[IDX_COI] -
        k[959]*y[IDX_H2I] - k[973]*y[IDX_H2COI] - k[1012]*y[IDX_H2OI] -
        k[1129]*y[IDX_HCNI] - k[1162]*y[IDX_HCOI] - k[1172]*y[IDX_HNCI] -
        k[1320]*y[IDX_N2I] - k[1410]*y[IDX_NH2I] - k[1441]*y[IDX_NH3I] -
        k[1459]*y[IDX_NHI] - k[1462]*y[IDX_NOI] - k[1489]*y[IDX_CO2I] -
        k[1510]*y[IDX_OI] - k[1547]*y[IDX_OHI] - k[1554]*y[IDX_SI] - k[2274];
    data[4861] = 0.0 - k[1547]*y[IDX_O2HII];
    data[4862] = 0.0 - k[1554]*y[IDX_O2HII];
    data[4863] = 0.0 - k[1609]*y[IDX_OCNI];
    data[4864] = 0.0 - k[635]*y[IDX_OCNI];
    data[4865] = 0.0 + k[1686]*y[IDX_NOI];
    data[4866] = 0.0 + k[1709]*y[IDX_NO2I] + k[1711]*y[IDX_NOI] +
        k[1713]*y[IDX_O2I] + k[1933]*y[IDX_OHI];
    data[4867] = 0.0 - k[1772]*y[IDX_OCNI] - k[1773]*y[IDX_OCNI] -
        k[1774]*y[IDX_OCNI];
    data[4868] = 0.0 + k[1885]*y[IDX_OI];
    data[4869] = 0.0 + k[1891]*y[IDX_OI];
    data[4870] = 0.0 + k[1813]*y[IDX_NI];
    data[4871] = 0.0 - k[1262]*y[IDX_OCNI] - k[1263]*y[IDX_OCNI];
    data[4872] = 0.0 + k[1813]*y[IDX_HCOI];
    data[4873] = 0.0 + k[1943]*y[IDX_OHI];
    data[4874] = 0.0 + k[1686]*y[IDX_CHI] + k[1711]*y[IDX_CNI] -
        k[1860]*y[IDX_OCNI];
    data[4875] = 0.0 + k[1709]*y[IDX_CNI];
    data[4876] = 0.0 + k[1885]*y[IDX_H2CNI] + k[1891]*y[IDX_HCNI] -
        k[1910]*y[IDX_OCNI] - k[1911]*y[IDX_OCNI];
    data[4877] = 0.0 + k[1713]*y[IDX_CNI] - k[1863]*y[IDX_OCNI] -
        k[1864]*y[IDX_OCNI];
    data[4878] = 0.0 - k[439] - k[635]*y[IDX_CII] - k[1262]*y[IDX_HeII] -
        k[1263]*y[IDX_HeII] - k[1609]*y[IDX_CI] - k[1772]*y[IDX_HI] -
        k[1773]*y[IDX_HI] - k[1774]*y[IDX_HI] - k[1860]*y[IDX_NOI] -
        k[1863]*y[IDX_O2I] - k[1864]*y[IDX_O2I] - k[1910]*y[IDX_OI] -
        k[1911]*y[IDX_OI] - k[2065] - k[2158];
    data[4879] = 0.0 + k[1933]*y[IDX_CNI] + k[1943]*y[IDX_NCCNI];
    data[4880] = 0.0 + k[2487] + k[2488] + k[2489] + k[2490];
    data[4881] = 0.0 - k[1610]*y[IDX_OCSI];
    data[4882] = 0.0 - k[24]*y[IDX_OCSI] - k[636]*y[IDX_OCSI];
    data[4883] = 0.0 - k[1695]*y[IDX_OCSI] + k[1700]*y[IDX_SOI];
    data[4884] = 0.0 - k[736]*y[IDX_OCSI] - k[737]*y[IDX_OCSI];
    data[4885] = 0.0 - k[751]*y[IDX_OCSI] - k[752]*y[IDX_OCSI];
    data[4886] = 0.0 - k[785]*y[IDX_OCSI];
    data[4887] = 0.0 - k[80]*y[IDX_OCSI] - k[802]*y[IDX_OCSI];
    data[4888] = 0.0 + k[1936]*y[IDX_OHI];
    data[4889] = 0.0 + k[553]*y[IDX_HOCSII];
    data[4890] = 0.0 - k[1775]*y[IDX_OCSI];
    data[4891] = 0.0 - k[137]*y[IDX_OCSI] - k[904]*y[IDX_OCSI];
    data[4892] = 0.0 + k[1006]*y[IDX_HOCSII];
    data[4893] = 0.0 - k[184]*y[IDX_OCSI];
    data[4894] = 0.0 + k[190]*y[IDX_OCSII];
    data[4895] = 0.0 - k[1065]*y[IDX_OCSI];
    data[4896] = 0.0 + k[1951]*y[IDX_SI];
    data[4897] = 0.0 - k[1149]*y[IDX_OCSI];
    data[4898] = 0.0 + k[1895]*y[IDX_OI];
    data[4899] = 0.0 - k[1264]*y[IDX_OCSI] - k[1265]*y[IDX_OCSI] -
        k[1266]*y[IDX_OCSI] - k[1267]*y[IDX_OCSI];
    data[4900] = 0.0 + k[553]*y[IDX_EM] + k[1006]*y[IDX_H2OI];
    data[4901] = 0.0 - k[250]*y[IDX_OCSI] - k[1312]*y[IDX_OCSI] -
        k[1313]*y[IDX_OCSI];
    data[4902] = 0.0 - k[257]*y[IDX_OCSI] - k[1318]*y[IDX_OCSI];
    data[4903] = 0.0 + k[291]*y[IDX_OCSII];
    data[4904] = 0.0 + k[1895]*y[IDX_HCSI] - k[1912]*y[IDX_OCSI] -
        k[1913]*y[IDX_OCSI];
    data[4905] = 0.0 - k[318]*y[IDX_OCSI] - k[1479]*y[IDX_OCSI];
    data[4906] = 0.0 - k[24]*y[IDX_CII] - k[80]*y[IDX_CH4II] -
        k[137]*y[IDX_HII] - k[184]*y[IDX_H2OII] - k[250]*y[IDX_NII] -
        k[257]*y[IDX_N2II] - k[318]*y[IDX_OII] - k[440] - k[441] -
        k[636]*y[IDX_CII] - k[736]*y[IDX_CHII] - k[737]*y[IDX_CHII] -
        k[751]*y[IDX_CH2II] - k[752]*y[IDX_CH2II] - k[785]*y[IDX_CH3II] -
        k[802]*y[IDX_CH4II] - k[904]*y[IDX_HII] - k[1065]*y[IDX_H3II] -
        k[1149]*y[IDX_HCOII] - k[1264]*y[IDX_HeII] - k[1265]*y[IDX_HeII] -
        k[1266]*y[IDX_HeII] - k[1267]*y[IDX_HeII] - k[1312]*y[IDX_NII] -
        k[1313]*y[IDX_NII] - k[1318]*y[IDX_N2II] - k[1479]*y[IDX_OII] -
        k[1552]*y[IDX_SII] - k[1562]*y[IDX_SOII] - k[1565]*y[IDX_SiII] -
        k[1610]*y[IDX_CI] - k[1695]*y[IDX_CHI] - k[1775]*y[IDX_HI] -
        k[1912]*y[IDX_OI] - k[1913]*y[IDX_OI] - k[2066] - k[2067] - k[2259];
    data[4907] = 0.0 + k[190]*y[IDX_H2SI] + k[291]*y[IDX_NH3I];
    data[4908] = 0.0 + k[1936]*y[IDX_CSI];
    data[4909] = 0.0 + k[1951]*y[IDX_HCOI];
    data[4910] = 0.0 - k[1552]*y[IDX_OCSI];
    data[4911] = 0.0 - k[1565]*y[IDX_OCSI];
    data[4912] = 0.0 + k[1700]*y[IDX_CHI];
    data[4913] = 0.0 - k[1562]*y[IDX_OCSI];
    data[4914] = 0.0 + k[24]*y[IDX_OCSI];
    data[4915] = 0.0 + k[80]*y[IDX_OCSI];
    data[4916] = 0.0 + k[1485]*y[IDX_O2I];
    data[4917] = 0.0 - k[579]*y[IDX_OCSII] - k[580]*y[IDX_OCSII] -
        k[581]*y[IDX_OCSII];
    data[4918] = 0.0 + k[137]*y[IDX_OCSI];
    data[4919] = 0.0 + k[184]*y[IDX_OCSI];
    data[4920] = 0.0 - k[190]*y[IDX_OCSII];
    data[4921] = 0.0 + k[1501]*y[IDX_OI];
    data[4922] = 0.0 + k[250]*y[IDX_OCSI];
    data[4923] = 0.0 + k[257]*y[IDX_OCSI];
    data[4924] = 0.0 - k[291]*y[IDX_OCSII];
    data[4925] = 0.0 + k[1501]*y[IDX_HCSII];
    data[4926] = 0.0 + k[318]*y[IDX_OCSI];
    data[4927] = 0.0 + k[1485]*y[IDX_CSII];
    data[4928] = 0.0 + k[24]*y[IDX_CII] + k[80]*y[IDX_CH4II] +
        k[137]*y[IDX_HII] + k[184]*y[IDX_H2OII] + k[250]*y[IDX_NII] +
        k[257]*y[IDX_N2II] + k[318]*y[IDX_OII] + k[440] + k[2066];
    data[4929] = 0.0 - k[190]*y[IDX_H2SI] - k[291]*y[IDX_NH3I] -
        k[579]*y[IDX_EM] - k[580]*y[IDX_EM] - k[581]*y[IDX_EM] - k[2269];
    data[4930] = 0.0 + k[690]*y[IDX_H2OII] - k[1611]*y[IDX_OHI] -
        k[1612]*y[IDX_OHI];
    data[4931] = 0.0 - k[637]*y[IDX_OHI];
    data[4932] = 0.0 + k[329]*y[IDX_OHII] + k[976]*y[IDX_H2OII];
    data[4933] = 0.0 - k[339]*y[IDX_OHI] + k[990]*y[IDX_H2OI];
    data[4934] = 0.0 + k[330]*y[IDX_OHII] + k[977]*y[IDX_H2OII];
    data[4935] = 0.0 - k[1927]*y[IDX_OHI] - k[1928]*y[IDX_OHI] -
        k[1929]*y[IDX_OHI];
    data[4936] = 0.0 - k[1930]*y[IDX_OHI];
    data[4937] = 0.0 + k[1870]*y[IDX_OI];
    data[4938] = 0.0 - k[1931]*y[IDX_OHI];
    data[4939] = 0.0 + k[372] + k[1969];
    data[4940] = 0.0 + k[466]*y[IDX_EM];
    data[4941] = 0.0 + k[992]*y[IDX_H2OI];
    data[4942] = 0.0 + k[92]*y[IDX_OHII] + k[843]*y[IDX_H2OII] +
        k[1689]*y[IDX_O2I] + k[1691]*y[IDX_O2HI] + k[1694]*y[IDX_OI] -
        k[1696]*y[IDX_OHI];
    data[4943] = 0.0 + k[732]*y[IDX_O2I] - k[738]*y[IDX_OHI];
    data[4944] = 0.0 + k[71]*y[IDX_OHII] + k[758]*y[IDX_H2OII] +
        k[1630]*y[IDX_NOI] + k[1636]*y[IDX_O2I] + k[1640]*y[IDX_OI] -
        k[1641]*y[IDX_OHI] - k[1642]*y[IDX_OHI] - k[1643]*y[IDX_OHI];
    data[4945] = 0.0 + k[749]*y[IDX_O2I];
    data[4946] = 0.0 + k[1652]*y[IDX_H2OI] + k[1660]*y[IDX_O2I] -
        k[1666]*y[IDX_OHI] - k[1667]*y[IDX_OHI] - k[1668]*y[IDX_OHI];
    data[4947] = 0.0 - k[786]*y[IDX_OHI];
    data[4948] = 0.0 + k[389] + k[1201]*y[IDX_HeII] + k[1467]*y[IDX_OII] +
        k[1992];
    data[4949] = 0.0 + k[488]*y[IDX_EM];
    data[4950] = 0.0 + k[1468]*y[IDX_OII] - k[1672]*y[IDX_OHI] +
        k[1879]*y[IDX_OI];
    data[4951] = 0.0 + k[1493]*y[IDX_OI];
    data[4952] = 0.0 - k[1538]*y[IDX_OHI];
    data[4953] = 0.0 - k[1932]*y[IDX_OHI] - k[1933]*y[IDX_OHI];
    data[4954] = 0.0 - k[340]*y[IDX_OHI] + k[995]*y[IDX_H2OI];
    data[4955] = 0.0 + k[978]*y[IDX_H2OII] + k[1719]*y[IDX_O2HI] +
        k[1745]*y[IDX_HI] - k[1934]*y[IDX_OHI];
    data[4956] = 0.0 - k[341]*y[IDX_OHI] + k[997]*y[IDX_H2OI] -
        k[1539]*y[IDX_OHI];
    data[4957] = 0.0 + k[1744]*y[IDX_HI];
    data[4958] = 0.0 - k[1935]*y[IDX_OHI] - k[1936]*y[IDX_OHI];
    data[4959] = 0.0 + k[466]*y[IDX_C2H5OH2II] + k[488]*y[IDX_CH3OH2II] +
        k[514]*y[IDX_H2OII] + k[521]*y[IDX_H3COII] + k[530]*y[IDX_H3OII] +
        k[531]*y[IDX_H3OII] + k[545]*y[IDX_HCO2II] + k[552]*y[IDX_HOCSII] +
        k[560]*y[IDX_HSO2II] + k[603]*y[IDX_SiOHII];
    data[4960] = 0.0 + k[11]*y[IDX_H2OI] - k[13]*y[IDX_OHI] +
        k[1107]*y[IDX_SO2II] + k[1744]*y[IDX_CO2I] + k[1745]*y[IDX_COI] +
        k[1748]*y[IDX_H2OI] + k[1757]*y[IDX_HNOI] + k[1763]*y[IDX_NO2I] +
        k[1765]*y[IDX_NOI] + k[1768]*y[IDX_O2I] + k[1771]*y[IDX_O2HI] +
        k[1771]*y[IDX_O2HI] + k[1774]*y[IDX_OCNI] - k[1776]*y[IDX_OHI] +
        k[1779]*y[IDX_SOI] + k[2124]*y[IDX_OI] - k[2125]*y[IDX_OHI];
    data[4961] = 0.0 - k[138]*y[IDX_OHI] + k[903]*y[IDX_NO2I];
    data[4962] = 0.0 + k[4]*y[IDX_H2OI] - k[7]*y[IDX_OHI] +
        k[1732]*y[IDX_O2I] + k[1732]*y[IDX_O2I] + k[1733]*y[IDX_OI] -
        k[1734]*y[IDX_OHI];
    data[4963] = 0.0 - k[171]*y[IDX_OHI] - k[933]*y[IDX_OHI];
    data[4964] = 0.0 + k[331]*y[IDX_OHII] + k[979]*y[IDX_H2OII] +
        k[1471]*y[IDX_OII] + k[1886]*y[IDX_OI] - k[1937]*y[IDX_OHI];
    data[4965] = 0.0 + k[4]*y[IDX_H2I] + k[11]*y[IDX_HI] +
        k[332]*y[IDX_OHII] + k[401] + k[980]*y[IDX_H2OII] + k[990]*y[IDX_C2II] +
        k[992]*y[IDX_C2NII] + k[995]*y[IDX_CNII] + k[997]*y[IDX_COII] +
        k[1010]*y[IDX_N2II] + k[1223]*y[IDX_HeII] + k[1359]*y[IDX_NHII] +
        k[1379]*y[IDX_NH2II] + k[1414]*y[IDX_NH3II] + k[1652]*y[IDX_CH3I] +
        k[1748]*y[IDX_HI] + k[1841]*y[IDX_NHI] + k[1887]*y[IDX_OI] +
        k[1887]*y[IDX_OI] + k[2018];
    data[4966] = 0.0 + k[514]*y[IDX_EM] + k[690]*y[IDX_CI] +
        k[758]*y[IDX_CH2I] + k[843]*y[IDX_CHI] + k[976]*y[IDX_C2I] +
        k[977]*y[IDX_C2HI] + k[978]*y[IDX_COI] + k[979]*y[IDX_H2COI] +
        k[980]*y[IDX_H2OI] + k[981]*y[IDX_H2SI] + k[983]*y[IDX_HCNI] +
        k[985]*y[IDX_HCOI] + k[986]*y[IDX_HNCI] + k[987]*y[IDX_SI] +
        k[989]*y[IDX_SO2I] + k[1400]*y[IDX_NH2I] + k[1425]*y[IDX_NH3I] -
        k[1540]*y[IDX_OHI];
    data[4967] = 0.0 + k[333]*y[IDX_OHII] + k[981]*y[IDX_H2OII] +
        k[1472]*y[IDX_OII] + k[1888]*y[IDX_OI] - k[1938]*y[IDX_OHI];
    data[4968] = 0.0 + k[1498]*y[IDX_OI];
    data[4969] = 0.0 + k[1059]*y[IDX_NO2I] - k[1066]*y[IDX_OHI];
    data[4970] = 0.0 + k[521]*y[IDX_EM];
    data[4971] = 0.0 + k[530]*y[IDX_EM] + k[531]*y[IDX_EM];
    data[4972] = 0.0 + k[983]*y[IDX_H2OII] + k[1889]*y[IDX_OI] -
        k[1939]*y[IDX_OHI] - k[1940]*y[IDX_OHI];
    data[4973] = 0.0 - k[1541]*y[IDX_OHI];
    data[4974] = 0.0 + k[334]*y[IDX_OHII] + k[985]*y[IDX_H2OII] +
        k[1784]*y[IDX_O2I] + k[1893]*y[IDX_OI] - k[1941]*y[IDX_OHI];
    data[4975] = 0.0 - k[1542]*y[IDX_OHI] - k[1543]*y[IDX_OHI];
    data[4976] = 0.0 + k[545]*y[IDX_EM];
    data[4977] = 0.0 + k[1201]*y[IDX_CH3OHI] + k[1223]*y[IDX_H2OI] -
        k[1268]*y[IDX_OHI];
    data[4978] = 0.0 + k[986]*y[IDX_H2OII];
    data[4979] = 0.0 + k[1757]*y[IDX_HI] + k[1897]*y[IDX_OI] -
        k[1942]*y[IDX_OHI];
    data[4980] = 0.0 - k[1544]*y[IDX_OHI];
    data[4981] = 0.0 + k[552]*y[IDX_EM];
    data[4982] = 0.0 + k[1899]*y[IDX_OI];
    data[4983] = 0.0 + k[1503]*y[IDX_OI];
    data[4984] = 0.0 + k[560]*y[IDX_EM];
    data[4985] = 0.0 - k[1827]*y[IDX_OHI] - k[1828]*y[IDX_OHI];
    data[4986] = 0.0 - k[251]*y[IDX_OHI];
    data[4987] = 0.0 - k[342]*y[IDX_OHI] + k[1010]*y[IDX_H2OI];
    data[4988] = 0.0 - k[1545]*y[IDX_OHI];
    data[4989] = 0.0 - k[1943]*y[IDX_OHI];
    data[4990] = 0.0 + k[1841]*y[IDX_H2OI] + k[1848]*y[IDX_NOI] +
        k[1850]*y[IDX_O2I] + k[1852]*y[IDX_OI] - k[1853]*y[IDX_OHI] -
        k[1854]*y[IDX_OHI] - k[1855]*y[IDX_OHI];
    data[4991] = 0.0 + k[1359]*y[IDX_H2OI] + k[1368]*y[IDX_O2I] -
        k[1371]*y[IDX_OHI];
    data[4992] = 0.0 + k[277]*y[IDX_OHII] + k[1400]*y[IDX_H2OII] +
        k[1835]*y[IDX_NOI] - k[1836]*y[IDX_OHI] - k[1837]*y[IDX_OHI] +
        k[1903]*y[IDX_OI];
    data[4993] = 0.0 + k[1379]*y[IDX_H2OI] + k[1391]*y[IDX_O2I];
    data[4994] = 0.0 + k[335]*y[IDX_OHII] + k[1425]*y[IDX_H2OII] +
        k[1904]*y[IDX_OI] - k[1944]*y[IDX_OHI];
    data[4995] = 0.0 + k[1414]*y[IDX_H2OI] - k[1546]*y[IDX_OHI];
    data[4996] = 0.0 + k[336]*y[IDX_OHII] + k[1630]*y[IDX_CH2I] +
        k[1765]*y[IDX_HI] + k[1835]*y[IDX_NH2I] + k[1848]*y[IDX_NHI] -
        k[1945]*y[IDX_OHI];
    data[4997] = 0.0 + k[903]*y[IDX_HII] + k[1059]*y[IDX_H3II] +
        k[1763]*y[IDX_HI];
    data[4998] = 0.0 + k[1493]*y[IDX_CH4II] + k[1498]*y[IDX_H2SII] +
        k[1503]*y[IDX_HSII] + k[1640]*y[IDX_CH2I] + k[1694]*y[IDX_CHI] +
        k[1733]*y[IDX_H2I] + k[1852]*y[IDX_NHI] + k[1870]*y[IDX_C2H4I] +
        k[1879]*y[IDX_CH4I] + k[1886]*y[IDX_H2COI] + k[1887]*y[IDX_H2OI] +
        k[1887]*y[IDX_H2OI] + k[1888]*y[IDX_H2SI] + k[1889]*y[IDX_HCNI] +
        k[1893]*y[IDX_HCOI] + k[1897]*y[IDX_HNOI] + k[1899]*y[IDX_HSI] +
        k[1903]*y[IDX_NH2I] + k[1904]*y[IDX_NH3I] + k[1909]*y[IDX_O2HI] -
        k[1914]*y[IDX_OHI] + k[1925]*y[IDX_SiH4I] + k[2124]*y[IDX_HI];
    data[4999] = 0.0 - k[319]*y[IDX_OHI] + k[1467]*y[IDX_CH3OHI] +
        k[1468]*y[IDX_CH4I] + k[1471]*y[IDX_H2COI] + k[1472]*y[IDX_H2SI] -
        k[1480]*y[IDX_OHI];
    data[5000] = 0.0 + k[337]*y[IDX_OHII] + k[732]*y[IDX_CHII] +
        k[749]*y[IDX_CH2II] + k[1368]*y[IDX_NHII] + k[1391]*y[IDX_NH2II] +
        k[1567]*y[IDX_SiH2II] + k[1636]*y[IDX_CH2I] + k[1660]*y[IDX_CH3I] +
        k[1689]*y[IDX_CHI] + k[1732]*y[IDX_H2I] + k[1732]*y[IDX_H2I] +
        k[1768]*y[IDX_HI] + k[1784]*y[IDX_HCOI] + k[1850]*y[IDX_NHI];
    data[5001] = 0.0 + k[1691]*y[IDX_CHI] + k[1719]*y[IDX_COI] +
        k[1771]*y[IDX_HI] + k[1771]*y[IDX_HI] + k[1909]*y[IDX_OI] -
        k[1946]*y[IDX_OHI] + k[2064];
    data[5002] = 0.0 - k[1547]*y[IDX_OHI];
    data[5003] = 0.0 + k[1774]*y[IDX_HI];
    data[5004] = 0.0 - k[7]*y[IDX_H2I] - k[13]*y[IDX_HI] - k[138]*y[IDX_HII]
        - k[171]*y[IDX_H2II] - k[251]*y[IDX_NII] - k[319]*y[IDX_OII] -
        k[339]*y[IDX_C2II] - k[340]*y[IDX_CNII] - k[341]*y[IDX_COII] -
        k[342]*y[IDX_N2II] - k[442] - k[637]*y[IDX_CII] - k[738]*y[IDX_CHII] -
        k[786]*y[IDX_CH3II] - k[933]*y[IDX_H2II] - k[1066]*y[IDX_H3II] -
        k[1268]*y[IDX_HeII] - k[1371]*y[IDX_NHII] - k[1480]*y[IDX_OII] -
        k[1532]*y[IDX_OHII] - k[1538]*y[IDX_CH5II] - k[1539]*y[IDX_COII] -
        k[1540]*y[IDX_H2OII] - k[1541]*y[IDX_HCNII] - k[1542]*y[IDX_HCOII] -
        k[1543]*y[IDX_HCOII] - k[1544]*y[IDX_HNOII] - k[1545]*y[IDX_N2HII] -
        k[1546]*y[IDX_NH3II] - k[1547]*y[IDX_O2HII] - k[1548]*y[IDX_SII] -
        k[1549]*y[IDX_SiII] - k[1611]*y[IDX_CI] - k[1612]*y[IDX_CI] -
        k[1641]*y[IDX_CH2I] - k[1642]*y[IDX_CH2I] - k[1643]*y[IDX_CH2I] -
        k[1666]*y[IDX_CH3I] - k[1667]*y[IDX_CH3I] - k[1668]*y[IDX_CH3I] -
        k[1672]*y[IDX_CH4I] - k[1696]*y[IDX_CHI] - k[1734]*y[IDX_H2I] -
        k[1776]*y[IDX_HI] - k[1827]*y[IDX_NI] - k[1828]*y[IDX_NI] -
        k[1836]*y[IDX_NH2I] - k[1837]*y[IDX_NH2I] - k[1853]*y[IDX_NHI] -
        k[1854]*y[IDX_NHI] - k[1855]*y[IDX_NHI] - k[1914]*y[IDX_OI] -
        k[1927]*y[IDX_C2H2I] - k[1928]*y[IDX_C2H2I] - k[1929]*y[IDX_C2H2I] -
        k[1930]*y[IDX_C2H3I] - k[1931]*y[IDX_C2H5I] - k[1932]*y[IDX_CNI] -
        k[1933]*y[IDX_CNI] - k[1934]*y[IDX_COI] - k[1935]*y[IDX_CSI] -
        k[1936]*y[IDX_CSI] - k[1937]*y[IDX_H2COI] - k[1938]*y[IDX_H2SI] -
        k[1939]*y[IDX_HCNI] - k[1940]*y[IDX_HCNI] - k[1941]*y[IDX_HCOI] -
        k[1942]*y[IDX_HNOI] - k[1943]*y[IDX_NCCNI] - k[1944]*y[IDX_NH3I] -
        k[1945]*y[IDX_NOI] - k[1946]*y[IDX_O2HI] - k[1947]*y[IDX_OHI] -
        k[1947]*y[IDX_OHI] - k[1947]*y[IDX_OHI] - k[1947]*y[IDX_OHI] -
        k[1948]*y[IDX_SI] - k[1949]*y[IDX_SOI] - k[1950]*y[IDX_SiI] - k[2069] -
        k[2070] - k[2125]*y[IDX_HI] - k[2211];
    data[5005] = 0.0 + k[71]*y[IDX_CH2I] + k[92]*y[IDX_CHI] +
        k[277]*y[IDX_NH2I] + k[329]*y[IDX_C2I] + k[330]*y[IDX_C2HI] +
        k[331]*y[IDX_H2COI] + k[332]*y[IDX_H2OI] + k[333]*y[IDX_H2SI] +
        k[334]*y[IDX_HCOI] + k[335]*y[IDX_NH3I] + k[336]*y[IDX_NOI] +
        k[337]*y[IDX_O2I] + k[338]*y[IDX_SI] - k[1532]*y[IDX_OHI];
    data[5006] = 0.0 + k[338]*y[IDX_OHII] + k[987]*y[IDX_H2OII] -
        k[1948]*y[IDX_OHI];
    data[5007] = 0.0 - k[1548]*y[IDX_OHI];
    data[5008] = 0.0 - k[1950]*y[IDX_OHI];
    data[5009] = 0.0 - k[1549]*y[IDX_OHI];
    data[5010] = 0.0 + k[1567]*y[IDX_O2I];
    data[5011] = 0.0 + k[1925]*y[IDX_OI];
    data[5012] = 0.0 + k[603]*y[IDX_EM];
    data[5013] = 0.0 + k[1779]*y[IDX_HI] - k[1949]*y[IDX_OHI];
    data[5014] = 0.0 + k[989]*y[IDX_H2OII];
    data[5015] = 0.0 + k[1107]*y[IDX_HI];
    data[5016] = 0.0 - k[702]*y[IDX_OHII];
    data[5017] = 0.0 - k[329]*y[IDX_OHII] - k[1517]*y[IDX_OHII];
    data[5018] = 0.0 + k[339]*y[IDX_OHI];
    data[5019] = 0.0 - k[330]*y[IDX_OHII] - k[1518]*y[IDX_OHII];
    data[5020] = 0.0 - k[92]*y[IDX_OHII] - k[860]*y[IDX_OHII];
    data[5021] = 0.0 - k[71]*y[IDX_OHII] - k[771]*y[IDX_OHII];
    data[5022] = 0.0 + k[1200]*y[IDX_HeII];
    data[5023] = 0.0 - k[819]*y[IDX_OHII] - k[820]*y[IDX_OHII];
    data[5024] = 0.0 - k[1519]*y[IDX_OHII];
    data[5025] = 0.0 + k[340]*y[IDX_OHI];
    data[5026] = 0.0 - k[1521]*y[IDX_OHII];
    data[5027] = 0.0 + k[341]*y[IDX_OHI];
    data[5028] = 0.0 - k[1520]*y[IDX_OHII];
    data[5029] = 0.0 - k[582]*y[IDX_OHII];
    data[5030] = 0.0 + k[138]*y[IDX_OHI];
    data[5031] = 0.0 + k[958]*y[IDX_OII] - k[960]*y[IDX_OHII];
    data[5032] = 0.0 + k[171]*y[IDX_OHI] + k[932]*y[IDX_OI];
    data[5033] = 0.0 - k[331]*y[IDX_OHII] - k[1522]*y[IDX_OHII];
    data[5034] = 0.0 - k[332]*y[IDX_OHII] + k[1222]*y[IDX_HeII] -
        k[1523]*y[IDX_OHII];
    data[5035] = 0.0 + k[2016];
    data[5036] = 0.0 - k[333]*y[IDX_OHII] - k[1524]*y[IDX_OHII];
    data[5037] = 0.0 + k[1064]*y[IDX_OI];
    data[5038] = 0.0 - k[1525]*y[IDX_OHII];
    data[5039] = 0.0 - k[334]*y[IDX_OHII] + k[1476]*y[IDX_OII] -
        k[1526]*y[IDX_OHII] - k[1527]*y[IDX_OHII];
    data[5040] = 0.0 + k[1200]*y[IDX_CH3OHI] + k[1222]*y[IDX_H2OI];
    data[5041] = 0.0 - k[1528]*y[IDX_OHII];
    data[5042] = 0.0 - k[1340]*y[IDX_OHII];
    data[5043] = 0.0 + k[251]*y[IDX_OHI];
    data[5044] = 0.0 - k[1529]*y[IDX_OHII];
    data[5045] = 0.0 + k[342]*y[IDX_OHI];
    data[5046] = 0.0 + k[1506]*y[IDX_OI];
    data[5047] = 0.0 - k[1460]*y[IDX_OHII];
    data[5048] = 0.0 + k[1370]*y[IDX_OI];
    data[5049] = 0.0 - k[277]*y[IDX_OHII] - k[1411]*y[IDX_OHII];
    data[5050] = 0.0 - k[335]*y[IDX_OHII] - k[1530]*y[IDX_OHII];
    data[5051] = 0.0 - k[336]*y[IDX_OHII] - k[1531]*y[IDX_OHII];
    data[5052] = 0.0 + k[932]*y[IDX_H2II] + k[1064]*y[IDX_H3II] +
        k[1370]*y[IDX_NHII] + k[1506]*y[IDX_N2HII] + k[1510]*y[IDX_O2HII] -
        k[1511]*y[IDX_OHII];
    data[5053] = 0.0 + k[319]*y[IDX_OHI] + k[958]*y[IDX_H2I] +
        k[1476]*y[IDX_HCOI];
    data[5054] = 0.0 - k[337]*y[IDX_OHII];
    data[5055] = 0.0 + k[1510]*y[IDX_OI];
    data[5056] = 0.0 + k[138]*y[IDX_HII] + k[171]*y[IDX_H2II] +
        k[251]*y[IDX_NII] + k[319]*y[IDX_OII] + k[339]*y[IDX_C2II] +
        k[340]*y[IDX_CNII] + k[341]*y[IDX_COII] + k[342]*y[IDX_N2II] -
        k[1532]*y[IDX_OHII] + k[2070];
    data[5057] = 0.0 - k[71]*y[IDX_CH2I] - k[92]*y[IDX_CHI] -
        k[277]*y[IDX_NH2I] - k[329]*y[IDX_C2I] - k[330]*y[IDX_C2HI] -
        k[331]*y[IDX_H2COI] - k[332]*y[IDX_H2OI] - k[333]*y[IDX_H2SI] -
        k[334]*y[IDX_HCOI] - k[335]*y[IDX_NH3I] - k[336]*y[IDX_NOI] -
        k[337]*y[IDX_O2I] - k[338]*y[IDX_SI] - k[582]*y[IDX_EM] -
        k[702]*y[IDX_CI] - k[771]*y[IDX_CH2I] - k[819]*y[IDX_CH4I] -
        k[820]*y[IDX_CH4I] - k[860]*y[IDX_CHI] - k[960]*y[IDX_H2I] -
        k[1340]*y[IDX_NI] - k[1411]*y[IDX_NH2I] - k[1460]*y[IDX_NHI] -
        k[1511]*y[IDX_OI] - k[1517]*y[IDX_C2I] - k[1518]*y[IDX_C2HI] -
        k[1519]*y[IDX_CNI] - k[1520]*y[IDX_CO2I] - k[1521]*y[IDX_COI] -
        k[1522]*y[IDX_H2COI] - k[1523]*y[IDX_H2OI] - k[1524]*y[IDX_H2SI] -
        k[1525]*y[IDX_HCNI] - k[1526]*y[IDX_HCOI] - k[1527]*y[IDX_HCOI] -
        k[1528]*y[IDX_HNCI] - k[1529]*y[IDX_N2I] - k[1530]*y[IDX_NH3I] -
        k[1531]*y[IDX_NOI] - k[1532]*y[IDX_OHI] - k[1533]*y[IDX_SI] -
        k[1534]*y[IDX_SI] - k[1535]*y[IDX_SiI] - k[1536]*y[IDX_SiHI] -
        k[1537]*y[IDX_SiOI] - k[2068] - k[2233];
    data[5058] = 0.0 - k[338]*y[IDX_OHII] - k[1533]*y[IDX_OHII] -
        k[1534]*y[IDX_OHII];
    data[5059] = 0.0 - k[1535]*y[IDX_OHII];
    data[5060] = 0.0 - k[1536]*y[IDX_OHII];
    data[5061] = 0.0 - k[1537]*y[IDX_OHII];
    data[5062] = 0.0 + k[1592]*y[IDX_CSI] + k[1596]*y[IDX_HSI] +
        k[1607]*y[IDX_NSI] + k[1613]*y[IDX_S2I] + k[1616]*y[IDX_SOI] -
        k[2106]*y[IDX_SI];
    data[5063] = 0.0 - k[345]*y[IDX_SI] + k[641]*y[IDX_SOI] +
        k[646]*y[IDX_SiSI] - k[2099]*y[IDX_SI];
    data[5064] = 0.0 - k[1573]*y[IDX_SI];
    data[5065] = 0.0 - k[35]*y[IDX_SI] - k[650]*y[IDX_SI];
    data[5066] = 0.0 - k[41]*y[IDX_SI];
    data[5067] = 0.0 + k[851]*y[IDX_HSII] - k[1697]*y[IDX_SI] -
        k[1698]*y[IDX_SI];
    data[5068] = 0.0 - k[59]*y[IDX_SI] - k[739]*y[IDX_SI] -
        k[740]*y[IDX_SI];
    data[5069] = 0.0 - k[1644]*y[IDX_SI] - k[1645]*y[IDX_SI];
    data[5070] = 0.0 - k[753]*y[IDX_SI];
    data[5071] = 0.0 - k[1669]*y[IDX_SI];
    data[5072] = 0.0 - k[787]*y[IDX_SI];
    data[5073] = 0.0 - k[1673]*y[IDX_SI];
    data[5074] = 0.0 - k[835]*y[IDX_SI];
    data[5075] = 0.0 - k[1714]*y[IDX_SI];
    data[5076] = 0.0 - k[99]*y[IDX_SI];
    data[5077] = 0.0 - k[106]*y[IDX_SI];
    data[5078] = 0.0 + k[396] + k[1215]*y[IDX_HeII] + k[1592]*y[IDX_CI] +
        k[1809]*y[IDX_NI] + k[1883]*y[IDX_OI] + k[2007];
    data[5079] = 0.0 + k[500]*y[IDX_EM] + k[1496]*y[IDX_OI];
    data[5080] = 0.0 + k[500]*y[IDX_CSII] + k[516]*y[IDX_H2SII] +
        k[535]*y[IDX_H3SII] + k[546]*y[IDX_HCSII] + k[554]*y[IDX_HSII] +
        k[555]*y[IDX_HS2II] + k[576]*y[IDX_NSII] + k[581]*y[IDX_OCSII] +
        k[583]*y[IDX_S2II] + k[583]*y[IDX_S2II] + k[584]*y[IDX_SOII] +
        k[585]*y[IDX_SO2II] + k[605]*y[IDX_SiSII] + k[2143]*y[IDX_SII];
    data[5081] = 0.0 + k[1758]*y[IDX_HSI] + k[1767]*y[IDX_NSI] +
        k[1777]*y[IDX_S2I] + k[1779]*y[IDX_SOI];
    data[5082] = 0.0 - k[140]*y[IDX_SI];
    data[5083] = 0.0 - k[1735]*y[IDX_SI];
    data[5084] = 0.0 - k[173]*y[IDX_SI] - k[968]*y[IDX_SI];
    data[5085] = 0.0 + k[1221]*y[IDX_HeII];
    data[5086] = 0.0 + k[1007]*y[IDX_HSII];
    data[5087] = 0.0 - k[185]*y[IDX_SI] - k[987]*y[IDX_SI] -
        k[988]*y[IDX_SI];
    data[5088] = 0.0 + k[404] + k[1175]*y[IDX_HSII] + k[1384]*y[IDX_NH2II] +
        k[2022];
    data[5089] = 0.0 - k[346]*y[IDX_SI] + k[516]*y[IDX_EM];
    data[5090] = 0.0 - k[1068]*y[IDX_SI];
    data[5091] = 0.0 + k[535]*y[IDX_EM] - k[1553]*y[IDX_SI];
    data[5092] = 0.0 + k[1126]*y[IDX_HSII];
    data[5093] = 0.0 - k[199]*y[IDX_SI] - k[1117]*y[IDX_SI];
    data[5094] = 0.0 + k[205]*y[IDX_SII] - k[1951]*y[IDX_SI] -
        k[1952]*y[IDX_SI];
    data[5095] = 0.0 - k[1151]*y[IDX_SI];
    data[5096] = 0.0 + k[1814]*y[IDX_NI];
    data[5097] = 0.0 + k[546]*y[IDX_EM] + k[1502]*y[IDX_OI];
    data[5098] = 0.0 + k[1215]*y[IDX_CSI] + k[1221]*y[IDX_H2CSI] +
        k[1260]*y[IDX_NSI] + k[1267]*y[IDX_OCSI] + k[1269]*y[IDX_S2I] +
        k[1273]*y[IDX_SOI] + k[1288]*y[IDX_SiSI];
    data[5099] = 0.0 + k[1170]*y[IDX_HSII];
    data[5100] = 0.0 - k[1174]*y[IDX_SI];
    data[5101] = 0.0 + k[418] + k[1596]*y[IDX_CI] + k[1758]*y[IDX_HI] +
        k[1789]*y[IDX_HSI] + k[1789]*y[IDX_HSI] + k[1817]*y[IDX_NI] +
        k[1899]*y[IDX_OI] - k[1953]*y[IDX_SI] + k[2043];
    data[5102] = 0.0 - k[347]*y[IDX_SI] + k[554]*y[IDX_EM] +
        k[851]*y[IDX_CHI] + k[1007]*y[IDX_H2OI] + k[1126]*y[IDX_HCNI] +
        k[1170]*y[IDX_HNCI] + k[1175]*y[IDX_H2SI] + k[1437]*y[IDX_NH3I] +
        k[2040];
    data[5103] = 0.0 + k[417] + k[2042];
    data[5104] = 0.0 + k[555]*y[IDX_EM];
    data[5105] = 0.0 + k[229]*y[IDX_SII];
    data[5106] = 0.0 + k[1809]*y[IDX_CSI] + k[1814]*y[IDX_HCSI] +
        k[1817]*y[IDX_HSI] + k[1824]*y[IDX_NSI] + k[1829]*y[IDX_S2I] +
        k[1831]*y[IDX_SOI];
    data[5107] = 0.0 - k[258]*y[IDX_SI];
    data[5108] = 0.0 - k[1324]*y[IDX_SI];
    data[5109] = 0.0 - k[1856]*y[IDX_SI] - k[1857]*y[IDX_SI];
    data[5110] = 0.0 - k[265]*y[IDX_SI] - k[1372]*y[IDX_SI] -
        k[1373]*y[IDX_SI];
    data[5111] = 0.0 - k[270]*y[IDX_SI] + k[1384]*y[IDX_H2SI] -
        k[1392]*y[IDX_SI] - k[1393]*y[IDX_SI];
    data[5112] = 0.0 + k[292]*y[IDX_SII] + k[1437]*y[IDX_HSII];
    data[5113] = 0.0 + k[303]*y[IDX_SII] - k[1861]*y[IDX_SI] -
        k[1862]*y[IDX_SI];
    data[5114] = 0.0 + k[434] + k[1260]*y[IDX_HeII] + k[1607]*y[IDX_CI] +
        k[1767]*y[IDX_HI] + k[1824]*y[IDX_NI] + k[1907]*y[IDX_OI] + k[2059];
    data[5115] = 0.0 + k[576]*y[IDX_EM] + k[1509]*y[IDX_OI];
    data[5116] = 0.0 + k[1496]*y[IDX_CSII] + k[1502]*y[IDX_HCSII] +
        k[1509]*y[IDX_NSII] + k[1883]*y[IDX_CSI] + k[1899]*y[IDX_HSI] +
        k[1907]*y[IDX_NSI] + k[1912]*y[IDX_OCSI] + k[1915]*y[IDX_S2I] +
        k[1917]*y[IDX_SOI];
    data[5117] = 0.0 - k[1865]*y[IDX_SI];
    data[5118] = 0.0 - k[323]*y[IDX_SI] - k[1484]*y[IDX_SI];
    data[5119] = 0.0 - k[1554]*y[IDX_SI];
    data[5120] = 0.0 + k[441] + k[1267]*y[IDX_HeII] + k[1912]*y[IDX_OI] +
        k[2067];
    data[5121] = 0.0 + k[581]*y[IDX_EM];
    data[5122] = 0.0 - k[1948]*y[IDX_SI];
    data[5123] = 0.0 - k[338]*y[IDX_SI] - k[1533]*y[IDX_SI] -
        k[1534]*y[IDX_SI];
    data[5124] = 0.0 - k[35]*y[IDX_C2II] - k[41]*y[IDX_C2HII] -
        k[59]*y[IDX_CHII] - k[99]*y[IDX_CNII] - k[106]*y[IDX_COII] -
        k[140]*y[IDX_HII] - k[173]*y[IDX_H2COII] - k[185]*y[IDX_H2OII] -
        k[199]*y[IDX_HCNII] - k[258]*y[IDX_N2II] - k[265]*y[IDX_NHII] -
        k[270]*y[IDX_NH2II] - k[323]*y[IDX_O2II] - k[338]*y[IDX_OHII] -
        k[345]*y[IDX_CII] - k[346]*y[IDX_H2SII] - k[347]*y[IDX_HSII] - k[444] -
        k[650]*y[IDX_C2II] - k[739]*y[IDX_CHII] - k[740]*y[IDX_CHII] -
        k[753]*y[IDX_CH2II] - k[787]*y[IDX_CH3II] - k[835]*y[IDX_CH5II] -
        k[968]*y[IDX_H2COII] - k[987]*y[IDX_H2OII] - k[988]*y[IDX_H2OII] -
        k[1068]*y[IDX_H3II] - k[1117]*y[IDX_HCNII] - k[1151]*y[IDX_HCOII] -
        k[1174]*y[IDX_HNOII] - k[1324]*y[IDX_N2HII] - k[1372]*y[IDX_NHII] -
        k[1373]*y[IDX_NHII] - k[1392]*y[IDX_NH2II] - k[1393]*y[IDX_NH2II] -
        k[1484]*y[IDX_O2II] - k[1533]*y[IDX_OHII] - k[1534]*y[IDX_OHII] -
        k[1553]*y[IDX_H3SII] - k[1554]*y[IDX_O2HII] - k[1555]*y[IDX_SiOII] -
        k[1568]*y[IDX_SiH2II] - k[1573]*y[IDX_C2I] - k[1644]*y[IDX_CH2I] -
        k[1645]*y[IDX_CH2I] - k[1669]*y[IDX_CH3I] - k[1673]*y[IDX_CH4I] -
        k[1697]*y[IDX_CHI] - k[1698]*y[IDX_CHI] - k[1714]*y[IDX_CNI] -
        k[1735]*y[IDX_H2I] - k[1856]*y[IDX_NHI] - k[1857]*y[IDX_NHI] -
        k[1861]*y[IDX_NOI] - k[1862]*y[IDX_NOI] - k[1865]*y[IDX_O2I] -
        k[1948]*y[IDX_OHI] - k[1951]*y[IDX_HCOI] - k[1952]*y[IDX_HCOI] -
        k[1953]*y[IDX_HSI] - k[1954]*y[IDX_SO2I] - k[1955]*y[IDX_SOI] - k[2073]
        - k[2099]*y[IDX_CII] - k[2106]*y[IDX_CI] - k[2261];
    data[5125] = 0.0 + k[205]*y[IDX_HCOI] + k[229]*y[IDX_MgI] +
        k[292]*y[IDX_NH3I] + k[303]*y[IDX_NOI] + k[343]*y[IDX_SiCI] +
        k[344]*y[IDX_SiSI] + k[354]*y[IDX_SiI] + k[355]*y[IDX_SiHI] +
        k[2143]*y[IDX_EM];
    data[5126] = 0.0 + k[443] + k[443] + k[1269]*y[IDX_HeII] +
        k[1613]*y[IDX_CI] + k[1777]*y[IDX_HI] + k[1829]*y[IDX_NI] +
        k[1915]*y[IDX_OI] + k[2072] + k[2072];
    data[5127] = 0.0 + k[583]*y[IDX_EM] + k[583]*y[IDX_EM];
    data[5128] = 0.0 + k[354]*y[IDX_SII];
    data[5129] = 0.0 + k[343]*y[IDX_SII];
    data[5130] = 0.0 + k[355]*y[IDX_SII];
    data[5131] = 0.0 - k[1568]*y[IDX_SI];
    data[5132] = 0.0 - k[1555]*y[IDX_SI];
    data[5133] = 0.0 + k[344]*y[IDX_SII] + k[457] + k[646]*y[IDX_CII] +
        k[1288]*y[IDX_HeII] + k[2095];
    data[5134] = 0.0 + k[605]*y[IDX_EM];
    data[5135] = 0.0 + k[446] + k[641]*y[IDX_CII] + k[1273]*y[IDX_HeII] +
        k[1616]*y[IDX_CI] + k[1779]*y[IDX_HI] + k[1831]*y[IDX_NI] +
        k[1917]*y[IDX_OI] - k[1955]*y[IDX_SI] + k[2075];
    data[5136] = 0.0 + k[584]*y[IDX_EM];
    data[5137] = 0.0 - k[1954]*y[IDX_SI];
    data[5138] = 0.0 + k[585]*y[IDX_EM];
    data[5139] = 0.0 - k[2105]*y[IDX_SII];
    data[5140] = 0.0 + k[345]*y[IDX_SI] + k[640]*y[IDX_SOI];
    data[5141] = 0.0 - k[658]*y[IDX_SII];
    data[5142] = 0.0 + k[35]*y[IDX_SI];
    data[5143] = 0.0 + k[41]*y[IDX_SI];
    data[5144] = 0.0 - k[676]*y[IDX_SII];
    data[5145] = 0.0 - k[861]*y[IDX_SII];
    data[5146] = 0.0 + k[59]*y[IDX_SI];
    data[5147] = 0.0 - k[772]*y[IDX_SII];
    data[5148] = 0.0 - k[790]*y[IDX_SII];
    data[5149] = 0.0 - k[821]*y[IDX_SII] - k[822]*y[IDX_SII];
    data[5150] = 0.0 + k[99]*y[IDX_SI];
    data[5151] = 0.0 + k[106]*y[IDX_SI];
    data[5152] = 0.0 + k[1214]*y[IDX_HeII];
    data[5153] = 0.0 + k[2005];
    data[5154] = 0.0 - k[2143]*y[IDX_SII];
    data[5155] = 0.0 + k[1105]*y[IDX_HSII];
    data[5156] = 0.0 + k[140]*y[IDX_SI] + k[895]*y[IDX_H2SI] +
        k[902]*y[IDX_HSI];
    data[5157] = 0.0 - k[961]*y[IDX_SII] - k[2117]*y[IDX_SII];
    data[5158] = 0.0 + k[924]*y[IDX_H2SI];
    data[5159] = 0.0 - k[974]*y[IDX_SII] - k[975]*y[IDX_SII];
    data[5160] = 0.0 + k[173]*y[IDX_SI];
    data[5161] = 0.0 + k[1220]*y[IDX_HeII];
    data[5162] = 0.0 + k[185]*y[IDX_SI];
    data[5163] = 0.0 + k[895]*y[IDX_HII] + k[924]*y[IDX_H2II] +
        k[1227]*y[IDX_HeII] + k[1302]*y[IDX_NII] + k[1316]*y[IDX_N2II] +
        k[1473]*y[IDX_OII] - k[1550]*y[IDX_SII] - k[1551]*y[IDX_SII];
    data[5164] = 0.0 + k[346]*y[IDX_SI];
    data[5165] = 0.0 + k[199]*y[IDX_SI];
    data[5166] = 0.0 - k[205]*y[IDX_SII] - k[1163]*y[IDX_SII];
    data[5167] = 0.0 + k[1214]*y[IDX_CSI] + k[1220]*y[IDX_H2CSI] +
        k[1227]*y[IDX_H2SI] + k[1247]*y[IDX_HS2I] + k[1249]*y[IDX_HSI] +
        k[1259]*y[IDX_NSI] + k[1266]*y[IDX_OCSI] + k[1269]*y[IDX_S2I] +
        k[1270]*y[IDX_SO2I] + k[1272]*y[IDX_SOI] + k[1287]*y[IDX_SiSI];
    data[5168] = 0.0 + k[902]*y[IDX_HII] + k[1249]*y[IDX_HeII];
    data[5169] = 0.0 + k[347]*y[IDX_SI] + k[1105]*y[IDX_HI] +
        k[1503]*y[IDX_OI] + k[2039];
    data[5170] = 0.0 + k[1247]*y[IDX_HeII];
    data[5171] = 0.0 - k[229]*y[IDX_SII];
    data[5172] = 0.0 + k[1302]*y[IDX_H2SI] + k[1313]*y[IDX_OCSI];
    data[5173] = 0.0 + k[258]*y[IDX_SI] + k[1316]*y[IDX_H2SI] +
        k[1318]*y[IDX_OCSI];
    data[5174] = 0.0 - k[1461]*y[IDX_SII];
    data[5175] = 0.0 + k[265]*y[IDX_SI];
    data[5176] = 0.0 + k[270]*y[IDX_SI];
    data[5177] = 0.0 - k[292]*y[IDX_SII];
    data[5178] = 0.0 - k[303]*y[IDX_SII];
    data[5179] = 0.0 + k[1259]*y[IDX_HeII];
    data[5180] = 0.0 + k[1503]*y[IDX_HSII];
    data[5181] = 0.0 + k[1473]*y[IDX_H2SI] + k[1479]*y[IDX_OCSI];
    data[5182] = 0.0 - k[1486]*y[IDX_SII];
    data[5183] = 0.0 + k[323]*y[IDX_SI];
    data[5184] = 0.0 + k[1266]*y[IDX_HeII] + k[1313]*y[IDX_NII] +
        k[1318]*y[IDX_N2II] + k[1479]*y[IDX_OII] - k[1552]*y[IDX_SII];
    data[5185] = 0.0 - k[1548]*y[IDX_SII];
    data[5186] = 0.0 + k[338]*y[IDX_SI];
    data[5187] = 0.0 + k[35]*y[IDX_C2II] + k[41]*y[IDX_C2HII] +
        k[59]*y[IDX_CHII] + k[99]*y[IDX_CNII] + k[106]*y[IDX_COII] +
        k[140]*y[IDX_HII] + k[173]*y[IDX_H2COII] + k[185]*y[IDX_H2OII] +
        k[199]*y[IDX_HCNII] + k[258]*y[IDX_N2II] + k[265]*y[IDX_NHII] +
        k[270]*y[IDX_NH2II] + k[323]*y[IDX_O2II] + k[338]*y[IDX_OHII] +
        k[345]*y[IDX_CII] + k[346]*y[IDX_H2SII] + k[347]*y[IDX_HSII] + k[444] +
        k[2073];
    data[5188] = 0.0 - k[205]*y[IDX_HCOI] - k[229]*y[IDX_MgI] -
        k[292]*y[IDX_NH3I] - k[303]*y[IDX_NOI] - k[343]*y[IDX_SiCI] -
        k[344]*y[IDX_SiSI] - k[354]*y[IDX_SiI] - k[355]*y[IDX_SiHI] -
        k[658]*y[IDX_C2I] - k[676]*y[IDX_C2H4I] - k[772]*y[IDX_CH2I] -
        k[790]*y[IDX_CH3I] - k[821]*y[IDX_CH4I] - k[822]*y[IDX_CH4I] -
        k[861]*y[IDX_CHI] - k[961]*y[IDX_H2I] - k[974]*y[IDX_H2COI] -
        k[975]*y[IDX_H2COI] - k[1163]*y[IDX_HCOI] - k[1461]*y[IDX_NHI] -
        k[1486]*y[IDX_O2I] - k[1548]*y[IDX_OHI] - k[1550]*y[IDX_H2SI] -
        k[1551]*y[IDX_H2SI] - k[1552]*y[IDX_OCSI] - k[1569]*y[IDX_SiHI] -
        k[2105]*y[IDX_CI] - k[2117]*y[IDX_H2I] - k[2143]*y[IDX_EM] - k[2264];
    data[5189] = 0.0 + k[1269]*y[IDX_HeII];
    data[5190] = 0.0 - k[354]*y[IDX_SII];
    data[5191] = 0.0 - k[343]*y[IDX_SII];
    data[5192] = 0.0 - k[355]*y[IDX_SII] - k[1569]*y[IDX_SII];
    data[5193] = 0.0 - k[344]*y[IDX_SII] + k[1287]*y[IDX_HeII];
    data[5194] = 0.0 + k[640]*y[IDX_CII] + k[1272]*y[IDX_HeII];
    data[5195] = 0.0 + k[1270]*y[IDX_HeII];
    data[5196] = 0.0 - k[1613]*y[IDX_S2I];
    data[5197] = 0.0 + k[556]*y[IDX_HS2II];
    data[5198] = 0.0 - k[1777]*y[IDX_S2I];
    data[5199] = 0.0 - k[139]*y[IDX_S2I];
    data[5200] = 0.0 + k[1019]*y[IDX_HS2II];
    data[5201] = 0.0 - k[1067]*y[IDX_S2I];
    data[5202] = 0.0 - k[1092]*y[IDX_S2I];
    data[5203] = 0.0 - k[1150]*y[IDX_S2I];
    data[5204] = 0.0 - k[1269]*y[IDX_S2I];
    data[5205] = 0.0 + k[1953]*y[IDX_SI];
    data[5206] = 0.0 + k[556]*y[IDX_EM] + k[1019]*y[IDX_H2SI];
    data[5207] = 0.0 - k[1829]*y[IDX_S2I];
    data[5208] = 0.0 + k[304]*y[IDX_S2II];
    data[5209] = 0.0 - k[1915]*y[IDX_S2I];
    data[5210] = 0.0 + k[1953]*y[IDX_HSI] + k[1955]*y[IDX_SOI];
    data[5211] = 0.0 - k[139]*y[IDX_HII] - k[443] - k[1067]*y[IDX_H3II] -
        k[1092]*y[IDX_H3OII] - k[1150]*y[IDX_HCOII] - k[1269]*y[IDX_HeII] -
        k[1613]*y[IDX_CI] - k[1777]*y[IDX_HI] - k[1829]*y[IDX_NI] -
        k[1915]*y[IDX_OI] - k[2071] - k[2072] - k[2201];
    data[5212] = 0.0 + k[304]*y[IDX_NOI];
    data[5213] = 0.0 + k[1955]*y[IDX_SI];
    data[5214] = 0.0 - k[793]*y[IDX_S2II];
    data[5215] = 0.0 - k[583]*y[IDX_S2II];
    data[5216] = 0.0 + k[139]*y[IDX_S2I];
    data[5217] = 0.0 + k[1021]*y[IDX_SOII] + k[1551]*y[IDX_SII];
    data[5218] = 0.0 + k[1248]*y[IDX_HS2I];
    data[5219] = 0.0 + k[1248]*y[IDX_HeII];
    data[5220] = 0.0 - k[304]*y[IDX_S2II];
    data[5221] = 0.0 + k[1552]*y[IDX_SII] + k[1562]*y[IDX_SOII];
    data[5222] = 0.0 + k[1551]*y[IDX_H2SI] + k[1552]*y[IDX_OCSI];
    data[5223] = 0.0 + k[139]*y[IDX_HII] + k[2071];
    data[5224] = 0.0 - k[304]*y[IDX_NOI] - k[583]*y[IDX_EM] -
        k[793]*y[IDX_CH3OHI] - k[2281];
    data[5225] = 0.0 + k[1021]*y[IDX_H2SI] + k[1562]*y[IDX_OCSI];
    data[5226] = 0.0 - k[26]*y[IDX_SiI];
    data[5227] = 0.0 - k[1575]*y[IDX_SiI];
    data[5228] = 0.0 - k[670]*y[IDX_SiI];
    data[5229] = 0.0 + k[863]*y[IDX_SiHII] + k[864]*y[IDX_SiOII];
    data[5230] = 0.0 - k[60]*y[IDX_SiI];
    data[5231] = 0.0 - k[1957]*y[IDX_SiI];
    data[5232] = 0.0 - k[1956]*y[IDX_SiI];
    data[5233] = 0.0 - k[348]*y[IDX_SiI];
    data[5234] = 0.0 + k[561]*y[IDX_HSiSII] + k[587]*y[IDX_SiCII] +
        k[588]*y[IDX_SiC2II] + k[592]*y[IDX_SiHII] + k[593]*y[IDX_SiH2II] +
        k[594]*y[IDX_SiH2II] + k[602]*y[IDX_SiOII] + k[603]*y[IDX_SiOHII] +
        k[605]*y[IDX_SiSII] + k[2144]*y[IDX_SiII];
    data[5235] = 0.0 - k[143]*y[IDX_SiI];
    data[5236] = 0.0 - k[349]*y[IDX_SiI];
    data[5237] = 0.0 + k[1014]*y[IDX_SiHII];
    data[5238] = 0.0 - k[186]*y[IDX_SiI];
    data[5239] = 0.0 - k[350]*y[IDX_SiI];
    data[5240] = 0.0 - k[1071]*y[IDX_SiI];
    data[5241] = 0.0 - k[1093]*y[IDX_SiI];
    data[5242] = 0.0 - k[1566]*y[IDX_SiI];
    data[5243] = 0.0 - k[219]*y[IDX_SiI] + k[1277]*y[IDX_SiCI] +
        k[1286]*y[IDX_SiOI] + k[1287]*y[IDX_SiSI];
    data[5244] = 0.0 - k[351]*y[IDX_SiI];
    data[5245] = 0.0 + k[561]*y[IDX_EM];
    data[5246] = 0.0 + k[231]*y[IDX_SiII];
    data[5247] = 0.0 + k[1343]*y[IDX_SiOII] + k[1832]*y[IDX_SiCI];
    data[5248] = 0.0 + k[1442]*y[IDX_SiHII];
    data[5249] = 0.0 - k[281]*y[IDX_SiI];
    data[5250] = 0.0 - k[1958]*y[IDX_SiI];
    data[5251] = 0.0 - k[352]*y[IDX_SiI];
    data[5252] = 0.0 + k[1920]*y[IDX_SiCI] - k[2131]*y[IDX_SiI];
    data[5253] = 0.0 - k[1959]*y[IDX_SiI];
    data[5254] = 0.0 - k[353]*y[IDX_SiI];
    data[5255] = 0.0 - k[1950]*y[IDX_SiI];
    data[5256] = 0.0 - k[1535]*y[IDX_SiI];
    data[5257] = 0.0 - k[354]*y[IDX_SiI];
    data[5258] = 0.0 - k[26]*y[IDX_CII] - k[60]*y[IDX_CHII] -
        k[143]*y[IDX_HII] - k[186]*y[IDX_H2OII] - k[219]*y[IDX_HeII] -
        k[281]*y[IDX_NH3II] - k[348]*y[IDX_CSII] - k[349]*y[IDX_H2COII] -
        k[350]*y[IDX_H2SII] - k[351]*y[IDX_HSII] - k[352]*y[IDX_NOII] -
        k[353]*y[IDX_O2II] - k[354]*y[IDX_SII] - k[448] - k[670]*y[IDX_C2H2II] -
        k[1071]*y[IDX_H3II] - k[1093]*y[IDX_H3OII] - k[1535]*y[IDX_OHII] -
        k[1566]*y[IDX_HCOII] - k[1575]*y[IDX_C2H2I] - k[1950]*y[IDX_OHI] -
        k[1956]*y[IDX_CO2I] - k[1957]*y[IDX_COI] - k[1958]*y[IDX_NOI] -
        k[1959]*y[IDX_O2I] - k[2077] - k[2131]*y[IDX_OI] - k[2170];
    data[5259] = 0.0 + k[231]*y[IDX_MgI] + k[2144]*y[IDX_EM];
    data[5260] = 0.0 + k[451] + k[1277]*y[IDX_HeII] + k[1832]*y[IDX_NI] +
        k[1920]*y[IDX_OI] + k[2081];
    data[5261] = 0.0 + k[587]*y[IDX_EM];
    data[5262] = 0.0 + k[2078];
    data[5263] = 0.0 + k[588]*y[IDX_EM];
    data[5264] = 0.0 + k[455] + k[2091];
    data[5265] = 0.0 + k[592]*y[IDX_EM] + k[863]*y[IDX_CHI] +
        k[1014]*y[IDX_H2OI] + k[1442]*y[IDX_NH3I];
    data[5266] = 0.0 + k[593]*y[IDX_EM] + k[594]*y[IDX_EM];
    data[5267] = 0.0 + k[456] + k[1286]*y[IDX_HeII] + k[2093];
    data[5268] = 0.0 + k[602]*y[IDX_EM] + k[864]*y[IDX_CHI] +
        k[1343]*y[IDX_NI];
    data[5269] = 0.0 + k[603]*y[IDX_EM];
    data[5270] = 0.0 + k[457] + k[1287]*y[IDX_HeII] + k[2095];
    data[5271] = 0.0 + k[605]*y[IDX_EM];
    data[5272] = 0.0 + k[704]*y[IDX_SiOII];
    data[5273] = 0.0 + k[26]*y[IDX_SiI] + k[642]*y[IDX_SiCI] +
        k[645]*y[IDX_SiOI];
    data[5274] = 0.0 - k[684]*y[IDX_SiII];
    data[5275] = 0.0 + k[671]*y[IDX_SiH4I];
    data[5276] = 0.0 - k[1563]*y[IDX_SiII];
    data[5277] = 0.0 - k[862]*y[IDX_SiII];
    data[5278] = 0.0 + k[60]*y[IDX_SiI];
    data[5279] = 0.0 + k[773]*y[IDX_SiOII];
    data[5280] = 0.0 - k[1564]*y[IDX_SiII];
    data[5281] = 0.0 + k[881]*y[IDX_SiOII];
    data[5282] = 0.0 + k[348]*y[IDX_SiI];
    data[5283] = 0.0 - k[2144]*y[IDX_SiII];
    data[5284] = 0.0 + k[1108]*y[IDX_SiHII] + k[1109]*y[IDX_SiSII] -
        k[2126]*y[IDX_SiII];
    data[5285] = 0.0 + k[143]*y[IDX_SiI] + k[908]*y[IDX_SiHI];
    data[5286] = 0.0 - k[2118]*y[IDX_SiII];
    data[5287] = 0.0 + k[349]*y[IDX_SiI];
    data[5288] = 0.0 - k[1013]*y[IDX_SiII];
    data[5289] = 0.0 + k[186]*y[IDX_SiI];
    data[5290] = 0.0 + k[350]*y[IDX_SiI];
    data[5291] = 0.0 + k[219]*y[IDX_SiI] + k[1274]*y[IDX_SiC2I] +
        k[1276]*y[IDX_SiCI] + k[1278]*y[IDX_SiH2I] + k[1282]*y[IDX_SiH4I] +
        k[1284]*y[IDX_SiHI] + k[1285]*y[IDX_SiOI] + k[1288]*y[IDX_SiSI];
    data[5292] = 0.0 + k[351]*y[IDX_SiI];
    data[5293] = 0.0 - k[231]*y[IDX_SiII];
    data[5294] = 0.0 + k[1342]*y[IDX_SiCII] + k[1344]*y[IDX_SiOII];
    data[5295] = 0.0 + k[281]*y[IDX_SiI];
    data[5296] = 0.0 + k[352]*y[IDX_SiI];
    data[5297] = 0.0 + k[1516]*y[IDX_SiOII] - k[2130]*y[IDX_SiII];
    data[5298] = 0.0 + k[353]*y[IDX_SiI];
    data[5299] = 0.0 - k[1565]*y[IDX_SiII];
    data[5300] = 0.0 - k[1549]*y[IDX_SiII];
    data[5301] = 0.0 + k[1555]*y[IDX_SiOII];
    data[5302] = 0.0 + k[354]*y[IDX_SiI];
    data[5303] = 0.0 + k[26]*y[IDX_CII] + k[60]*y[IDX_CHII] +
        k[143]*y[IDX_HII] + k[186]*y[IDX_H2OII] + k[219]*y[IDX_HeII] +
        k[281]*y[IDX_NH3II] + k[348]*y[IDX_CSII] + k[349]*y[IDX_H2COII] +
        k[350]*y[IDX_H2SII] + k[351]*y[IDX_HSII] + k[352]*y[IDX_NOII] +
        k[353]*y[IDX_O2II] + k[354]*y[IDX_SII] + k[448] + k[2077];
    data[5304] = 0.0 - k[231]*y[IDX_MgI] - k[684]*y[IDX_C2HI] -
        k[862]*y[IDX_CHI] - k[1013]*y[IDX_H2OI] - k[1549]*y[IDX_OHI] -
        k[1563]*y[IDX_C2H5OHI] - k[1564]*y[IDX_CH3OHI] - k[1565]*y[IDX_OCSI] -
        k[2118]*y[IDX_H2I] - k[2126]*y[IDX_HI] - k[2130]*y[IDX_OI] -
        k[2144]*y[IDX_EM] - k[2173];
    data[5305] = 0.0 + k[642]*y[IDX_CII] + k[1276]*y[IDX_HeII];
    data[5306] = 0.0 + k[1342]*y[IDX_NI];
    data[5307] = 0.0 + k[1274]*y[IDX_HeII];
    data[5308] = 0.0 + k[908]*y[IDX_HII] + k[1284]*y[IDX_HeII];
    data[5309] = 0.0 + k[1108]*y[IDX_HI] + k[2082];
    data[5310] = 0.0 + k[1278]*y[IDX_HeII];
    data[5311] = 0.0 + k[671]*y[IDX_C2H2II] + k[1282]*y[IDX_HeII];
    data[5312] = 0.0 + k[645]*y[IDX_CII] + k[1285]*y[IDX_HeII];
    data[5313] = 0.0 + k[704]*y[IDX_CI] + k[773]*y[IDX_CH2I] +
        k[881]*y[IDX_COI] + k[1344]*y[IDX_NI] + k[1516]*y[IDX_OI] +
        k[1555]*y[IDX_SI] + k[2092];
    data[5314] = 0.0 + k[1288]*y[IDX_HeII];
    data[5315] = 0.0 + k[1109]*y[IDX_HI];
    data[5316] = 0.0 + k[2407] + k[2408] + k[2409] + k[2410];
    data[5317] = 0.0 + k[1617]*y[IDX_SiHI];
    data[5318] = 0.0 - k[29]*y[IDX_SiCI] - k[642]*y[IDX_SiCI];
    data[5319] = 0.0 + k[589]*y[IDX_SiC2II] + k[591]*y[IDX_SiC3II];
    data[5320] = 0.0 - k[146]*y[IDX_SiCI];
    data[5321] = 0.0 - k[1276]*y[IDX_SiCI] - k[1277]*y[IDX_SiCI];
    data[5322] = 0.0 - k[1832]*y[IDX_SiCI];
    data[5323] = 0.0 + k[1918]*y[IDX_SiC2I] - k[1920]*y[IDX_SiCI] -
        k[1921]*y[IDX_SiCI];
    data[5324] = 0.0 - k[343]*y[IDX_SiCI];
    data[5325] = 0.0 - k[29]*y[IDX_CII] - k[146]*y[IDX_HII] -
        k[343]*y[IDX_SII] - k[451] - k[642]*y[IDX_CII] - k[1276]*y[IDX_HeII] -
        k[1277]*y[IDX_HeII] - k[1832]*y[IDX_NI] - k[1920]*y[IDX_OI] -
        k[1921]*y[IDX_OI] - k[2081] - k[2181];
    data[5326] = 0.0 + k[449] + k[1918]*y[IDX_OI];
    data[5327] = 0.0 + k[589]*y[IDX_EM];
    data[5328] = 0.0 + k[2079];
    data[5329] = 0.0 + k[591]*y[IDX_EM];
    data[5330] = 0.0 + k[1617]*y[IDX_CI];
    data[5331] = 0.0 + k[703]*y[IDX_SiHII];
    data[5332] = 0.0 + k[29]*y[IDX_SiCI] + k[643]*y[IDX_SiH2I] +
        k[644]*y[IDX_SiHI] + k[646]*y[IDX_SiSI];
    data[5333] = 0.0 + k[659]*y[IDX_SiOII];
    data[5334] = 0.0 + k[862]*y[IDX_SiII];
    data[5335] = 0.0 - k[587]*y[IDX_SiCII];
    data[5336] = 0.0 + k[146]*y[IDX_SiCI];
    data[5337] = 0.0 - k[1342]*y[IDX_SiCII];
    data[5338] = 0.0 - k[1512]*y[IDX_SiCII];
    data[5339] = 0.0 + k[343]*y[IDX_SiCI];
    data[5340] = 0.0 + k[862]*y[IDX_CHI];
    data[5341] = 0.0 + k[29]*y[IDX_CII] + k[146]*y[IDX_HII] +
        k[343]*y[IDX_SII];
    data[5342] = 0.0 - k[587]*y[IDX_EM] - k[1342]*y[IDX_NI] -
        k[1512]*y[IDX_OI] - k[2183];
    data[5343] = 0.0 + k[644]*y[IDX_CII];
    data[5344] = 0.0 + k[703]*y[IDX_CI];
    data[5345] = 0.0 + k[643]*y[IDX_CII];
    data[5346] = 0.0 + k[659]*y[IDX_C2I];
    data[5347] = 0.0 + k[646]*y[IDX_CII];
    data[5348] = 0.0 + k[2475] + k[2476] + k[2477] + k[2478];
    data[5349] = 0.0 - k[27]*y[IDX_SiC2I];
    data[5350] = 0.0 + k[1575]*y[IDX_SiI];
    data[5351] = 0.0 + k[590]*y[IDX_SiC3II];
    data[5352] = 0.0 - k[144]*y[IDX_SiC2I];
    data[5353] = 0.0 - k[1274]*y[IDX_SiC2I];
    data[5354] = 0.0 - k[1918]*y[IDX_SiC2I] + k[1919]*y[IDX_SiC3I];
    data[5355] = 0.0 + k[1575]*y[IDX_C2H2I];
    data[5356] = 0.0 - k[27]*y[IDX_CII] - k[144]*y[IDX_HII] - k[449] -
        k[1274]*y[IDX_HeII] - k[1918]*y[IDX_OI] - k[2078] - k[2182];
    data[5357] = 0.0 + k[450] + k[1919]*y[IDX_OI] + k[2080];
    data[5358] = 0.0 + k[590]*y[IDX_EM];
    data[5359] = 0.0 + k[27]*y[IDX_SiC2I];
    data[5360] = 0.0 + k[684]*y[IDX_SiII];
    data[5361] = 0.0 + k[670]*y[IDX_SiI];
    data[5362] = 0.0 - k[588]*y[IDX_SiC2II] - k[589]*y[IDX_SiC2II];
    data[5363] = 0.0 + k[144]*y[IDX_SiC2I];
    data[5364] = 0.0 + k[1275]*y[IDX_SiC3I];
    data[5365] = 0.0 + k[670]*y[IDX_C2H2II];
    data[5366] = 0.0 + k[684]*y[IDX_C2HI];
    data[5367] = 0.0 + k[27]*y[IDX_CII] + k[144]*y[IDX_HII];
    data[5368] = 0.0 - k[588]*y[IDX_EM] - k[589]*y[IDX_EM] - k[2184];
    data[5369] = 0.0 + k[1275]*y[IDX_HeII];
    data[5370] = 0.0 + k[2495] + k[2496] + k[2497] + k[2498];
    data[5371] = 0.0 - k[28]*y[IDX_SiC3I];
    data[5372] = 0.0 - k[145]*y[IDX_SiC3I];
    data[5373] = 0.0 - k[1275]*y[IDX_SiC3I];
    data[5374] = 0.0 - k[1919]*y[IDX_SiC3I];
    data[5375] = 0.0 - k[28]*y[IDX_CII] - k[145]*y[IDX_HII] - k[450] -
        k[1275]*y[IDX_HeII] - k[1919]*y[IDX_OI] - k[2079] - k[2080] - k[2185];
    data[5376] = 0.0 + k[28]*y[IDX_SiC3I];
    data[5377] = 0.0 - k[590]*y[IDX_SiC3II] - k[591]*y[IDX_SiC3II];
    data[5378] = 0.0 + k[145]*y[IDX_SiC3I];
    data[5379] = 0.0 + k[28]*y[IDX_CII] + k[145]*y[IDX_HII];
    data[5380] = 0.0 - k[590]*y[IDX_EM] - k[591]*y[IDX_EM] - k[2186];
    data[5381] = 0.0 - k[1617]*y[IDX_SiHI];
    data[5382] = 0.0 - k[644]*y[IDX_SiHI];
    data[5383] = 0.0 + k[595]*y[IDX_SiH2II] + k[597]*y[IDX_SiH3II];
    data[5384] = 0.0 - k[150]*y[IDX_SiHI] - k[908]*y[IDX_SiHI];
    data[5385] = 0.0 - k[1075]*y[IDX_SiHI];
    data[5386] = 0.0 - k[1095]*y[IDX_SiHI];
    data[5387] = 0.0 - k[1155]*y[IDX_SiHI];
    data[5388] = 0.0 - k[1284]*y[IDX_SiHI];
    data[5389] = 0.0 - k[1926]*y[IDX_SiHI];
    data[5390] = 0.0 - k[1536]*y[IDX_SiHI];
    data[5391] = 0.0 - k[355]*y[IDX_SiHI] - k[1569]*y[IDX_SiHI];
    data[5392] = 0.0 - k[150]*y[IDX_HII] - k[355]*y[IDX_SII] - k[455] -
        k[644]*y[IDX_CII] - k[908]*y[IDX_HII] - k[1075]*y[IDX_H3II] -
        k[1095]*y[IDX_H3OII] - k[1155]*y[IDX_HCOII] - k[1284]*y[IDX_HeII] -
        k[1536]*y[IDX_OHII] - k[1569]*y[IDX_SII] - k[1617]*y[IDX_CI] -
        k[1926]*y[IDX_OI] - k[2091] - k[2172];
    data[5393] = 0.0 + k[452] + k[2084];
    data[5394] = 0.0 + k[595]*y[IDX_EM];
    data[5395] = 0.0 + k[2087];
    data[5396] = 0.0 + k[597]*y[IDX_EM];
    data[5397] = 0.0 + k[2090];
    data[5398] = 0.0 - k[703]*y[IDX_SiHII];
    data[5399] = 0.0 + k[672]*y[IDX_SiH4I];
    data[5400] = 0.0 - k[863]*y[IDX_SiHII];
    data[5401] = 0.0 - k[592]*y[IDX_SiHII];
    data[5402] = 0.0 - k[1108]*y[IDX_SiHII] + k[2126]*y[IDX_SiII];
    data[5403] = 0.0 + k[150]*y[IDX_SiHI] + k[905]*y[IDX_SiH2I];
    data[5404] = 0.0 - k[2119]*y[IDX_SiHII];
    data[5405] = 0.0 - k[1014]*y[IDX_SiHII];
    data[5406] = 0.0 + k[1071]*y[IDX_SiI];
    data[5407] = 0.0 + k[1093]*y[IDX_SiI];
    data[5408] = 0.0 + k[1566]*y[IDX_SiI];
    data[5409] = 0.0 + k[1279]*y[IDX_SiH2I] + k[1280]*y[IDX_SiH3I] +
        k[1283]*y[IDX_SiH4I];
    data[5410] = 0.0 - k[1442]*y[IDX_SiHII];
    data[5411] = 0.0 - k[1513]*y[IDX_SiHII];
    data[5412] = 0.0 + k[1535]*y[IDX_SiI];
    data[5413] = 0.0 + k[355]*y[IDX_SiHI];
    data[5414] = 0.0 + k[1071]*y[IDX_H3II] + k[1093]*y[IDX_H3OII] +
        k[1535]*y[IDX_OHII] + k[1566]*y[IDX_HCOII];
    data[5415] = 0.0 + k[2126]*y[IDX_HI];
    data[5416] = 0.0 + k[150]*y[IDX_HII] + k[355]*y[IDX_SII];
    data[5417] = 0.0 - k[592]*y[IDX_EM] - k[703]*y[IDX_CI] -
        k[863]*y[IDX_CHI] - k[1014]*y[IDX_H2OI] - k[1108]*y[IDX_HI] -
        k[1442]*y[IDX_NH3I] - k[1513]*y[IDX_OI] - k[2082] - k[2119]*y[IDX_H2I] -
        k[2174];
    data[5418] = 0.0 + k[905]*y[IDX_HII] + k[1279]*y[IDX_HeII];
    data[5419] = 0.0 + k[1280]*y[IDX_HeII];
    data[5420] = 0.0 + k[672]*y[IDX_C2H2II] + k[1283]*y[IDX_HeII];
    data[5421] = 0.0 - k[30]*y[IDX_SiH2I] - k[643]*y[IDX_SiH2I];
    data[5422] = 0.0 + k[596]*y[IDX_SiH3II] + k[598]*y[IDX_SiH4II];
    data[5423] = 0.0 - k[147]*y[IDX_SiH2I] - k[905]*y[IDX_SiH2I];
    data[5424] = 0.0 - k[1072]*y[IDX_SiH2I];
    data[5425] = 0.0 - k[1094]*y[IDX_SiH2I];
    data[5426] = 0.0 - k[1153]*y[IDX_SiH2I];
    data[5427] = 0.0 - k[1278]*y[IDX_SiH2I] - k[1279]*y[IDX_SiH2I];
    data[5428] = 0.0 - k[1922]*y[IDX_SiH2I] - k[1923]*y[IDX_SiH2I];
    data[5429] = 0.0 - k[30]*y[IDX_CII] - k[147]*y[IDX_HII] - k[452] -
        k[643]*y[IDX_CII] - k[905]*y[IDX_HII] - k[1072]*y[IDX_H3II] -
        k[1094]*y[IDX_H3OII] - k[1153]*y[IDX_HCOII] - k[1278]*y[IDX_HeII] -
        k[1279]*y[IDX_HeII] - k[1922]*y[IDX_OI] - k[1923]*y[IDX_OI] - k[2083] -
        k[2084] - k[2175];
    data[5430] = 0.0 + k[453] + k[2085];
    data[5431] = 0.0 + k[596]*y[IDX_EM];
    data[5432] = 0.0 + k[454] + k[2088];
    data[5433] = 0.0 + k[598]*y[IDX_EM];
    data[5434] = 0.0 + k[30]*y[IDX_SiH2I];
    data[5435] = 0.0 + k[673]*y[IDX_SiH4I];
    data[5436] = 0.0 - k[593]*y[IDX_SiH2II] - k[594]*y[IDX_SiH2II] -
        k[595]*y[IDX_SiH2II];
    data[5437] = 0.0 + k[147]*y[IDX_SiH2I] + k[906]*y[IDX_SiH3I];
    data[5438] = 0.0 + k[2118]*y[IDX_SiII];
    data[5439] = 0.0 + k[1075]*y[IDX_SiHI];
    data[5440] = 0.0 + k[1095]*y[IDX_SiHI];
    data[5441] = 0.0 + k[1155]*y[IDX_SiHI];
    data[5442] = 0.0 + k[1281]*y[IDX_SiH3I];
    data[5443] = 0.0 - k[1514]*y[IDX_SiH2II];
    data[5444] = 0.0 - k[1567]*y[IDX_SiH2II];
    data[5445] = 0.0 + k[1536]*y[IDX_SiHI];
    data[5446] = 0.0 - k[1568]*y[IDX_SiH2II];
    data[5447] = 0.0 + k[2118]*y[IDX_H2I];
    data[5448] = 0.0 + k[1075]*y[IDX_H3II] + k[1095]*y[IDX_H3OII] +
        k[1155]*y[IDX_HCOII] + k[1536]*y[IDX_OHII];
    data[5449] = 0.0 + k[30]*y[IDX_CII] + k[147]*y[IDX_HII] + k[2083];
    data[5450] = 0.0 - k[593]*y[IDX_EM] - k[594]*y[IDX_EM] -
        k[595]*y[IDX_EM] - k[1514]*y[IDX_OI] - k[1567]*y[IDX_O2I] -
        k[1568]*y[IDX_SI] - k[2176];
    data[5451] = 0.0 + k[906]*y[IDX_HII] + k[1281]*y[IDX_HeII];
    data[5452] = 0.0 + k[673]*y[IDX_C2H2II];
    data[5453] = 0.0 - k[31]*y[IDX_SiH3I];
    data[5454] = 0.0 + k[1715]*y[IDX_SiH4I];
    data[5455] = 0.0 + k[880]*y[IDX_SiH4II];
    data[5456] = 0.0 + k[599]*y[IDX_SiH4II] + k[600]*y[IDX_SiH5II];
    data[5457] = 0.0 - k[148]*y[IDX_SiH3I] - k[906]*y[IDX_SiH3I];
    data[5458] = 0.0 + k[1015]*y[IDX_SiH4II];
    data[5459] = 0.0 - k[1073]*y[IDX_SiH3I];
    data[5460] = 0.0 - k[1280]*y[IDX_SiH3I] - k[1281]*y[IDX_SiH3I];
    data[5461] = 0.0 - k[1924]*y[IDX_SiH3I] + k[1925]*y[IDX_SiH4I];
    data[5462] = 0.0 - k[31]*y[IDX_CII] - k[148]*y[IDX_HII] - k[453] -
        k[906]*y[IDX_HII] - k[1073]*y[IDX_H3II] - k[1280]*y[IDX_HeII] -
        k[1281]*y[IDX_HeII] - k[1924]*y[IDX_OI] - k[2085] - k[2086] - k[2087] -
        k[2177];
    data[5463] = 0.0 + k[1715]*y[IDX_CNI] + k[1925]*y[IDX_OI] + k[2089];
    data[5464] = 0.0 + k[599]*y[IDX_EM] + k[880]*y[IDX_COI] +
        k[1015]*y[IDX_H2OI];
    data[5465] = 0.0 + k[600]*y[IDX_EM];
    data[5466] = 0.0 + k[31]*y[IDX_SiH3I];
    data[5467] = 0.0 + k[674]*y[IDX_SiH4I];
    data[5468] = 0.0 + k[789]*y[IDX_SiH4I];
    data[5469] = 0.0 + k[836]*y[IDX_SiH4I];
    data[5470] = 0.0 - k[596]*y[IDX_SiH3II] - k[597]*y[IDX_SiH3II];
    data[5471] = 0.0 + k[148]*y[IDX_SiH3I] + k[907]*y[IDX_SiH4I];
    data[5472] = 0.0 + k[2119]*y[IDX_SiHII] - k[2120]*y[IDX_SiH3II];
    data[5473] = 0.0 + k[1072]*y[IDX_SiH2I];
    data[5474] = 0.0 + k[1094]*y[IDX_SiH2I];
    data[5475] = 0.0 + k[1153]*y[IDX_SiH2I];
    data[5476] = 0.0 - k[1515]*y[IDX_SiH3II];
    data[5477] = 0.0 + k[2119]*y[IDX_H2I];
    data[5478] = 0.0 + k[1072]*y[IDX_H3II] + k[1094]*y[IDX_H3OII] +
        k[1153]*y[IDX_HCOII];
    data[5479] = 0.0 + k[31]*y[IDX_CII] + k[148]*y[IDX_HII] + k[2086];
    data[5480] = 0.0 - k[596]*y[IDX_EM] - k[597]*y[IDX_EM] -
        k[1515]*y[IDX_OI] - k[2120]*y[IDX_H2I] - k[2178];
    data[5481] = 0.0 + k[674]*y[IDX_C2H2II] + k[789]*y[IDX_CH3II] +
        k[836]*y[IDX_CH5II] + k[907]*y[IDX_HII];
    data[5482] = 0.0 + k[2383] + k[2384] + k[2385] + k[2386];
    data[5483] = 0.0 - k[671]*y[IDX_SiH4I] - k[672]*y[IDX_SiH4I] -
        k[673]*y[IDX_SiH4I] - k[674]*y[IDX_SiH4I];
    data[5484] = 0.0 - k[789]*y[IDX_SiH4I];
    data[5485] = 0.0 - k[836]*y[IDX_SiH4I];
    data[5486] = 0.0 - k[1715]*y[IDX_SiH4I];
    data[5487] = 0.0 + k[601]*y[IDX_SiH5II];
    data[5488] = 0.0 - k[149]*y[IDX_SiH4I] - k[907]*y[IDX_SiH4I];
    data[5489] = 0.0 + k[1016]*y[IDX_SiH5II];
    data[5490] = 0.0 - k[1074]*y[IDX_SiH4I];
    data[5491] = 0.0 - k[1154]*y[IDX_SiH4I];
    data[5492] = 0.0 - k[1282]*y[IDX_SiH4I] - k[1283]*y[IDX_SiH4I];
    data[5493] = 0.0 - k[1925]*y[IDX_SiH4I];
    data[5494] = 0.0 - k[149]*y[IDX_HII] - k[454] - k[671]*y[IDX_C2H2II] -
        k[672]*y[IDX_C2H2II] - k[673]*y[IDX_C2H2II] - k[674]*y[IDX_C2H2II] -
        k[789]*y[IDX_CH3II] - k[836]*y[IDX_CH5II] - k[907]*y[IDX_HII] -
        k[1074]*y[IDX_H3II] - k[1154]*y[IDX_HCOII] - k[1282]*y[IDX_HeII] -
        k[1283]*y[IDX_HeII] - k[1715]*y[IDX_CNI] - k[1925]*y[IDX_OI] - k[2088] -
        k[2089] - k[2090] - k[2179];
    data[5495] = 0.0 + k[601]*y[IDX_EM] + k[1016]*y[IDX_H2OI];
    data[5496] = 0.0 - k[880]*y[IDX_SiH4II];
    data[5497] = 0.0 - k[598]*y[IDX_SiH4II] - k[599]*y[IDX_SiH4II];
    data[5498] = 0.0 + k[149]*y[IDX_SiH4I];
    data[5499] = 0.0 - k[963]*y[IDX_SiH4II];
    data[5500] = 0.0 - k[1015]*y[IDX_SiH4II];
    data[5501] = 0.0 + k[1073]*y[IDX_SiH3I];
    data[5502] = 0.0 + k[1073]*y[IDX_H3II];
    data[5503] = 0.0 + k[149]*y[IDX_HII];
    data[5504] = 0.0 - k[598]*y[IDX_EM] - k[599]*y[IDX_EM] -
        k[880]*y[IDX_COI] - k[963]*y[IDX_H2I] - k[1015]*y[IDX_H2OI] - k[2180];
    data[5505] = 0.0 - k[600]*y[IDX_SiH5II] - k[601]*y[IDX_SiH5II];
    data[5506] = 0.0 + k[963]*y[IDX_SiH4II] + k[2120]*y[IDX_SiH3II];
    data[5507] = 0.0 - k[1016]*y[IDX_SiH5II];
    data[5508] = 0.0 + k[1074]*y[IDX_SiH4I];
    data[5509] = 0.0 + k[1154]*y[IDX_SiH4I];
    data[5510] = 0.0 + k[2120]*y[IDX_H2I];
    data[5511] = 0.0 + k[1074]*y[IDX_H3II] + k[1154]*y[IDX_HCOII];
    data[5512] = 0.0 + k[963]*y[IDX_H2I];
    data[5513] = 0.0 - k[600]*y[IDX_EM] - k[601]*y[IDX_EM] -
        k[1016]*y[IDX_H2OI] - k[2193];
    data[5514] = 0.0 + k[2427] + k[2428] + k[2429] + k[2430];
    data[5515] = 0.0 - k[645]*y[IDX_SiOI];
    data[5516] = 0.0 + k[1957]*y[IDX_SiI];
    data[5517] = 0.0 + k[1956]*y[IDX_SiI];
    data[5518] = 0.0 + k[604]*y[IDX_SiOHII];
    data[5519] = 0.0 - k[151]*y[IDX_SiOI];
    data[5520] = 0.0 + k[405] + k[2023] + k[2024];
    data[5521] = 0.0 - k[1076]*y[IDX_SiOI];
    data[5522] = 0.0 - k[1096]*y[IDX_SiOI];
    data[5523] = 0.0 + k[206]*y[IDX_SiOII];
    data[5524] = 0.0 - k[1156]*y[IDX_SiOI];
    data[5525] = 0.0 - k[1285]*y[IDX_SiOI] - k[1286]*y[IDX_SiOI];
    data[5526] = 0.0 + k[232]*y[IDX_SiOII];
    data[5527] = 0.0 + k[1443]*y[IDX_SiOHII];
    data[5528] = 0.0 + k[305]*y[IDX_SiOII] + k[1958]*y[IDX_SiI];
    data[5529] = 0.0 + k[1921]*y[IDX_SiCI] + k[1922]*y[IDX_SiH2I] +
        k[1923]*y[IDX_SiH2I] + k[1926]*y[IDX_SiHI] + k[2131]*y[IDX_SiI];
    data[5530] = 0.0 + k[1487]*y[IDX_SiSII] + k[1959]*y[IDX_SiI];
    data[5531] = 0.0 + k[1950]*y[IDX_SiI];
    data[5532] = 0.0 - k[1537]*y[IDX_SiOI];
    data[5533] = 0.0 + k[1950]*y[IDX_OHI] + k[1956]*y[IDX_CO2I] +
        k[1957]*y[IDX_COI] + k[1958]*y[IDX_NOI] + k[1959]*y[IDX_O2I] +
        k[2131]*y[IDX_OI];
    data[5534] = 0.0 + k[1921]*y[IDX_OI];
    data[5535] = 0.0 + k[1926]*y[IDX_OI];
    data[5536] = 0.0 + k[1922]*y[IDX_OI] + k[1923]*y[IDX_OI];
    data[5537] = 0.0 - k[151]*y[IDX_HII] - k[456] - k[645]*y[IDX_CII] -
        k[1076]*y[IDX_H3II] - k[1096]*y[IDX_H3OII] - k[1156]*y[IDX_HCOII] -
        k[1285]*y[IDX_HeII] - k[1286]*y[IDX_HeII] - k[1537]*y[IDX_OHII] -
        k[2093] - k[2094] - k[2171];
    data[5538] = 0.0 + k[206]*y[IDX_HCOI] + k[232]*y[IDX_MgI] +
        k[305]*y[IDX_NOI];
    data[5539] = 0.0 + k[604]*y[IDX_EM] + k[1443]*y[IDX_NH3I];
    data[5540] = 0.0 + k[1487]*y[IDX_O2I];
    data[5541] = 0.0 - k[704]*y[IDX_SiOII];
    data[5542] = 0.0 - k[659]*y[IDX_SiOII];
    data[5543] = 0.0 - k[864]*y[IDX_SiOII];
    data[5544] = 0.0 - k[773]*y[IDX_SiOII];
    data[5545] = 0.0 - k[881]*y[IDX_SiOII];
    data[5546] = 0.0 - k[602]*y[IDX_SiOII];
    data[5547] = 0.0 + k[151]*y[IDX_SiOI];
    data[5548] = 0.0 - k[964]*y[IDX_SiOII];
    data[5549] = 0.0 - k[206]*y[IDX_SiOII];
    data[5550] = 0.0 - k[232]*y[IDX_SiOII];
    data[5551] = 0.0 - k[1343]*y[IDX_SiOII] - k[1344]*y[IDX_SiOII];
    data[5552] = 0.0 - k[305]*y[IDX_SiOII];
    data[5553] = 0.0 + k[1512]*y[IDX_SiCII] + k[1513]*y[IDX_SiHII] -
        k[1516]*y[IDX_SiOII] + k[2130]*y[IDX_SiII];
    data[5554] = 0.0 + k[1488]*y[IDX_SiSII];
    data[5555] = 0.0 + k[1549]*y[IDX_SiII];
    data[5556] = 0.0 - k[1555]*y[IDX_SiOII];
    data[5557] = 0.0 + k[1549]*y[IDX_OHI] + k[2130]*y[IDX_OI];
    data[5558] = 0.0 + k[1512]*y[IDX_OI];
    data[5559] = 0.0 + k[1513]*y[IDX_OI];
    data[5560] = 0.0 + k[151]*y[IDX_HII] + k[2094];
    data[5561] = 0.0 - k[206]*y[IDX_HCOI] - k[232]*y[IDX_MgI] -
        k[305]*y[IDX_NOI] - k[602]*y[IDX_EM] - k[659]*y[IDX_C2I] -
        k[704]*y[IDX_CI] - k[773]*y[IDX_CH2I] - k[864]*y[IDX_CHI] -
        k[881]*y[IDX_COI] - k[964]*y[IDX_H2I] - k[1343]*y[IDX_NI] -
        k[1344]*y[IDX_NI] - k[1516]*y[IDX_OI] - k[1555]*y[IDX_SI] - k[2092] -
        k[2187];
    data[5562] = 0.0 + k[1488]*y[IDX_O2I];
    data[5563] = 0.0 + k[1563]*y[IDX_SiII];
    data[5564] = 0.0 + k[1564]*y[IDX_SiII];
    data[5565] = 0.0 - k[603]*y[IDX_SiOHII] - k[604]*y[IDX_SiOHII];
    data[5566] = 0.0 + k[896]*y[IDX_H2SiOI];
    data[5567] = 0.0 + k[964]*y[IDX_SiOII];
    data[5568] = 0.0 + k[1009]*y[IDX_HSiSII] + k[1013]*y[IDX_SiII];
    data[5569] = 0.0 + k[896]*y[IDX_HII] + k[1228]*y[IDX_HeII];
    data[5570] = 0.0 + k[1076]*y[IDX_SiOI];
    data[5571] = 0.0 + k[1096]*y[IDX_SiOI];
    data[5572] = 0.0 + k[1156]*y[IDX_SiOI];
    data[5573] = 0.0 + k[1228]*y[IDX_H2SiOI];
    data[5574] = 0.0 + k[1009]*y[IDX_H2OI];
    data[5575] = 0.0 - k[1443]*y[IDX_SiOHII];
    data[5576] = 0.0 + k[1514]*y[IDX_SiH2II] + k[1515]*y[IDX_SiH3II];
    data[5577] = 0.0 + k[1567]*y[IDX_SiH2II];
    data[5578] = 0.0 + k[1537]*y[IDX_SiOI];
    data[5579] = 0.0 + k[1013]*y[IDX_H2OI] + k[1563]*y[IDX_C2H5OHI] +
        k[1564]*y[IDX_CH3OHI];
    data[5580] = 0.0 + k[1514]*y[IDX_OI] + k[1567]*y[IDX_O2I];
    data[5581] = 0.0 + k[1515]*y[IDX_OI];
    data[5582] = 0.0 + k[1076]*y[IDX_H3II] + k[1096]*y[IDX_H3OII] +
        k[1156]*y[IDX_HCOII] + k[1537]*y[IDX_OHII];
    data[5583] = 0.0 + k[964]*y[IDX_H2I];
    data[5584] = 0.0 - k[603]*y[IDX_EM] - k[604]*y[IDX_EM] -
        k[1443]*y[IDX_NH3I] - k[2188];
    data[5585] = 0.0 + k[2483] + k[2484] + k[2485] + k[2486];
    data[5586] = 0.0 - k[32]*y[IDX_SiSI] - k[646]*y[IDX_SiSI];
    data[5587] = 0.0 + k[562]*y[IDX_HSiSII];
    data[5588] = 0.0 - k[152]*y[IDX_SiSI];
    data[5589] = 0.0 + k[1020]*y[IDX_HSiSII];
    data[5590] = 0.0 - k[1077]*y[IDX_SiSI];
    data[5591] = 0.0 + k[1127]*y[IDX_HSiSII];
    data[5592] = 0.0 - k[1157]*y[IDX_SiSI];
    data[5593] = 0.0 - k[1287]*y[IDX_SiSI] - k[1288]*y[IDX_SiSI];
    data[5594] = 0.0 + k[562]*y[IDX_EM] + k[1020]*y[IDX_H2SI] +
        k[1127]*y[IDX_HCNI] + k[1439]*y[IDX_NH3I];
    data[5595] = 0.0 + k[1439]*y[IDX_HSiSII];
    data[5596] = 0.0 - k[344]*y[IDX_SiSI];
    data[5597] = 0.0 - k[32]*y[IDX_CII] - k[152]*y[IDX_HII] -
        k[344]*y[IDX_SII] - k[457] - k[646]*y[IDX_CII] - k[1077]*y[IDX_H3II] -
        k[1157]*y[IDX_HCOII] - k[1287]*y[IDX_HeII] - k[1288]*y[IDX_HeII] -
        k[2095] - k[2189];
    data[5598] = 0.0 + k[32]*y[IDX_SiSI];
    data[5599] = 0.0 - k[605]*y[IDX_SiSII];
    data[5600] = 0.0 - k[1109]*y[IDX_SiSII];
    data[5601] = 0.0 + k[152]*y[IDX_SiSI];
    data[5602] = 0.0 - k[1487]*y[IDX_SiSII] - k[1488]*y[IDX_SiSII];
    data[5603] = 0.0 + k[1565]*y[IDX_SiII];
    data[5604] = 0.0 + k[344]*y[IDX_SiSI] + k[1569]*y[IDX_SiHI];
    data[5605] = 0.0 + k[1565]*y[IDX_OCSI];
    data[5606] = 0.0 + k[1569]*y[IDX_SII];
    data[5607] = 0.0 + k[32]*y[IDX_CII] + k[152]*y[IDX_HII] +
        k[344]*y[IDX_SII];
    data[5608] = 0.0 - k[605]*y[IDX_EM] - k[1109]*y[IDX_HI] -
        k[1487]*y[IDX_O2I] - k[1488]*y[IDX_O2I] - k[2190];
    data[5609] = 0.0 + k[2459] + k[2460] + k[2461] + k[2462];
    data[5610] = 0.0 + k[1614]*y[IDX_SO2I] - k[1615]*y[IDX_SOI] -
        k[1616]*y[IDX_SOI];
    data[5611] = 0.0 - k[25]*y[IDX_SOI] - k[639]*y[IDX_SOI] -
        k[640]*y[IDX_SOI] - k[641]*y[IDX_SOI];
    data[5612] = 0.0 - k[1699]*y[IDX_SOI] - k[1700]*y[IDX_SOI];
    data[5613] = 0.0 - k[788]*y[IDX_SOI];
    data[5614] = 0.0 + k[1884]*y[IDX_OI];
    data[5615] = 0.0 + k[557]*y[IDX_HSOII] + k[559]*y[IDX_HSO2II] +
        k[560]*y[IDX_HSO2II] + k[579]*y[IDX_OCSII] + k[586]*y[IDX_SO2II];
    data[5616] = 0.0 - k[1778]*y[IDX_SOI] - k[1779]*y[IDX_SOI];
    data[5617] = 0.0 - k[142]*y[IDX_SOI];
    data[5618] = 0.0 - k[1070]*y[IDX_SOI];
    data[5619] = 0.0 - k[1152]*y[IDX_SOI];
    data[5620] = 0.0 - k[1272]*y[IDX_SOI] - k[1273]*y[IDX_SOI];
    data[5621] = 0.0 + k[1900]*y[IDX_OI];
    data[5622] = 0.0 + k[557]*y[IDX_EM];
    data[5623] = 0.0 + k[559]*y[IDX_EM] + k[560]*y[IDX_EM];
    data[5624] = 0.0 + k[230]*y[IDX_SOII];
    data[5625] = 0.0 - k[1830]*y[IDX_SOI] - k[1831]*y[IDX_SOI];
    data[5626] = 0.0 + k[293]*y[IDX_SOII];
    data[5627] = 0.0 + k[1862]*y[IDX_SI];
    data[5628] = 0.0 + k[1908]*y[IDX_OI];
    data[5629] = 0.0 + k[1884]*y[IDX_CSI] + k[1900]*y[IDX_HSI] +
        k[1908]*y[IDX_NSI] + k[1913]*y[IDX_OCSI] + k[1915]*y[IDX_S2I] +
        k[1916]*y[IDX_SO2I] - k[1917]*y[IDX_SOI] - k[2129]*y[IDX_SOI];
    data[5630] = 0.0 + k[1488]*y[IDX_SiSII] + k[1865]*y[IDX_SI] -
        k[1866]*y[IDX_SOI];
    data[5631] = 0.0 + k[1913]*y[IDX_OI];
    data[5632] = 0.0 + k[579]*y[IDX_EM];
    data[5633] = 0.0 + k[1948]*y[IDX_SI] - k[1949]*y[IDX_SOI];
    data[5634] = 0.0 + k[1555]*y[IDX_SiOII] + k[1862]*y[IDX_NOI] +
        k[1865]*y[IDX_O2I] + k[1948]*y[IDX_OHI] + k[1954]*y[IDX_SO2I] +
        k[1954]*y[IDX_SO2I] - k[1955]*y[IDX_SOI];
    data[5635] = 0.0 + k[1915]*y[IDX_OI];
    data[5636] = 0.0 + k[1555]*y[IDX_SI];
    data[5637] = 0.0 + k[1488]*y[IDX_O2I];
    data[5638] = 0.0 - k[25]*y[IDX_CII] - k[142]*y[IDX_HII] - k[446] -
        k[447] - k[639]*y[IDX_CII] - k[640]*y[IDX_CII] - k[641]*y[IDX_CII] -
        k[788]*y[IDX_CH3II] - k[1070]*y[IDX_H3II] - k[1152]*y[IDX_HCOII] -
        k[1272]*y[IDX_HeII] - k[1273]*y[IDX_HeII] - k[1615]*y[IDX_CI] -
        k[1616]*y[IDX_CI] - k[1699]*y[IDX_CHI] - k[1700]*y[IDX_CHI] -
        k[1778]*y[IDX_HI] - k[1779]*y[IDX_HI] - k[1830]*y[IDX_NI] -
        k[1831]*y[IDX_NI] - k[1866]*y[IDX_O2I] - k[1917]*y[IDX_OI] -
        k[1949]*y[IDX_OHI] - k[1955]*y[IDX_SI] - k[2075] - k[2076] -
        k[2129]*y[IDX_OI] - k[2256];
    data[5639] = 0.0 + k[230]*y[IDX_MgI] + k[293]*y[IDX_NH3I];
    data[5640] = 0.0 + k[445] + k[1614]*y[IDX_CI] + k[1916]*y[IDX_OI] +
        k[1954]*y[IDX_SI] + k[1954]*y[IDX_SI] + k[2074];
    data[5641] = 0.0 + k[586]*y[IDX_EM];
    data[5642] = 0.0 + k[25]*y[IDX_SOI] + k[638]*y[IDX_SO2I];
    data[5643] = 0.0 - k[1556]*y[IDX_SOII] - k[1557]*y[IDX_SOII] -
        k[1558]*y[IDX_SOII];
    data[5644] = 0.0 - k[1559]*y[IDX_SOII] - k[1560]*y[IDX_SOII] -
        k[1561]*y[IDX_SOII];
    data[5645] = 0.0 + k[879]*y[IDX_SO2II];
    data[5646] = 0.0 + k[873]*y[IDX_SO2I];
    data[5647] = 0.0 - k[584]*y[IDX_SOII];
    data[5648] = 0.0 + k[1107]*y[IDX_SO2II];
    data[5649] = 0.0 + k[142]*y[IDX_SOI];
    data[5650] = 0.0 - k[1021]*y[IDX_SOII];
    data[5651] = 0.0 + k[1499]*y[IDX_OI];
    data[5652] = 0.0 + k[1271]*y[IDX_SO2I];
    data[5653] = 0.0 + k[1504]*y[IDX_OI];
    data[5654] = 0.0 - k[230]*y[IDX_SOII];
    data[5655] = 0.0 - k[1341]*y[IDX_SOII];
    data[5656] = 0.0 - k[293]*y[IDX_SOII];
    data[5657] = 0.0 + k[1499]*y[IDX_H2SII] + k[1504]*y[IDX_HSII];
    data[5658] = 0.0 + k[1481]*y[IDX_SO2I];
    data[5659] = 0.0 + k[1486]*y[IDX_SII] + k[1487]*y[IDX_SiSII];
    data[5660] = 0.0 + k[1484]*y[IDX_SI];
    data[5661] = 0.0 - k[1562]*y[IDX_SOII];
    data[5662] = 0.0 + k[1548]*y[IDX_SII];
    data[5663] = 0.0 + k[1534]*y[IDX_SI];
    data[5664] = 0.0 + k[1484]*y[IDX_O2II] + k[1534]*y[IDX_OHII];
    data[5665] = 0.0 + k[1486]*y[IDX_O2I] + k[1548]*y[IDX_OHI];
    data[5666] = 0.0 + k[1487]*y[IDX_O2I];
    data[5667] = 0.0 + k[25]*y[IDX_CII] + k[142]*y[IDX_HII] + k[447] +
        k[2076];
    data[5668] = 0.0 - k[230]*y[IDX_MgI] - k[293]*y[IDX_NH3I] -
        k[584]*y[IDX_EM] - k[1021]*y[IDX_H2SI] - k[1341]*y[IDX_NI] -
        k[1556]*y[IDX_C2H2I] - k[1557]*y[IDX_C2H2I] - k[1558]*y[IDX_C2H2I] -
        k[1559]*y[IDX_C2H4I] - k[1560]*y[IDX_C2H4I] - k[1561]*y[IDX_C2H4I] -
        k[1562]*y[IDX_OCSI] - k[2267];
    data[5669] = 0.0 + k[638]*y[IDX_CII] + k[873]*y[IDX_COII] +
        k[1271]*y[IDX_HeII] + k[1481]*y[IDX_OII];
    data[5670] = 0.0 + k[879]*y[IDX_COI] + k[1107]*y[IDX_HI];
    data[5671] = 0.0 + k[2499] + k[2500] + k[2501] + k[2502];
    data[5672] = 0.0 - k[1614]*y[IDX_SO2I];
    data[5673] = 0.0 - k[638]*y[IDX_SO2I];
    data[5674] = 0.0 - k[873]*y[IDX_SO2I];
    data[5675] = 0.0 + k[558]*y[IDX_HSO2II];
    data[5676] = 0.0 - k[141]*y[IDX_SO2I];
    data[5677] = 0.0 + k[1008]*y[IDX_HSO2II];
    data[5678] = 0.0 - k[989]*y[IDX_SO2I];
    data[5679] = 0.0 - k[1069]*y[IDX_SO2I];
    data[5680] = 0.0 - k[218]*y[IDX_SO2I] - k[1270]*y[IDX_SO2I] -
        k[1271]*y[IDX_SO2I];
    data[5681] = 0.0 + k[558]*y[IDX_EM] + k[1008]*y[IDX_H2OI] +
        k[1438]*y[IDX_NH3I];
    data[5682] = 0.0 + k[1438]*y[IDX_HSO2II];
    data[5683] = 0.0 - k[1916]*y[IDX_SO2I] + k[2129]*y[IDX_SOI];
    data[5684] = 0.0 - k[320]*y[IDX_SO2I] - k[1481]*y[IDX_SO2I];
    data[5685] = 0.0 + k[325]*y[IDX_SO2II] + k[1866]*y[IDX_SOI];
    data[5686] = 0.0 + k[1949]*y[IDX_SOI];
    data[5687] = 0.0 - k[1954]*y[IDX_SO2I];
    data[5688] = 0.0 + k[1866]*y[IDX_O2I] + k[1949]*y[IDX_OHI] +
        k[2129]*y[IDX_OI];
    data[5689] = 0.0 - k[141]*y[IDX_HII] - k[218]*y[IDX_HeII] -
        k[320]*y[IDX_OII] - k[445] - k[638]*y[IDX_CII] - k[873]*y[IDX_COII] -
        k[989]*y[IDX_H2OII] - k[1069]*y[IDX_H3II] - k[1270]*y[IDX_HeII] -
        k[1271]*y[IDX_HeII] - k[1481]*y[IDX_OII] - k[1614]*y[IDX_CI] -
        k[1916]*y[IDX_OI] - k[1954]*y[IDX_SI] - k[2074] - k[2260];
    data[5690] = 0.0 + k[325]*y[IDX_O2I];
    data[5691] = 0.0 - k[879]*y[IDX_SO2II];
    data[5692] = 0.0 - k[585]*y[IDX_SO2II] - k[586]*y[IDX_SO2II];
    data[5693] = 0.0 - k[1107]*y[IDX_SO2II];
    data[5694] = 0.0 + k[141]*y[IDX_SO2I];
    data[5695] = 0.0 - k[962]*y[IDX_SO2II];
    data[5696] = 0.0 + k[218]*y[IDX_SO2I];
    data[5697] = 0.0 + k[320]*y[IDX_SO2I];
    data[5698] = 0.0 - k[325]*y[IDX_SO2II];
    data[5699] = 0.0 + k[141]*y[IDX_HII] + k[218]*y[IDX_HeII] +
        k[320]*y[IDX_OII];
    data[5700] = 0.0 - k[325]*y[IDX_O2I] - k[585]*y[IDX_EM] -
        k[586]*y[IDX_EM] - k[879]*y[IDX_COI] - k[962]*y[IDX_H2I] -
        k[1107]*y[IDX_HI] - k[2284];
    
    // clang-format on

    /* */

    return NAUNET_SUCCESS;
}