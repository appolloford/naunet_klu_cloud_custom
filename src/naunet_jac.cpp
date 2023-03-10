#include <math.h>
/* */
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_sparse.h>  // access to sparse SUNMatrix
/* */
#include "naunet_constants.h"
#include "naunet_macros.h"
#include "naunet_ode.h"
#include "naunet_physics.h"

#ifdef USE_CUDA
#define NVEC_CUDA_CONTENT(x) ((N_VectorContent_Cuda)(x->content))
#define NVEC_CUDA_STREAM(x) (NVEC_CUDA_CONTENT(x)->stream_exec_policy->stream())
#define NVEC_CUDA_BLOCKSIZE(x) \
    (NVEC_CUDA_CONTENT(x)->stream_exec_policy->blockSize())
#define NVEC_CUDA_GRIDSIZE(x, n) \
    (NVEC_CUDA_CONTENT(x)->stream_exec_policy->gridSize(n))
#endif

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
    realtype nH = u_data->nH;
    realtype Tgas = u_data->Tgas;
    realtype zeta = u_data->zeta;
    realtype Av = u_data->Av;
    realtype omega = u_data->omega;
    realtype G0 = u_data->G0;
    realtype gdens = u_data->gdens;
    realtype rG = u_data->rG;
    realtype sites = u_data->sites;
    realtype fr = u_data->fr;
    realtype opt_crd = u_data->opt_crd;
    realtype opt_uvd = u_data->opt_uvd;
    realtype opt_h2d = u_data->opt_h2d;
    realtype eb_crd = u_data->eb_crd;
    realtype eb_uvd = u_data->eb_uvd;
    realtype eb_h2d = u_data->eb_h2d;
    realtype crdeseff = u_data->crdeseff;
    realtype uvcreff = u_data->uvcreff;
    realtype h2deseff = u_data->h2deseff;
    realtype opt_thd = u_data->opt_thd;
    
    realtype h2col = 0.5*1.59e21*Av;
    realtype cocol = 1e-5 * h2col;
    realtype lambdabar = GetCharactWavelength(h2col, cocol);
    realtype H2shielding = GetShieldingFactor(IDX_H2I, h2col, h2col, Tgas, 1);
    realtype H2formation = 1.0e-17 * sqrt(Tgas) * nH;
    realtype H2dissociation = 5.1e-11 * G0 * GetGrainScattering(Av, 1000.0) * H2shielding;
    realtype mant = GetMantleDens(y);
    realtype mantabund = mant / nH;
    realtype gxsec = (pi*rG*rG) * gdens;
    realtype garea = 4.0 * gxsec;
    realtype unisites = sites * (4*pi*rG*rG);
    realtype densites = garea * sites;
        
#if (NHEATPROCS || NCOOLPROCS)
    if (mu < 0) mu = GetMu(y);
    if (gamma < 0) gamma = GetGamma(y);
#endif

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
    rowptrs[1] = 2;
    rowptrs[2] = 4;
    rowptrs[3] = 6;
    rowptrs[4] = 8;
    rowptrs[5] = 10;
    rowptrs[6] = 12;
    rowptrs[7] = 14;
    rowptrs[8] = 15;
    rowptrs[9] = 17;
    rowptrs[10] = 19;
    rowptrs[11] = 21;
    rowptrs[12] = 22;
    rowptrs[13] = 24;
    rowptrs[14] = 27;
    rowptrs[15] = 30;
    rowptrs[16] = 33;
    rowptrs[17] = 36;
    rowptrs[18] = 39;
    rowptrs[19] = 42;
    rowptrs[20] = 45;
    rowptrs[21] = 48;
    rowptrs[22] = 51;
    rowptrs[23] = 54;
    rowptrs[24] = 57;
    rowptrs[25] = 60;
    rowptrs[26] = 63;
    rowptrs[27] = 66;
    rowptrs[28] = 69;
    rowptrs[29] = 72;
    rowptrs[30] = 75;
    rowptrs[31] = 78;
    rowptrs[32] = 82;
    rowptrs[33] = 87;
    rowptrs[34] = 91;
    rowptrs[35] = 95;
    rowptrs[36] = 99;
    rowptrs[37] = 103;
    rowptrs[38] = 107;
    rowptrs[39] = 111;
    rowptrs[40] = 115;
    rowptrs[41] = 121;
    rowptrs[42] = 126;
    rowptrs[43] = 133;
    rowptrs[44] = 140;
    rowptrs[45] = 146;
    rowptrs[46] = 152;
    rowptrs[47] = 160;
    rowptrs[48] = 168;
    rowptrs[49] = 176;
    rowptrs[50] = 181;
    rowptrs[51] = 188;
    rowptrs[52] = 192;
    rowptrs[53] = 202;
    rowptrs[54] = 208;
    rowptrs[55] = 214;
    rowptrs[56] = 222;
    rowptrs[57] = 231;
    rowptrs[58] = 237;
    rowptrs[59] = 244;
    rowptrs[60] = 252;
    rowptrs[61] = 259;
    rowptrs[62] = 264;
    rowptrs[63] = 276;
    rowptrs[64] = 288;
    rowptrs[65] = 296;
    rowptrs[66] = 307;
    rowptrs[67] = 316;
    rowptrs[68] = 324;
    rowptrs[69] = 332;
    rowptrs[70] = 339;
    rowptrs[71] = 348;
    rowptrs[72] = 357;
    rowptrs[73] = 365;
    rowptrs[74] = 371;
    rowptrs[75] = 380;
    rowptrs[76] = 388;
    rowptrs[77] = 400;
    rowptrs[78] = 410;
    rowptrs[79] = 419;
    rowptrs[80] = 428;
    rowptrs[81] = 437;
    rowptrs[82] = 445;
    rowptrs[83] = 456;
    rowptrs[84] = 470;
    rowptrs[85] = 482;
    rowptrs[86] = 495;
    rowptrs[87] = 505;
    rowptrs[88] = 513;
    rowptrs[89] = 527;
    rowptrs[90] = 539;
    rowptrs[91] = 553;
    rowptrs[92] = 564;
    rowptrs[93] = 574;
    rowptrs[94] = 586;
    rowptrs[95] = 601;
    rowptrs[96] = 615;
    rowptrs[97] = 626;
    rowptrs[98] = 637;
    rowptrs[99] = 650;
    rowptrs[100] = 665;
    rowptrs[101] = 678;
    rowptrs[102] = 696;
    rowptrs[103] = 708;
    rowptrs[104] = 721;
    rowptrs[105] = 732;
    rowptrs[106] = 749;
    rowptrs[107] = 764;
    rowptrs[108] = 779;
    rowptrs[109] = 794;
    rowptrs[110] = 808;
    rowptrs[111] = 821;
    rowptrs[112] = 837;
    rowptrs[113] = 850;
    rowptrs[114] = 865;
    rowptrs[115] = 882;
    rowptrs[116] = 900;
    rowptrs[117] = 913;
    rowptrs[118] = 929;
    rowptrs[119] = 949;
    rowptrs[120] = 965;
    rowptrs[121] = 984;
    rowptrs[122] = 1004;
    rowptrs[123] = 1027;
    rowptrs[124] = 1044;
    rowptrs[125] = 1066;
    rowptrs[126] = 1088;
    rowptrs[127] = 1111;
    rowptrs[128] = 1125;
    rowptrs[129] = 1144;
    rowptrs[130] = 1167;
    rowptrs[131] = 1191;
    rowptrs[132] = 1218;
    rowptrs[133] = 1246;
    rowptrs[134] = 1273;
    rowptrs[135] = 1295;
    rowptrs[136] = 1322;
    rowptrs[137] = 1355;
    rowptrs[138] = 1378;
    rowptrs[139] = 1405;
    rowptrs[140] = 1440;
    rowptrs[141] = 1466;
    rowptrs[142] = 1491;
    rowptrs[143] = 1520;
    rowptrs[144] = 1555;
    rowptrs[145] = 1584;
    rowptrs[146] = 1617;
    rowptrs[147] = 1652;
    rowptrs[148] = 1686;
    rowptrs[149] = 1723;
    rowptrs[150] = 1758;
    rowptrs[151] = 1793;
    rowptrs[152] = 1825;
    rowptrs[153] = 1873;
    rowptrs[154] = 1911;
    rowptrs[155] = 1945;
    rowptrs[156] = 1989;
    rowptrs[157] = 2035;
    rowptrs[158] = 2072;
    rowptrs[159] = 2114;
    rowptrs[160] = 2154;
    rowptrs[161] = 2198;
    rowptrs[162] = 2241;
    rowptrs[163] = 2284;
    rowptrs[164] = 2332;
    rowptrs[165] = 2367;
    rowptrs[166] = 2420;
    rowptrs[167] = 2458;
    rowptrs[168] = 2489;
    rowptrs[169] = 2531;
    rowptrs[170] = 2584;
    rowptrs[171] = 2629;
    rowptrs[172] = 2679;
    rowptrs[173] = 2724;
    rowptrs[174] = 2773;
    rowptrs[175] = 2823;
    rowptrs[176] = 2878;
    rowptrs[177] = 2930;
    rowptrs[178] = 2982;
    rowptrs[179] = 3028;
    rowptrs[180] = 3074;
    rowptrs[181] = 3130;
    rowptrs[182] = 3175;
    rowptrs[183] = 3227;
    rowptrs[184] = 3292;
    rowptrs[185] = 3354;
    rowptrs[186] = 3412;
    rowptrs[187] = 3473;
    rowptrs[188] = 3539;
    rowptrs[189] = 3596;
    rowptrs[190] = 3660;
    rowptrs[191] = 3728;
    rowptrs[192] = 3793;
    rowptrs[193] = 3859;
    rowptrs[194] = 3932;
    rowptrs[195] = 4002;
    rowptrs[196] = 4077;
    rowptrs[197] = 4141;
    rowptrs[198] = 4199;
    rowptrs[199] = 4271;
    rowptrs[200] = 4332;
    rowptrs[201] = 4409;
    rowptrs[202] = 4495;
    rowptrs[203] = 4577;
    rowptrs[204] = 4643;
    rowptrs[205] = 4712;
    rowptrs[206] = 4771;
    rowptrs[207] = 4843;
    rowptrs[208] = 4947;
    rowptrs[209] = 5040;
    rowptrs[210] = 5126;
    rowptrs[211] = 5216;
    rowptrs[212] = 5317;
    rowptrs[213] = 5430;
    rowptrs[214] = 5556;
    rowptrs[215] = 5701;
    
    // the column index of non-zero elements
    colvals[0] = 0;
    colvals[1] = 128;
    colvals[2] = 1;
    colvals[3] = 137;
    colvals[4] = 2;
    colvals[5] = 103;
    colvals[6] = 3;
    colvals[7] = 50;
    colvals[8] = 4;
    colvals[9] = 64;
    colvals[10] = 5;
    colvals[11] = 78;
    colvals[12] = 6;
    colvals[13] = 117;
    colvals[14] = 7;
    colvals[15] = 8;
    colvals[16] = 67;
    colvals[17] = 9;
    colvals[18] = 159;
    colvals[19] = 10;
    colvals[20] = 101;
    colvals[21] = 11;
    colvals[22] = 12;
    colvals[23] = 40;
    colvals[24] = 13;
    colvals[25] = 169;
    colvals[26] = 171;
    colvals[27] = 14;
    colvals[28] = 157;
    colvals[29] = 182;
    colvals[30] = 15;
    colvals[31] = 49;
    colvals[32] = 69;
    colvals[33] = 16;
    colvals[34] = 161;
    colvals[35] = 211;
    colvals[36] = 17;
    colvals[37] = 80;
    colvals[38] = 95;
    colvals[39] = 18;
    colvals[40] = 57;
    colvals[41] = 114;
    colvals[42] = 19;
    colvals[43] = 140;
    colvals[44] = 141;
    colvals[45] = 20;
    colvals[46] = 68;
    colvals[47] = 107;
    colvals[48] = 21;
    colvals[49] = 173;
    colvals[50] = 194;
    colvals[51] = 22;
    colvals[52] = 166;
    colvals[53] = 198;
    colvals[54] = 23;
    colvals[55] = 99;
    colvals[56] = 144;
    colvals[57] = 24;
    colvals[58] = 105;
    colvals[59] = 108;
    colvals[60] = 25;
    colvals[61] = 82;
    colvals[62] = 91;
    colvals[63] = 26;
    colvals[64] = 61;
    colvals[65] = 73;
    colvals[66] = 27;
    colvals[67] = 75;
    colvals[68] = 104;
    colvals[69] = 28;
    colvals[70] = 77;
    colvals[71] = 89;
    colvals[72] = 29;
    colvals[73] = 130;
    colvals[74] = 163;
    colvals[75] = 30;
    colvals[76] = 72;
    colvals[77] = 81;
    colvals[78] = 31;
    colvals[79] = 83;
    colvals[80] = 143;
    colvals[81] = 183;
    colvals[82] = 32;
    colvals[83] = 54;
    colvals[84] = 124;
    colvals[85] = 125;
    colvals[86] = 133;
    colvals[87] = 33;
    colvals[88] = 53;
    colvals[89] = 122;
    colvals[90] = 149;
    colvals[91] = 34;
    colvals[92] = 146;
    colvals[93] = 151;
    colvals[94] = 181;
    colvals[95] = 35;
    colvals[96] = 58;
    colvals[97] = 88;
    colvals[98] = 118;
    colvals[99] = 36;
    colvals[100] = 86;
    colvals[101] = 119;
    colvals[102] = 147;
    colvals[103] = 37;
    colvals[104] = 96;
    colvals[105] = 97;
    colvals[106] = 112;
    colvals[107] = 38;
    colvals[108] = 60;
    colvals[109] = 136;
    colvals[110] = 142;
    colvals[111] = 39;
    colvals[112] = 79;
    colvals[113] = 92;
    colvals[114] = 121;
    colvals[115] = 12;
    colvals[116] = 40;
    colvals[117] = 76;
    colvals[118] = 100;
    colvals[119] = 106;
    colvals[120] = 129;
    colvals[121] = 41;
    colvals[122] = 113;
    colvals[123] = 134;
    colvals[124] = 155;
    colvals[125] = 211;
    colvals[126] = 42;
    colvals[127] = 59;
    colvals[128] = 174;
    colvals[129] = 190;
    colvals[130] = 192;
    colvals[131] = 209;
    colvals[132] = 211;
    colvals[133] = 43;
    colvals[134] = 51;
    colvals[135] = 84;
    colvals[136] = 87;
    colvals[137] = 98;
    colvals[138] = 102;
    colvals[139] = 115;
    colvals[140] = 44;
    colvals[141] = 55;
    colvals[142] = 65;
    colvals[143] = 66;
    colvals[144] = 71;
    colvals[145] = 85;
    colvals[146] = 45;
    colvals[147] = 145;
    colvals[148] = 153;
    colvals[149] = 158;
    colvals[150] = 189;
    colvals[151] = 193;
    colvals[152] = 46;
    colvals[153] = 90;
    colvals[154] = 93;
    colvals[155] = 94;
    colvals[156] = 109;
    colvals[157] = 131;
    colvals[158] = 132;
    colvals[159] = 139;
    colvals[160] = 47;
    colvals[161] = 172;
    colvals[162] = 178;
    colvals[163] = 179;
    colvals[164] = 197;
    colvals[165] = 201;
    colvals[166] = 207;
    colvals[167] = 210;
    colvals[168] = 48;
    colvals[169] = 135;
    colvals[170] = 148;
    colvals[171] = 152;
    colvals[172] = 162;
    colvals[173] = 177;
    colvals[174] = 188;
    colvals[175] = 200;
    colvals[176] = 15;
    colvals[177] = 49;
    colvals[178] = 64;
    colvals[179] = 202;
    colvals[180] = 207;
    colvals[181] = 3;
    colvals[182] = 50;
    colvals[183] = 128;
    colvals[184] = 157;
    colvals[185] = 195;
    colvals[186] = 202;
    colvals[187] = 208;
    colvals[188] = 43;
    colvals[189] = 51;
    colvals[190] = 203;
    colvals[191] = 206;
    colvals[192] = 52;
    colvals[193] = 164;
    colvals[194] = 165;
    colvals[195] = 167;
    colvals[196] = 168;
    colvals[197] = 170;
    colvals[198] = 176;
    colvals[199] = 187;
    colvals[200] = 196;
    colvals[201] = 202;
    colvals[202] = 53;
    colvals[203] = 122;
    colvals[204] = 168;
    colvals[205] = 198;
    colvals[206] = 205;
    colvals[207] = 213;
    colvals[208] = 32;
    colvals[209] = 54;
    colvals[210] = 116;
    colvals[211] = 203;
    colvals[212] = 206;
    colvals[213] = 207;
    colvals[214] = 55;
    colvals[215] = 65;
    colvals[216] = 66;
    colvals[217] = 85;
    colvals[218] = 205;
    colvals[219] = 206;
    colvals[220] = 212;
    colvals[221] = 213;
    colvals[222] = 56;
    colvals[223] = 154;
    colvals[224] = 192;
    colvals[225] = 203;
    colvals[226] = 204;
    colvals[227] = 206;
    colvals[228] = 212;
    colvals[229] = 213;
    colvals[230] = 214;
    colvals[231] = 18;
    colvals[232] = 57;
    colvals[233] = 184;
    colvals[234] = 194;
    colvals[235] = 206;
    colvals[236] = 208;
    colvals[237] = 58;
    colvals[238] = 118;
    colvals[239] = 168;
    colvals[240] = 200;
    colvals[241] = 205;
    colvals[242] = 209;
    colvals[243] = 213;
    colvals[244] = 59;
    colvals[245] = 161;
    colvals[246] = 199;
    colvals[247] = 205;
    colvals[248] = 210;
    colvals[249] = 211;
    colvals[250] = 212;
    colvals[251] = 213;
    colvals[252] = 60;
    colvals[253] = 136;
    colvals[254] = 178;
    colvals[255] = 200;
    colvals[256] = 205;
    colvals[257] = 209;
    colvals[258] = 213;
    colvals[259] = 61;
    colvals[260] = 73;
    colvals[261] = 199;
    colvals[262] = 206;
    colvals[263] = 213;
    colvals[264] = 62;
    colvals[265] = 138;
    colvals[266] = 150;
    colvals[267] = 175;
    colvals[268] = 180;
    colvals[269] = 184;
    colvals[270] = 185;
    colvals[271] = 186;
    colvals[272] = 191;
    colvals[273] = 195;
    colvals[274] = 199;
    colvals[275] = 208;
    colvals[276] = 63;
    colvals[277] = 70;
    colvals[278] = 74;
    colvals[279] = 110;
    colvals[280] = 111;
    colvals[281] = 116;
    colvals[282] = 120;
    colvals[283] = 123;
    colvals[284] = 126;
    colvals[285] = 127;
    colvals[286] = 156;
    colvals[287] = 160;
    colvals[288] = 4;
    colvals[289] = 50;
    colvals[290] = 64;
    colvals[291] = 157;
    colvals[292] = 183;
    colvals[293] = 202;
    colvals[294] = 203;
    colvals[295] = 208;
    colvals[296] = 55;
    colvals[297] = 65;
    colvals[298] = 66;
    colvals[299] = 71;
    colvals[300] = 85;
    colvals[301] = 198;
    colvals[302] = 205;
    colvals[303] = 206;
    colvals[304] = 212;
    colvals[305] = 213;
    colvals[306] = 214;
    colvals[307] = 65;
    colvals[308] = 66;
    colvals[309] = 85;
    colvals[310] = 198;
    colvals[311] = 203;
    colvals[312] = 206;
    colvals[313] = 212;
    colvals[314] = 213;
    colvals[315] = 214;
    colvals[316] = 8;
    colvals[317] = 67;
    colvals[318] = 103;
    colvals[319] = 185;
    colvals[320] = 202;
    colvals[321] = 207;
    colvals[322] = 208;
    colvals[323] = 214;
    colvals[324] = 68;
    colvals[325] = 107;
    colvals[326] = 145;
    colvals[327] = 157;
    colvals[328] = 167;
    colvals[329] = 193;
    colvals[330] = 213;
    colvals[331] = 214;
    colvals[332] = 69;
    colvals[333] = 83;
    colvals[334] = 95;
    colvals[335] = 193;
    colvals[336] = 199;
    colvals[337] = 210;
    colvals[338] = 213;
    colvals[339] = 70;
    colvals[340] = 74;
    colvals[341] = 111;
    colvals[342] = 127;
    colvals[343] = 205;
    colvals[344] = 209;
    colvals[345] = 210;
    colvals[346] = 212;
    colvals[347] = 213;
    colvals[348] = 55;
    colvals[349] = 71;
    colvals[350] = 85;
    colvals[351] = 150;
    colvals[352] = 205;
    colvals[353] = 210;
    colvals[354] = 211;
    colvals[355] = 212;
    colvals[356] = 213;
    colvals[357] = 72;
    colvals[358] = 81;
    colvals[359] = 113;
    colvals[360] = 190;
    colvals[361] = 197;
    colvals[362] = 205;
    colvals[363] = 209;
    colvals[364] = 213;
    colvals[365] = 26;
    colvals[366] = 73;
    colvals[367] = 199;
    colvals[368] = 203;
    colvals[369] = 206;
    colvals[370] = 207;
    colvals[371] = 74;
    colvals[372] = 116;
    colvals[373] = 127;
    colvals[374] = 205;
    colvals[375] = 206;
    colvals[376] = 210;
    colvals[377] = 211;
    colvals[378] = 212;
    colvals[379] = 213;
    colvals[380] = 75;
    colvals[381] = 104;
    colvals[382] = 132;
    colvals[383] = 137;
    colvals[384] = 196;
    colvals[385] = 197;
    colvals[386] = 209;
    colvals[387] = 213;
    colvals[388] = 76;
    colvals[389] = 106;
    colvals[390] = 129;
    colvals[391] = 143;
    colvals[392] = 165;
    colvals[393] = 182;
    colvals[394] = 186;
    colvals[395] = 193;
    colvals[396] = 202;
    colvals[397] = 205;
    colvals[398] = 210;
    colvals[399] = 213;
    colvals[400] = 77;
    colvals[401] = 89;
    colvals[402] = 117;
    colvals[403] = 137;
    colvals[404] = 180;
    colvals[405] = 182;
    colvals[406] = 191;
    colvals[407] = 197;
    colvals[408] = 209;
    colvals[409] = 213;
    colvals[410] = 5;
    colvals[411] = 78;
    colvals[412] = 128;
    colvals[413] = 137;
    colvals[414] = 157;
    colvals[415] = 201;
    colvals[416] = 203;
    colvals[417] = 207;
    colvals[418] = 214;
    colvals[419] = 79;
    colvals[420] = 92;
    colvals[421] = 121;
    colvals[422] = 178;
    colvals[423] = 196;
    colvals[424] = 205;
    colvals[425] = 210;
    colvals[426] = 212;
    colvals[427] = 213;
    colvals[428] = 49;
    colvals[429] = 69;
    colvals[430] = 80;
    colvals[431] = 183;
    colvals[432] = 193;
    colvals[433] = 202;
    colvals[434] = 203;
    colvals[435] = 207;
    colvals[436] = 213;
    colvals[437] = 30;
    colvals[438] = 72;
    colvals[439] = 81;
    colvals[440] = 197;
    colvals[441] = 203;
    colvals[442] = 205;
    colvals[443] = 209;
    colvals[444] = 213;
    colvals[445] = 73;
    colvals[446] = 82;
    colvals[447] = 91;
    colvals[448] = 156;
    colvals[449] = 160;
    colvals[450] = 171;
    colvals[451] = 182;
    colvals[452] = 199;
    colvals[453] = 203;
    colvals[454] = 206;
    colvals[455] = 213;
    colvals[456] = 83;
    colvals[457] = 89;
    colvals[458] = 95;
    colvals[459] = 143;
    colvals[460] = 169;
    colvals[461] = 171;
    colvals[462] = 183;
    colvals[463] = 186;
    colvals[464] = 193;
    colvals[465] = 195;
    colvals[466] = 199;
    colvals[467] = 203;
    colvals[468] = 208;
    colvals[469] = 213;
    colvals[470] = 51;
    colvals[471] = 84;
    colvals[472] = 87;
    colvals[473] = 102;
    colvals[474] = 134;
    colvals[475] = 135;
    colvals[476] = 197;
    colvals[477] = 200;
    colvals[478] = 205;
    colvals[479] = 206;
    colvals[480] = 209;
    colvals[481] = 213;
    colvals[482] = 44;
    colvals[483] = 65;
    colvals[484] = 71;
    colvals[485] = 85;
    colvals[486] = 150;
    colvals[487] = 203;
    colvals[488] = 205;
    colvals[489] = 206;
    colvals[490] = 210;
    colvals[491] = 211;
    colvals[492] = 212;
    colvals[493] = 213;
    colvals[494] = 214;
    colvals[495] = 86;
    colvals[496] = 136;
    colvals[497] = 138;
    colvals[498] = 147;
    colvals[499] = 180;
    colvals[500] = 186;
    colvals[501] = 205;
    colvals[502] = 209;
    colvals[503] = 210;
    colvals[504] = 213;
    colvals[505] = 84;
    colvals[506] = 87;
    colvals[507] = 197;
    colvals[508] = 203;
    colvals[509] = 205;
    colvals[510] = 206;
    colvals[511] = 209;
    colvals[512] = 213;
    colvals[513] = 88;
    colvals[514] = 118;
    colvals[515] = 142;
    colvals[516] = 148;
    colvals[517] = 162;
    colvals[518] = 164;
    colvals[519] = 187;
    colvals[520] = 188;
    colvals[521] = 199;
    colvals[522] = 200;
    colvals[523] = 202;
    colvals[524] = 206;
    colvals[525] = 207;
    colvals[526] = 213;
    colvals[527] = 28;
    colvals[528] = 77;
    colvals[529] = 89;
    colvals[530] = 103;
    colvals[531] = 137;
    colvals[532] = 195;
    colvals[533] = 197;
    colvals[534] = 199;
    colvals[535] = 203;
    colvals[536] = 208;
    colvals[537] = 209;
    colvals[538] = 213;
    colvals[539] = 90;
    colvals[540] = 93;
    colvals[541] = 137;
    colvals[542] = 142;
    colvals[543] = 147;
    colvals[544] = 162;
    colvals[545] = 175;
    colvals[546] = 177;
    colvals[547] = 180;
    colvals[548] = 188;
    colvals[549] = 191;
    colvals[550] = 205;
    colvals[551] = 209;
    colvals[552] = 213;
    colvals[553] = 25;
    colvals[554] = 61;
    colvals[555] = 73;
    colvals[556] = 91;
    colvals[557] = 156;
    colvals[558] = 157;
    colvals[559] = 199;
    colvals[560] = 203;
    colvals[561] = 206;
    colvals[562] = 207;
    colvals[563] = 213;
    colvals[564] = 92;
    colvals[565] = 121;
    colvals[566] = 172;
    colvals[567] = 198;
    colvals[568] = 203;
    colvals[569] = 206;
    colvals[570] = 211;
    colvals[571] = 212;
    colvals[572] = 213;
    colvals[573] = 214;
    colvals[574] = 46;
    colvals[575] = 90;
    colvals[576] = 93;
    colvals[577] = 94;
    colvals[578] = 185;
    colvals[579] = 199;
    colvals[580] = 200;
    colvals[581] = 203;
    colvals[582] = 205;
    colvals[583] = 206;
    colvals[584] = 209;
    colvals[585] = 213;
    colvals[586] = 93;
    colvals[587] = 94;
    colvals[588] = 109;
    colvals[589] = 137;
    colvals[590] = 142;
    colvals[591] = 147;
    colvals[592] = 152;
    colvals[593] = 157;
    colvals[594] = 175;
    colvals[595] = 180;
    colvals[596] = 185;
    colvals[597] = 188;
    colvals[598] = 205;
    colvals[599] = 206;
    colvals[600] = 213;
    colvals[601] = 17;
    colvals[602] = 50;
    colvals[603] = 69;
    colvals[604] = 95;
    colvals[605] = 107;
    colvals[606] = 157;
    colvals[607] = 159;
    colvals[608] = 171;
    colvals[609] = 189;
    colvals[610] = 193;
    colvals[611] = 199;
    colvals[612] = 202;
    colvals[613] = 203;
    colvals[614] = 210;
    colvals[615] = 96;
    colvals[616] = 112;
    colvals[617] = 120;
    colvals[618] = 177;
    colvals[619] = 193;
    colvals[620] = 196;
    colvals[621] = 200;
    colvals[622] = 205;
    colvals[623] = 209;
    colvals[624] = 210;
    colvals[625] = 213;
    colvals[626] = 97;
    colvals[627] = 112;
    colvals[628] = 123;
    colvals[629] = 147;
    colvals[630] = 160;
    colvals[631] = 188;
    colvals[632] = 198;
    colvals[633] = 199;
    colvals[634] = 206;
    colvals[635] = 213;
    colvals[636] = 214;
    colvals[637] = 51;
    colvals[638] = 87;
    colvals[639] = 98;
    colvals[640] = 115;
    colvals[641] = 162;
    colvals[642] = 177;
    colvals[643] = 188;
    colvals[644] = 197;
    colvals[645] = 203;
    colvals[646] = 205;
    colvals[647] = 206;
    colvals[648] = 209;
    colvals[649] = 213;
    colvals[650] = 23;
    colvals[651] = 99;
    colvals[652] = 128;
    colvals[653] = 174;
    colvals[654] = 185;
    colvals[655] = 191;
    colvals[656] = 192;
    colvals[657] = 195;
    colvals[658] = 198;
    colvals[659] = 201;
    colvals[660] = 202;
    colvals[661] = 207;
    colvals[662] = 211;
    colvals[663] = 212;
    colvals[664] = 214;
    colvals[665] = 100;
    colvals[666] = 113;
    colvals[667] = 117;
    colvals[668] = 130;
    colvals[669] = 146;
    colvals[670] = 153;
    colvals[671] = 180;
    colvals[672] = 182;
    colvals[673] = 193;
    colvals[674] = 197;
    colvals[675] = 205;
    colvals[676] = 209;
    colvals[677] = 213;
    colvals[678] = 10;
    colvals[679] = 101;
    colvals[680] = 114;
    colvals[681] = 122;
    colvals[682] = 172;
    colvals[683] = 184;
    colvals[684] = 185;
    colvals[685] = 187;
    colvals[686] = 189;
    colvals[687] = 194;
    colvals[688] = 198;
    colvals[689] = 201;
    colvals[690] = 202;
    colvals[691] = 205;
    colvals[692] = 206;
    colvals[693] = 207;
    colvals[694] = 211;
    colvals[695] = 214;
    colvals[696] = 87;
    colvals[697] = 102;
    colvals[698] = 115;
    colvals[699] = 134;
    colvals[700] = 142;
    colvals[701] = 147;
    colvals[702] = 177;
    colvals[703] = 188;
    colvals[704] = 194;
    colvals[705] = 203;
    colvals[706] = 206;
    colvals[707] = 213;
    colvals[708] = 2;
    colvals[709] = 75;
    colvals[710] = 103;
    colvals[711] = 104;
    colvals[712] = 127;
    colvals[713] = 160;
    colvals[714] = 182;
    colvals[715] = 185;
    colvals[716] = 201;
    colvals[717] = 202;
    colvals[718] = 207;
    colvals[719] = 208;
    colvals[720] = 213;
    colvals[721] = 27;
    colvals[722] = 75;
    colvals[723] = 104;
    colvals[724] = 132;
    colvals[725] = 160;
    colvals[726] = 196;
    colvals[727] = 197;
    colvals[728] = 199;
    colvals[729] = 205;
    colvals[730] = 209;
    colvals[731] = 213;
    colvals[732] = 105;
    colvals[733] = 108;
    colvals[734] = 110;
    colvals[735] = 112;
    colvals[736] = 123;
    colvals[737] = 124;
    colvals[738] = 126;
    colvals[739] = 160;
    colvals[740] = 183;
    colvals[741] = 188;
    colvals[742] = 195;
    colvals[743] = 199;
    colvals[744] = 202;
    colvals[745] = 206;
    colvals[746] = 207;
    colvals[747] = 208;
    colvals[748] = 213;
    colvals[749] = 67;
    colvals[750] = 68;
    colvals[751] = 69;
    colvals[752] = 76;
    colvals[753] = 80;
    colvals[754] = 106;
    colvals[755] = 107;
    colvals[756] = 171;
    colvals[757] = 202;
    colvals[758] = 203;
    colvals[759] = 205;
    colvals[760] = 206;
    colvals[761] = 207;
    colvals[762] = 208;
    colvals[763] = 213;
    colvals[764] = 20;
    colvals[765] = 68;
    colvals[766] = 107;
    colvals[767] = 157;
    colvals[768] = 159;
    colvals[769] = 167;
    colvals[770] = 171;
    colvals[771] = 189;
    colvals[772] = 193;
    colvals[773] = 199;
    colvals[774] = 201;
    colvals[775] = 202;
    colvals[776] = 203;
    colvals[777] = 208;
    colvals[778] = 214;
    colvals[779] = 24;
    colvals[780] = 61;
    colvals[781] = 73;
    colvals[782] = 82;
    colvals[783] = 91;
    colvals[784] = 108;
    colvals[785] = 123;
    colvals[786] = 188;
    colvals[787] = 199;
    colvals[788] = 202;
    colvals[789] = 203;
    colvals[790] = 206;
    colvals[791] = 207;
    colvals[792] = 208;
    colvals[793] = 213;
    colvals[794] = 94;
    colvals[795] = 109;
    colvals[796] = 137;
    colvals[797] = 142;
    colvals[798] = 157;
    colvals[799] = 184;
    colvals[800] = 200;
    colvals[801] = 202;
    colvals[802] = 203;
    colvals[803] = 205;
    colvals[804] = 206;
    colvals[805] = 207;
    colvals[806] = 213;
    colvals[807] = 214;
    colvals[808] = 74;
    colvals[809] = 110;
    colvals[810] = 111;
    colvals[811] = 116;
    colvals[812] = 127;
    colvals[813] = 197;
    colvals[814] = 199;
    colvals[815] = 203;
    colvals[816] = 205;
    colvals[817] = 206;
    colvals[818] = 207;
    colvals[819] = 209;
    colvals[820] = 213;
    colvals[821] = 110;
    colvals[822] = 111;
    colvals[823] = 116;
    colvals[824] = 126;
    colvals[825] = 127;
    colvals[826] = 150;
    colvals[827] = 180;
    colvals[828] = 182;
    colvals[829] = 197;
    colvals[830] = 199;
    colvals[831] = 205;
    colvals[832] = 206;
    colvals[833] = 207;
    colvals[834] = 209;
    colvals[835] = 212;
    colvals[836] = 213;
    colvals[837] = 37;
    colvals[838] = 96;
    colvals[839] = 112;
    colvals[840] = 177;
    colvals[841] = 188;
    colvals[842] = 193;
    colvals[843] = 196;
    colvals[844] = 199;
    colvals[845] = 203;
    colvals[846] = 205;
    colvals[847] = 206;
    colvals[848] = 209;
    colvals[849] = 213;
    colvals[850] = 113;
    colvals[851] = 134;
    colvals[852] = 138;
    colvals[853] = 155;
    colvals[854] = 174;
    colvals[855] = 180;
    colvals[856] = 186;
    colvals[857] = 190;
    colvals[858] = 193;
    colvals[859] = 196;
    colvals[860] = 197;
    colvals[861] = 205;
    colvals[862] = 209;
    colvals[863] = 210;
    colvals[864] = 213;
    colvals[865] = 67;
    colvals[866] = 101;
    colvals[867] = 107;
    colvals[868] = 114;
    colvals[869] = 189;
    colvals[870] = 192;
    colvals[871] = 193;
    colvals[872] = 194;
    colvals[873] = 195;
    colvals[874] = 198;
    colvals[875] = 199;
    colvals[876] = 201;
    colvals[877] = 202;
    colvals[878] = 203;
    colvals[879] = 207;
    colvals[880] = 208;
    colvals[881] = 214;
    colvals[882] = 98;
    colvals[883] = 102;
    colvals[884] = 115;
    colvals[885] = 136;
    colvals[886] = 152;
    colvals[887] = 177;
    colvals[888] = 194;
    colvals[889] = 197;
    colvals[890] = 200;
    colvals[891] = 202;
    colvals[892] = 203;
    colvals[893] = 205;
    colvals[894] = 206;
    colvals[895] = 207;
    colvals[896] = 208;
    colvals[897] = 209;
    colvals[898] = 213;
    colvals[899] = 214;
    colvals[900] = 70;
    colvals[901] = 74;
    colvals[902] = 116;
    colvals[903] = 127;
    colvals[904] = 189;
    colvals[905] = 199;
    colvals[906] = 203;
    colvals[907] = 205;
    colvals[908] = 206;
    colvals[909] = 207;
    colvals[910] = 210;
    colvals[911] = 211;
    colvals[912] = 213;
    colvals[913] = 6;
    colvals[914] = 100;
    colvals[915] = 117;
    colvals[916] = 130;
    colvals[917] = 146;
    colvals[918] = 153;
    colvals[919] = 180;
    colvals[920] = 182;
    colvals[921] = 185;
    colvals[922] = 189;
    colvals[923] = 197;
    colvals[924] = 203;
    colvals[925] = 205;
    colvals[926] = 206;
    colvals[927] = 209;
    colvals[928] = 213;
    colvals[929] = 35;
    colvals[930] = 58;
    colvals[931] = 115;
    colvals[932] = 118;
    colvals[933] = 136;
    colvals[934] = 152;
    colvals[935] = 187;
    colvals[936] = 189;
    colvals[937] = 194;
    colvals[938] = 199;
    colvals[939] = 200;
    colvals[940] = 202;
    colvals[941] = 203;
    colvals[942] = 205;
    colvals[943] = 206;
    colvals[944] = 207;
    colvals[945] = 208;
    colvals[946] = 209;
    colvals[947] = 213;
    colvals[948] = 214;
    colvals[949] = 119;
    colvals[950] = 131;
    colvals[951] = 132;
    colvals[952] = 138;
    colvals[953] = 147;
    colvals[954] = 151;
    colvals[955] = 167;
    colvals[956] = 172;
    colvals[957] = 177;
    colvals[958] = 178;
    colvals[959] = 196;
    colvals[960] = 198;
    colvals[961] = 199;
    colvals[962] = 206;
    colvals[963] = 207;
    colvals[964] = 213;
    colvals[965] = 110;
    colvals[966] = 116;
    colvals[967] = 120;
    colvals[968] = 123;
    colvals[969] = 127;
    colvals[970] = 160;
    colvals[971] = 179;
    colvals[972] = 182;
    colvals[973] = 197;
    colvals[974] = 198;
    colvals[975] = 199;
    colvals[976] = 200;
    colvals[977] = 203;
    colvals[978] = 205;
    colvals[979] = 206;
    colvals[980] = 207;
    colvals[981] = 209;
    colvals[982] = 212;
    colvals[983] = 213;
    colvals[984] = 39;
    colvals[985] = 79;
    colvals[986] = 92;
    colvals[987] = 121;
    colvals[988] = 136;
    colvals[989] = 161;
    colvals[990] = 172;
    colvals[991] = 178;
    colvals[992] = 196;
    colvals[993] = 198;
    colvals[994] = 199;
    colvals[995] = 200;
    colvals[996] = 201;
    colvals[997] = 203;
    colvals[998] = 205;
    colvals[999] = 206;
    colvals[1000] = 207;
    colvals[1001] = 208;
    colvals[1002] = 210;
    colvals[1003] = 213;
    colvals[1004] = 33;
    colvals[1005] = 53;
    colvals[1006] = 101;
    colvals[1007] = 122;
    colvals[1008] = 149;
    colvals[1009] = 165;
    colvals[1010] = 184;
    colvals[1011] = 185;
    colvals[1012] = 187;
    colvals[1013] = 189;
    colvals[1014] = 192;
    colvals[1015] = 194;
    colvals[1016] = 195;
    colvals[1017] = 198;
    colvals[1018] = 201;
    colvals[1019] = 202;
    colvals[1020] = 203;
    colvals[1021] = 205;
    colvals[1022] = 206;
    colvals[1023] = 207;
    colvals[1024] = 211;
    colvals[1025] = 213;
    colvals[1026] = 214;
    colvals[1027] = 110;
    colvals[1028] = 111;
    colvals[1029] = 116;
    colvals[1030] = 120;
    colvals[1031] = 123;
    colvals[1032] = 127;
    colvals[1033] = 179;
    colvals[1034] = 188;
    colvals[1035] = 197;
    colvals[1036] = 199;
    colvals[1037] = 203;
    colvals[1038] = 205;
    colvals[1039] = 206;
    colvals[1040] = 207;
    colvals[1041] = 208;
    colvals[1042] = 209;
    colvals[1043] = 213;
    colvals[1044] = 97;
    colvals[1045] = 105;
    colvals[1046] = 124;
    colvals[1047] = 126;
    colvals[1048] = 133;
    colvals[1049] = 140;
    colvals[1050] = 160;
    colvals[1051] = 183;
    colvals[1052] = 184;
    colvals[1053] = 192;
    colvals[1054] = 194;
    colvals[1055] = 195;
    colvals[1056] = 198;
    colvals[1057] = 200;
    colvals[1058] = 201;
    colvals[1059] = 202;
    colvals[1060] = 206;
    colvals[1061] = 207;
    colvals[1062] = 208;
    colvals[1063] = 211;
    colvals[1064] = 212;
    colvals[1065] = 213;
    colvals[1066] = 54;
    colvals[1067] = 96;
    colvals[1068] = 104;
    colvals[1069] = 111;
    colvals[1070] = 120;
    colvals[1071] = 124;
    colvals[1072] = 125;
    colvals[1073] = 133;
    colvals[1074] = 134;
    colvals[1075] = 160;
    colvals[1076] = 179;
    colvals[1077] = 196;
    colvals[1078] = 197;
    colvals[1079] = 198;
    colvals[1080] = 203;
    colvals[1081] = 205;
    colvals[1082] = 206;
    colvals[1083] = 207;
    colvals[1084] = 209;
    colvals[1085] = 210;
    colvals[1086] = 212;
    colvals[1087] = 213;
    colvals[1088] = 110;
    colvals[1089] = 116;
    colvals[1090] = 123;
    colvals[1091] = 126;
    colvals[1092] = 127;
    colvals[1093] = 156;
    colvals[1094] = 160;
    colvals[1095] = 179;
    colvals[1096] = 182;
    colvals[1097] = 188;
    colvals[1098] = 195;
    colvals[1099] = 196;
    colvals[1100] = 197;
    colvals[1101] = 203;
    colvals[1102] = 205;
    colvals[1103] = 206;
    colvals[1104] = 207;
    colvals[1105] = 208;
    colvals[1106] = 209;
    colvals[1107] = 210;
    colvals[1108] = 212;
    colvals[1109] = 213;
    colvals[1110] = 214;
    colvals[1111] = 63;
    colvals[1112] = 70;
    colvals[1113] = 127;
    colvals[1114] = 150;
    colvals[1115] = 180;
    colvals[1116] = 182;
    colvals[1117] = 189;
    colvals[1118] = 203;
    colvals[1119] = 205;
    colvals[1120] = 206;
    colvals[1121] = 207;
    colvals[1122] = 209;
    colvals[1123] = 210;
    colvals[1124] = 213;
    colvals[1125] = 0;
    colvals[1126] = 103;
    colvals[1127] = 104;
    colvals[1128] = 127;
    colvals[1129] = 128;
    colvals[1130] = 137;
    colvals[1131] = 182;
    colvals[1132] = 184;
    colvals[1133] = 185;
    colvals[1134] = 189;
    colvals[1135] = 198;
    colvals[1136] = 199;
    colvals[1137] = 201;
    colvals[1138] = 202;
    colvals[1139] = 203;
    colvals[1140] = 206;
    colvals[1141] = 207;
    colvals[1142] = 208;
    colvals[1143] = 214;
    colvals[1144] = 95;
    colvals[1145] = 106;
    colvals[1146] = 107;
    colvals[1147] = 129;
    colvals[1148] = 143;
    colvals[1149] = 159;
    colvals[1150] = 164;
    colvals[1151] = 167;
    colvals[1152] = 169;
    colvals[1153] = 177;
    colvals[1154] = 182;
    colvals[1155] = 183;
    colvals[1156] = 186;
    colvals[1157] = 187;
    colvals[1158] = 189;
    colvals[1159] = 193;
    colvals[1160] = 196;
    colvals[1161] = 199;
    colvals[1162] = 202;
    colvals[1163] = 203;
    colvals[1164] = 206;
    colvals[1165] = 210;
    colvals[1166] = 213;
    colvals[1167] = 81;
    colvals[1168] = 117;
    colvals[1169] = 130;
    colvals[1170] = 138;
    colvals[1171] = 144;
    colvals[1172] = 146;
    colvals[1173] = 149;
    colvals[1174] = 150;
    colvals[1175] = 154;
    colvals[1176] = 158;
    colvals[1177] = 163;
    colvals[1178] = 164;
    colvals[1179] = 179;
    colvals[1180] = 191;
    colvals[1181] = 196;
    colvals[1182] = 201;
    colvals[1183] = 203;
    colvals[1184] = 205;
    colvals[1185] = 207;
    colvals[1186] = 208;
    colvals[1187] = 209;
    colvals[1188] = 210;
    colvals[1189] = 211;
    colvals[1190] = 213;
    colvals[1191] = 93;
    colvals[1192] = 109;
    colvals[1193] = 118;
    colvals[1194] = 131;
    colvals[1195] = 136;
    colvals[1196] = 139;
    colvals[1197] = 140;
    colvals[1198] = 143;
    colvals[1199] = 147;
    colvals[1200] = 152;
    colvals[1201] = 156;
    colvals[1202] = 162;
    colvals[1203] = 167;
    colvals[1204] = 183;
    colvals[1205] = 186;
    colvals[1206] = 188;
    colvals[1207] = 191;
    colvals[1208] = 195;
    colvals[1209] = 198;
    colvals[1210] = 199;
    colvals[1211] = 200;
    colvals[1212] = 203;
    colvals[1213] = 206;
    colvals[1214] = 207;
    colvals[1215] = 208;
    colvals[1216] = 212;
    colvals[1217] = 213;
    colvals[1218] = 104;
    colvals[1219] = 109;
    colvals[1220] = 129;
    colvals[1221] = 131;
    colvals[1222] = 132;
    colvals[1223] = 137;
    colvals[1224] = 139;
    colvals[1225] = 142;
    colvals[1226] = 147;
    colvals[1227] = 148;
    colvals[1228] = 157;
    colvals[1229] = 175;
    colvals[1230] = 177;
    colvals[1231] = 180;
    colvals[1232] = 184;
    colvals[1233] = 186;
    colvals[1234] = 188;
    colvals[1235] = 191;
    colvals[1236] = 196;
    colvals[1237] = 197;
    colvals[1238] = 199;
    colvals[1239] = 200;
    colvals[1240] = 205;
    colvals[1241] = 207;
    colvals[1242] = 208;
    colvals[1243] = 209;
    colvals[1244] = 212;
    colvals[1245] = 213;
    colvals[1246] = 11;
    colvals[1247] = 54;
    colvals[1248] = 97;
    colvals[1249] = 108;
    colvals[1250] = 110;
    colvals[1251] = 123;
    colvals[1252] = 124;
    colvals[1253] = 125;
    colvals[1254] = 133;
    colvals[1255] = 140;
    colvals[1256] = 156;
    colvals[1257] = 163;
    colvals[1258] = 179;
    colvals[1259] = 192;
    colvals[1260] = 194;
    colvals[1261] = 196;
    colvals[1262] = 197;
    colvals[1263] = 198;
    colvals[1264] = 199;
    colvals[1265] = 201;
    colvals[1266] = 203;
    colvals[1267] = 205;
    colvals[1268] = 206;
    colvals[1269] = 207;
    colvals[1270] = 209;
    colvals[1271] = 211;
    colvals[1272] = 213;
    colvals[1273] = 41;
    colvals[1274] = 72;
    colvals[1275] = 102;
    colvals[1276] = 113;
    colvals[1277] = 134;
    colvals[1278] = 138;
    colvals[1279] = 155;
    colvals[1280] = 160;
    colvals[1281] = 166;
    colvals[1282] = 167;
    colvals[1283] = 172;
    colvals[1284] = 174;
    colvals[1285] = 180;
    colvals[1286] = 186;
    colvals[1287] = 196;
    colvals[1288] = 197;
    colvals[1289] = 199;
    colvals[1290] = 203;
    colvals[1291] = 205;
    colvals[1292] = 206;
    colvals[1293] = 209;
    colvals[1294] = 213;
    colvals[1295] = 96;
    colvals[1296] = 98;
    colvals[1297] = 135;
    colvals[1298] = 138;
    colvals[1299] = 148;
    colvals[1300] = 150;
    colvals[1301] = 153;
    colvals[1302] = 155;
    colvals[1303] = 159;
    colvals[1304] = 162;
    colvals[1305] = 168;
    colvals[1306] = 175;
    colvals[1307] = 177;
    colvals[1308] = 178;
    colvals[1309] = 179;
    colvals[1310] = 182;
    colvals[1311] = 186;
    colvals[1312] = 190;
    colvals[1313] = 193;
    colvals[1314] = 196;
    colvals[1315] = 197;
    colvals[1316] = 200;
    colvals[1317] = 205;
    colvals[1318] = 209;
    colvals[1319] = 212;
    colvals[1320] = 213;
    colvals[1321] = 214;
    colvals[1322] = 38;
    colvals[1323] = 60;
    colvals[1324] = 79;
    colvals[1325] = 92;
    colvals[1326] = 97;
    colvals[1327] = 115;
    colvals[1328] = 118;
    colvals[1329] = 119;
    colvals[1330] = 121;
    colvals[1331] = 124;
    colvals[1332] = 136;
    colvals[1333] = 139;
    colvals[1334] = 140;
    colvals[1335] = 142;
    colvals[1336] = 147;
    colvals[1337] = 152;
    colvals[1338] = 180;
    colvals[1339] = 194;
    colvals[1340] = 195;
    colvals[1341] = 196;
    colvals[1342] = 198;
    colvals[1343] = 199;
    colvals[1344] = 200;
    colvals[1345] = 201;
    colvals[1346] = 202;
    colvals[1347] = 203;
    colvals[1348] = 205;
    colvals[1349] = 206;
    colvals[1350] = 207;
    colvals[1351] = 208;
    colvals[1352] = 209;
    colvals[1353] = 213;
    colvals[1354] = 214;
    colvals[1355] = 1;
    colvals[1356] = 75;
    colvals[1357] = 103;
    colvals[1358] = 117;
    colvals[1359] = 127;
    colvals[1360] = 137;
    colvals[1361] = 142;
    colvals[1362] = 154;
    colvals[1363] = 172;
    colvals[1364] = 180;
    colvals[1365] = 182;
    colvals[1366] = 185;
    colvals[1367] = 188;
    colvals[1368] = 189;
    colvals[1369] = 191;
    colvals[1370] = 195;
    colvals[1371] = 197;
    colvals[1372] = 201;
    colvals[1373] = 202;
    colvals[1374] = 203;
    colvals[1375] = 206;
    colvals[1376] = 207;
    colvals[1377] = 213;
    colvals[1378] = 134;
    colvals[1379] = 138;
    colvals[1380] = 147;
    colvals[1381] = 150;
    colvals[1382] = 154;
    colvals[1383] = 157;
    colvals[1384] = 161;
    colvals[1385] = 163;
    colvals[1386] = 167;
    colvals[1387] = 172;
    colvals[1388] = 177;
    colvals[1389] = 180;
    colvals[1390] = 185;
    colvals[1391] = 190;
    colvals[1392] = 191;
    colvals[1393] = 192;
    colvals[1394] = 196;
    colvals[1395] = 198;
    colvals[1396] = 203;
    colvals[1397] = 205;
    colvals[1398] = 206;
    colvals[1399] = 207;
    colvals[1400] = 210;
    colvals[1401] = 211;
    colvals[1402] = 212;
    colvals[1403] = 213;
    colvals[1404] = 214;
    colvals[1405] = 7;
    colvals[1406] = 86;
    colvals[1407] = 90;
    colvals[1408] = 93;
    colvals[1409] = 94;
    colvals[1410] = 104;
    colvals[1411] = 109;
    colvals[1412] = 115;
    colvals[1413] = 118;
    colvals[1414] = 119;
    colvals[1415] = 131;
    colvals[1416] = 132;
    colvals[1417] = 136;
    colvals[1418] = 139;
    colvals[1419] = 140;
    colvals[1420] = 147;
    colvals[1421] = 152;
    colvals[1422] = 156;
    colvals[1423] = 183;
    colvals[1424] = 184;
    colvals[1425] = 195;
    colvals[1426] = 196;
    colvals[1427] = 197;
    colvals[1428] = 199;
    colvals[1429] = 200;
    colvals[1430] = 201;
    colvals[1431] = 202;
    colvals[1432] = 203;
    colvals[1433] = 205;
    colvals[1434] = 206;
    colvals[1435] = 207;
    colvals[1436] = 208;
    colvals[1437] = 209;
    colvals[1438] = 213;
    colvals[1439] = 214;
    colvals[1440] = 19;
    colvals[1441] = 124;
    colvals[1442] = 131;
    colvals[1443] = 140;
    colvals[1444] = 141;
    colvals[1445] = 142;
    colvals[1446] = 148;
    colvals[1447] = 150;
    colvals[1448] = 151;
    colvals[1449] = 160;
    colvals[1450] = 162;
    colvals[1451] = 166;
    colvals[1452] = 167;
    colvals[1453] = 173;
    colvals[1454] = 174;
    colvals[1455] = 176;
    colvals[1456] = 178;
    colvals[1457] = 180;
    colvals[1458] = 182;
    colvals[1459] = 186;
    colvals[1460] = 188;
    colvals[1461] = 199;
    colvals[1462] = 205;
    colvals[1463] = 206;
    colvals[1464] = 209;
    colvals[1465] = 213;
    colvals[1466] = 124;
    colvals[1467] = 131;
    colvals[1468] = 140;
    colvals[1469] = 141;
    colvals[1470] = 142;
    colvals[1471] = 148;
    colvals[1472] = 150;
    colvals[1473] = 151;
    colvals[1474] = 160;
    colvals[1475] = 162;
    colvals[1476] = 166;
    colvals[1477] = 167;
    colvals[1478] = 173;
    colvals[1479] = 174;
    colvals[1480] = 176;
    colvals[1481] = 178;
    colvals[1482] = 180;
    colvals[1483] = 182;
    colvals[1484] = 186;
    colvals[1485] = 188;
    colvals[1486] = 199;
    colvals[1487] = 205;
    colvals[1488] = 206;
    colvals[1489] = 209;
    colvals[1490] = 213;
    colvals[1491] = 92;
    colvals[1492] = 97;
    colvals[1493] = 121;
    colvals[1494] = 136;
    colvals[1495] = 137;
    colvals[1496] = 140;
    colvals[1497] = 142;
    colvals[1498] = 147;
    colvals[1499] = 148;
    colvals[1500] = 157;
    colvals[1501] = 161;
    colvals[1502] = 162;
    colvals[1503] = 166;
    colvals[1504] = 172;
    colvals[1505] = 177;
    colvals[1506] = 179;
    colvals[1507] = 188;
    colvals[1508] = 196;
    colvals[1509] = 198;
    colvals[1510] = 199;
    colvals[1511] = 200;
    colvals[1512] = 201;
    colvals[1513] = 202;
    colvals[1514] = 203;
    colvals[1515] = 206;
    colvals[1516] = 207;
    colvals[1517] = 211;
    colvals[1518] = 213;
    colvals[1519] = 214;
    colvals[1520] = 80;
    colvals[1521] = 143;
    colvals[1522] = 145;
    colvals[1523] = 151;
    colvals[1524] = 154;
    colvals[1525] = 157;
    colvals[1526] = 161;
    colvals[1527] = 165;
    colvals[1528] = 166;
    colvals[1529] = 167;
    colvals[1530] = 169;
    colvals[1531] = 171;
    colvals[1532] = 172;
    colvals[1533] = 178;
    colvals[1534] = 179;
    colvals[1535] = 183;
    colvals[1536] = 184;
    colvals[1537] = 186;
    colvals[1538] = 187;
    colvals[1539] = 191;
    colvals[1540] = 192;
    colvals[1541] = 194;
    colvals[1542] = 195;
    colvals[1543] = 198;
    colvals[1544] = 199;
    colvals[1545] = 200;
    colvals[1546] = 201;
    colvals[1547] = 202;
    colvals[1548] = 203;
    colvals[1549] = 206;
    colvals[1550] = 207;
    colvals[1551] = 208;
    colvals[1552] = 210;
    colvals[1553] = 212;
    colvals[1554] = 213;
    colvals[1555] = 144;
    colvals[1556] = 154;
    colvals[1557] = 159;
    colvals[1558] = 163;
    colvals[1559] = 164;
    colvals[1560] = 165;
    colvals[1561] = 166;
    colvals[1562] = 171;
    colvals[1563] = 181;
    colvals[1564] = 183;
    colvals[1565] = 184;
    colvals[1566] = 187;
    colvals[1567] = 189;
    colvals[1568] = 190;
    colvals[1569] = 192;
    colvals[1570] = 193;
    colvals[1571] = 194;
    colvals[1572] = 195;
    colvals[1573] = 196;
    colvals[1574] = 198;
    colvals[1575] = 200;
    colvals[1576] = 201;
    colvals[1577] = 205;
    colvals[1578] = 207;
    colvals[1579] = 208;
    colvals[1580] = 210;
    colvals[1581] = 211;
    colvals[1582] = 212;
    colvals[1583] = 213;
    colvals[1584] = 107;
    colvals[1585] = 114;
    colvals[1586] = 117;
    colvals[1587] = 145;
    colvals[1588] = 151;
    colvals[1589] = 154;
    colvals[1590] = 159;
    colvals[1591] = 165;
    colvals[1592] = 167;
    colvals[1593] = 171;
    colvals[1594] = 183;
    colvals[1595] = 184;
    colvals[1596] = 186;
    colvals[1597] = 187;
    colvals[1598] = 189;
    colvals[1599] = 190;
    colvals[1600] = 192;
    colvals[1601] = 193;
    colvals[1602] = 194;
    colvals[1603] = 195;
    colvals[1604] = 198;
    colvals[1605] = 199;
    colvals[1606] = 200;
    colvals[1607] = 201;
    colvals[1608] = 202;
    colvals[1609] = 203;
    colvals[1610] = 207;
    colvals[1611] = 208;
    colvals[1612] = 210;
    colvals[1613] = 211;
    colvals[1614] = 212;
    colvals[1615] = 213;
    colvals[1616] = 214;
    colvals[1617] = 117;
    colvals[1618] = 144;
    colvals[1619] = 146;
    colvals[1620] = 149;
    colvals[1621] = 151;
    colvals[1622] = 154;
    colvals[1623] = 159;
    colvals[1624] = 163;
    colvals[1625] = 164;
    colvals[1626] = 165;
    colvals[1627] = 167;
    colvals[1628] = 168;
    colvals[1629] = 171;
    colvals[1630] = 179;
    colvals[1631] = 181;
    colvals[1632] = 183;
    colvals[1633] = 184;
    colvals[1634] = 187;
    colvals[1635] = 190;
    colvals[1636] = 191;
    colvals[1637] = 192;
    colvals[1638] = 193;
    colvals[1639] = 194;
    colvals[1640] = 195;
    colvals[1641] = 196;
    colvals[1642] = 200;
    colvals[1643] = 201;
    colvals[1644] = 202;
    colvals[1645] = 205;
    colvals[1646] = 207;
    colvals[1647] = 208;
    colvals[1648] = 210;
    colvals[1649] = 211;
    colvals[1650] = 212;
    colvals[1651] = 213;
    colvals[1652] = 36;
    colvals[1653] = 86;
    colvals[1654] = 109;
    colvals[1655] = 119;
    colvals[1656] = 136;
    colvals[1657] = 138;
    colvals[1658] = 139;
    colvals[1659] = 142;
    colvals[1660] = 147;
    colvals[1661] = 151;
    colvals[1662] = 160;
    colvals[1663] = 167;
    colvals[1664] = 172;
    colvals[1665] = 175;
    colvals[1666] = 177;
    colvals[1667] = 178;
    colvals[1668] = 180;
    colvals[1669] = 186;
    colvals[1670] = 188;
    colvals[1671] = 192;
    colvals[1672] = 195;
    colvals[1673] = 196;
    colvals[1674] = 199;
    colvals[1675] = 200;
    colvals[1676] = 201;
    colvals[1677] = 203;
    colvals[1678] = 205;
    colvals[1679] = 206;
    colvals[1680] = 207;
    colvals[1681] = 208;
    colvals[1682] = 209;
    colvals[1683] = 210;
    colvals[1684] = 213;
    colvals[1685] = 214;
    colvals[1686] = 119;
    colvals[1687] = 135;
    colvals[1688] = 138;
    colvals[1689] = 140;
    colvals[1690] = 148;
    colvals[1691] = 151;
    colvals[1692] = 152;
    colvals[1693] = 154;
    colvals[1694] = 156;
    colvals[1695] = 161;
    colvals[1696] = 162;
    colvals[1697] = 166;
    colvals[1698] = 167;
    colvals[1699] = 168;
    colvals[1700] = 172;
    colvals[1701] = 177;
    colvals[1702] = 178;
    colvals[1703] = 179;
    colvals[1704] = 182;
    colvals[1705] = 188;
    colvals[1706] = 190;
    colvals[1707] = 192;
    colvals[1708] = 194;
    colvals[1709] = 196;
    colvals[1710] = 199;
    colvals[1711] = 200;
    colvals[1712] = 202;
    colvals[1713] = 203;
    colvals[1714] = 205;
    colvals[1715] = 206;
    colvals[1716] = 207;
    colvals[1717] = 208;
    colvals[1718] = 209;
    colvals[1719] = 210;
    colvals[1720] = 212;
    colvals[1721] = 213;
    colvals[1722] = 214;
    colvals[1723] = 144;
    colvals[1724] = 149;
    colvals[1725] = 154;
    colvals[1726] = 159;
    colvals[1727] = 163;
    colvals[1728] = 164;
    colvals[1729] = 165;
    colvals[1730] = 166;
    colvals[1731] = 168;
    colvals[1732] = 171;
    colvals[1733] = 176;
    colvals[1734] = 178;
    colvals[1735] = 179;
    colvals[1736] = 181;
    colvals[1737] = 183;
    colvals[1738] = 184;
    colvals[1739] = 187;
    colvals[1740] = 189;
    colvals[1741] = 190;
    colvals[1742] = 191;
    colvals[1743] = 192;
    colvals[1744] = 193;
    colvals[1745] = 194;
    colvals[1746] = 195;
    colvals[1747] = 196;
    colvals[1748] = 198;
    colvals[1749] = 200;
    colvals[1750] = 201;
    colvals[1751] = 202;
    colvals[1752] = 205;
    colvals[1753] = 207;
    colvals[1754] = 208;
    colvals[1755] = 210;
    colvals[1756] = 211;
    colvals[1757] = 213;
    colvals[1758] = 85;
    colvals[1759] = 127;
    colvals[1760] = 130;
    colvals[1761] = 138;
    colvals[1762] = 140;
    colvals[1763] = 146;
    colvals[1764] = 149;
    colvals[1765] = 150;
    colvals[1766] = 154;
    colvals[1767] = 159;
    colvals[1768] = 163;
    colvals[1769] = 165;
    colvals[1770] = 171;
    colvals[1771] = 177;
    colvals[1772] = 179;
    colvals[1773] = 180;
    colvals[1774] = 183;
    colvals[1775] = 184;
    colvals[1776] = 187;
    colvals[1777] = 190;
    colvals[1778] = 191;
    colvals[1779] = 192;
    colvals[1780] = 193;
    colvals[1781] = 195;
    colvals[1782] = 196;
    colvals[1783] = 200;
    colvals[1784] = 201;
    colvals[1785] = 205;
    colvals[1786] = 207;
    colvals[1787] = 208;
    colvals[1788] = 210;
    colvals[1789] = 211;
    colvals[1790] = 212;
    colvals[1791] = 213;
    colvals[1792] = 214;
    colvals[1793] = 140;
    colvals[1794] = 145;
    colvals[1795] = 147;
    colvals[1796] = 151;
    colvals[1797] = 164;
    colvals[1798] = 165;
    colvals[1799] = 167;
    colvals[1800] = 171;
    colvals[1801] = 177;
    colvals[1802] = 181;
    colvals[1803] = 183;
    colvals[1804] = 184;
    colvals[1805] = 187;
    colvals[1806] = 189;
    colvals[1807] = 190;
    colvals[1808] = 191;
    colvals[1809] = 192;
    colvals[1810] = 193;
    colvals[1811] = 194;
    colvals[1812] = 195;
    colvals[1813] = 196;
    colvals[1814] = 198;
    colvals[1815] = 200;
    colvals[1816] = 201;
    colvals[1817] = 202;
    colvals[1818] = 203;
    colvals[1819] = 207;
    colvals[1820] = 208;
    colvals[1821] = 210;
    colvals[1822] = 211;
    colvals[1823] = 212;
    colvals[1824] = 213;
    colvals[1825] = 51;
    colvals[1826] = 84;
    colvals[1827] = 87;
    colvals[1828] = 96;
    colvals[1829] = 97;
    colvals[1830] = 98;
    colvals[1831] = 109;
    colvals[1832] = 115;
    colvals[1833] = 118;
    colvals[1834] = 135;
    colvals[1835] = 136;
    colvals[1836] = 139;
    colvals[1837] = 140;
    colvals[1838] = 147;
    colvals[1839] = 148;
    colvals[1840] = 152;
    colvals[1841] = 156;
    colvals[1842] = 161;
    colvals[1843] = 162;
    colvals[1844] = 167;
    colvals[1845] = 168;
    colvals[1846] = 176;
    colvals[1847] = 177;
    colvals[1848] = 178;
    colvals[1849] = 180;
    colvals[1850] = 185;
    colvals[1851] = 187;
    colvals[1852] = 188;
    colvals[1853] = 190;
    colvals[1854] = 191;
    colvals[1855] = 192;
    colvals[1856] = 194;
    colvals[1857] = 195;
    colvals[1858] = 196;
    colvals[1859] = 199;
    colvals[1860] = 200;
    colvals[1861] = 201;
    colvals[1862] = 202;
    colvals[1863] = 203;
    colvals[1864] = 205;
    colvals[1865] = 206;
    colvals[1866] = 207;
    colvals[1867] = 208;
    colvals[1868] = 209;
    colvals[1869] = 210;
    colvals[1870] = 212;
    colvals[1871] = 213;
    colvals[1872] = 214;
    colvals[1873] = 96;
    colvals[1874] = 117;
    colvals[1875] = 129;
    colvals[1876] = 135;
    colvals[1877] = 144;
    colvals[1878] = 146;
    colvals[1879] = 149;
    colvals[1880] = 150;
    colvals[1881] = 153;
    colvals[1882] = 155;
    colvals[1883] = 158;
    colvals[1884] = 159;
    colvals[1885] = 162;
    colvals[1886] = 164;
    colvals[1887] = 165;
    colvals[1888] = 167;
    colvals[1889] = 168;
    colvals[1890] = 169;
    colvals[1891] = 174;
    colvals[1892] = 177;
    colvals[1893] = 178;
    colvals[1894] = 179;
    colvals[1895] = 180;
    colvals[1896] = 182;
    colvals[1897] = 184;
    colvals[1898] = 186;
    colvals[1899] = 187;
    colvals[1900] = 190;
    colvals[1901] = 191;
    colvals[1902] = 192;
    colvals[1903] = 193;
    colvals[1904] = 195;
    colvals[1905] = 196;
    colvals[1906] = 197;
    colvals[1907] = 205;
    colvals[1908] = 209;
    colvals[1909] = 212;
    colvals[1910] = 213;
    colvals[1911] = 56;
    colvals[1912] = 137;
    colvals[1913] = 154;
    colvals[1914] = 157;
    colvals[1915] = 163;
    colvals[1916] = 165;
    colvals[1917] = 171;
    colvals[1918] = 177;
    colvals[1919] = 181;
    colvals[1920] = 183;
    colvals[1921] = 184;
    colvals[1922] = 187;
    colvals[1923] = 189;
    colvals[1924] = 190;
    colvals[1925] = 191;
    colvals[1926] = 192;
    colvals[1927] = 193;
    colvals[1928] = 194;
    colvals[1929] = 195;
    colvals[1930] = 196;
    colvals[1931] = 198;
    colvals[1932] = 201;
    colvals[1933] = 202;
    colvals[1934] = 203;
    colvals[1935] = 204;
    colvals[1936] = 205;
    colvals[1937] = 206;
    colvals[1938] = 207;
    colvals[1939] = 208;
    colvals[1940] = 210;
    colvals[1941] = 211;
    colvals[1942] = 212;
    colvals[1943] = 213;
    colvals[1944] = 214;
    colvals[1945] = 104;
    colvals[1946] = 134;
    colvals[1947] = 135;
    colvals[1948] = 137;
    colvals[1949] = 138;
    colvals[1950] = 142;
    colvals[1951] = 144;
    colvals[1952] = 146;
    colvals[1953] = 149;
    colvals[1954] = 150;
    colvals[1955] = 153;
    colvals[1956] = 155;
    colvals[1957] = 158;
    colvals[1958] = 159;
    colvals[1959] = 164;
    colvals[1960] = 165;
    colvals[1961] = 166;
    colvals[1962] = 167;
    colvals[1963] = 168;
    colvals[1964] = 169;
    colvals[1965] = 172;
    colvals[1966] = 174;
    colvals[1967] = 175;
    colvals[1968] = 177;
    colvals[1969] = 178;
    colvals[1970] = 179;
    colvals[1971] = 180;
    colvals[1972] = 186;
    colvals[1973] = 187;
    colvals[1974] = 190;
    colvals[1975] = 191;
    colvals[1976] = 192;
    colvals[1977] = 193;
    colvals[1978] = 195;
    colvals[1979] = 196;
    colvals[1980] = 197;
    colvals[1981] = 198;
    colvals[1982] = 199;
    colvals[1983] = 205;
    colvals[1984] = 206;
    colvals[1985] = 207;
    colvals[1986] = 209;
    colvals[1987] = 210;
    colvals[1988] = 213;
    colvals[1989] = 82;
    colvals[1990] = 91;
    colvals[1991] = 96;
    colvals[1992] = 97;
    colvals[1993] = 105;
    colvals[1994] = 108;
    colvals[1995] = 112;
    colvals[1996] = 120;
    colvals[1997] = 123;
    colvals[1998] = 124;
    colvals[1999] = 125;
    colvals[2000] = 126;
    colvals[2001] = 131;
    colvals[2002] = 133;
    colvals[2003] = 140;
    colvals[2004] = 148;
    colvals[2005] = 156;
    colvals[2006] = 157;
    colvals[2007] = 160;
    colvals[2008] = 162;
    colvals[2009] = 163;
    colvals[2010] = 166;
    colvals[2011] = 173;
    colvals[2012] = 174;
    colvals[2013] = 176;
    colvals[2014] = 178;
    colvals[2015] = 179;
    colvals[2016] = 182;
    colvals[2017] = 186;
    colvals[2018] = 188;
    colvals[2019] = 194;
    colvals[2020] = 195;
    colvals[2021] = 196;
    colvals[2022] = 197;
    colvals[2023] = 198;
    colvals[2024] = 199;
    colvals[2025] = 201;
    colvals[2026] = 202;
    colvals[2027] = 203;
    colvals[2028] = 205;
    colvals[2029] = 206;
    colvals[2030] = 207;
    colvals[2031] = 209;
    colvals[2032] = 210;
    colvals[2033] = 211;
    colvals[2034] = 213;
    colvals[2035] = 14;
    colvals[2036] = 68;
    colvals[2037] = 89;
    colvals[2038] = 128;
    colvals[2039] = 137;
    colvals[2040] = 138;
    colvals[2041] = 140;
    colvals[2042] = 142;
    colvals[2043] = 154;
    colvals[2044] = 156;
    colvals[2045] = 157;
    colvals[2046] = 158;
    colvals[2047] = 166;
    colvals[2048] = 171;
    colvals[2049] = 172;
    colvals[2050] = 177;
    colvals[2051] = 178;
    colvals[2052] = 182;
    colvals[2053] = 183;
    colvals[2054] = 184;
    colvals[2055] = 185;
    colvals[2056] = 189;
    colvals[2057] = 190;
    colvals[2058] = 192;
    colvals[2059] = 194;
    colvals[2060] = 195;
    colvals[2061] = 196;
    colvals[2062] = 198;
    colvals[2063] = 199;
    colvals[2064] = 201;
    colvals[2065] = 202;
    colvals[2066] = 203;
    colvals[2067] = 206;
    colvals[2068] = 207;
    colvals[2069] = 208;
    colvals[2070] = 212;
    colvals[2071] = 214;
    colvals[2072] = 68;
    colvals[2073] = 144;
    colvals[2074] = 145;
    colvals[2075] = 149;
    colvals[2076] = 151;
    colvals[2077] = 154;
    colvals[2078] = 157;
    colvals[2079] = 158;
    colvals[2080] = 159;
    colvals[2081] = 161;
    colvals[2082] = 163;
    colvals[2083] = 164;
    colvals[2084] = 165;
    colvals[2085] = 167;
    colvals[2086] = 171;
    colvals[2087] = 175;
    colvals[2088] = 179;
    colvals[2089] = 183;
    colvals[2090] = 184;
    colvals[2091] = 186;
    colvals[2092] = 187;
    colvals[2093] = 189;
    colvals[2094] = 190;
    colvals[2095] = 191;
    colvals[2096] = 192;
    colvals[2097] = 193;
    colvals[2098] = 194;
    colvals[2099] = 195;
    colvals[2100] = 196;
    colvals[2101] = 198;
    colvals[2102] = 199;
    colvals[2103] = 200;
    colvals[2104] = 201;
    colvals[2105] = 202;
    colvals[2106] = 205;
    colvals[2107] = 206;
    colvals[2108] = 208;
    colvals[2109] = 210;
    colvals[2110] = 211;
    colvals[2111] = 212;
    colvals[2112] = 213;
    colvals[2113] = 214;
    colvals[2114] = 9;
    colvals[2115] = 57;
    colvals[2116] = 100;
    colvals[2117] = 117;
    colvals[2118] = 135;
    colvals[2119] = 144;
    colvals[2120] = 146;
    colvals[2121] = 149;
    colvals[2122] = 150;
    colvals[2123] = 153;
    colvals[2124] = 155;
    colvals[2125] = 158;
    colvals[2126] = 159;
    colvals[2127] = 162;
    colvals[2128] = 164;
    colvals[2129] = 165;
    colvals[2130] = 168;
    colvals[2131] = 169;
    colvals[2132] = 171;
    colvals[2133] = 174;
    colvals[2134] = 177;
    colvals[2135] = 178;
    colvals[2136] = 179;
    colvals[2137] = 182;
    colvals[2138] = 184;
    colvals[2139] = 186;
    colvals[2140] = 189;
    colvals[2141] = 190;
    colvals[2142] = 195;
    colvals[2143] = 196;
    colvals[2144] = 197;
    colvals[2145] = 199;
    colvals[2146] = 202;
    colvals[2147] = 203;
    colvals[2148] = 205;
    colvals[2149] = 206;
    colvals[2150] = 208;
    colvals[2151] = 209;
    colvals[2152] = 213;
    colvals[2153] = 214;
    colvals[2154] = 91;
    colvals[2155] = 97;
    colvals[2156] = 104;
    colvals[2157] = 105;
    colvals[2158] = 108;
    colvals[2159] = 110;
    colvals[2160] = 112;
    colvals[2161] = 123;
    colvals[2162] = 124;
    colvals[2163] = 126;
    colvals[2164] = 127;
    colvals[2165] = 131;
    colvals[2166] = 133;
    colvals[2167] = 134;
    colvals[2168] = 140;
    colvals[2169] = 147;
    colvals[2170] = 148;
    colvals[2171] = 156;
    colvals[2172] = 160;
    colvals[2173] = 162;
    colvals[2174] = 166;
    colvals[2175] = 171;
    colvals[2176] = 173;
    colvals[2177] = 174;
    colvals[2178] = 176;
    colvals[2179] = 178;
    colvals[2180] = 182;
    colvals[2181] = 184;
    colvals[2182] = 186;
    colvals[2183] = 188;
    colvals[2184] = 195;
    colvals[2185] = 199;
    colvals[2186] = 200;
    colvals[2187] = 201;
    colvals[2188] = 202;
    colvals[2189] = 203;
    colvals[2190] = 206;
    colvals[2191] = 207;
    colvals[2192] = 208;
    colvals[2193] = 210;
    colvals[2194] = 211;
    colvals[2195] = 212;
    colvals[2196] = 213;
    colvals[2197] = 214;
    colvals[2198] = 78;
    colvals[2199] = 114;
    colvals[2200] = 121;
    colvals[2201] = 131;
    colvals[2202] = 136;
    colvals[2203] = 143;
    colvals[2204] = 145;
    colvals[2205] = 147;
    colvals[2206] = 151;
    colvals[2207] = 154;
    colvals[2208] = 161;
    colvals[2209] = 163;
    colvals[2210] = 165;
    colvals[2211] = 166;
    colvals[2212] = 167;
    colvals[2213] = 171;
    colvals[2214] = 172;
    colvals[2215] = 177;
    colvals[2216] = 183;
    colvals[2217] = 184;
    colvals[2218] = 186;
    colvals[2219] = 187;
    colvals[2220] = 190;
    colvals[2221] = 191;
    colvals[2222] = 192;
    colvals[2223] = 193;
    colvals[2224] = 194;
    colvals[2225] = 195;
    colvals[2226] = 196;
    colvals[2227] = 198;
    colvals[2228] = 199;
    colvals[2229] = 200;
    colvals[2230] = 201;
    colvals[2231] = 203;
    colvals[2232] = 206;
    colvals[2233] = 207;
    colvals[2234] = 208;
    colvals[2235] = 209;
    colvals[2236] = 210;
    colvals[2237] = 211;
    colvals[2238] = 212;
    colvals[2239] = 213;
    colvals[2240] = 214;
    colvals[2241] = 51;
    colvals[2242] = 140;
    colvals[2243] = 144;
    colvals[2244] = 146;
    colvals[2245] = 147;
    colvals[2246] = 148;
    colvals[2247] = 149;
    colvals[2248] = 150;
    colvals[2249] = 151;
    colvals[2250] = 152;
    colvals[2251] = 154;
    colvals[2252] = 156;
    colvals[2253] = 158;
    colvals[2254] = 159;
    colvals[2255] = 162;
    colvals[2256] = 164;
    colvals[2257] = 167;
    colvals[2258] = 168;
    colvals[2259] = 172;
    colvals[2260] = 174;
    colvals[2261] = 177;
    colvals[2262] = 178;
    colvals[2263] = 179;
    colvals[2264] = 186;
    colvals[2265] = 188;
    colvals[2266] = 191;
    colvals[2267] = 192;
    colvals[2268] = 193;
    colvals[2269] = 194;
    colvals[2270] = 195;
    colvals[2271] = 196;
    colvals[2272] = 200;
    colvals[2273] = 202;
    colvals[2274] = 203;
    colvals[2275] = 205;
    colvals[2276] = 206;
    colvals[2277] = 207;
    colvals[2278] = 208;
    colvals[2279] = 209;
    colvals[2280] = 210;
    colvals[2281] = 212;
    colvals[2282] = 213;
    colvals[2283] = 214;
    colvals[2284] = 29;
    colvals[2285] = 81;
    colvals[2286] = 92;
    colvals[2287] = 99;
    colvals[2288] = 101;
    colvals[2289] = 114;
    colvals[2290] = 117;
    colvals[2291] = 121;
    colvals[2292] = 122;
    colvals[2293] = 124;
    colvals[2294] = 130;
    colvals[2295] = 138;
    colvals[2296] = 142;
    colvals[2297] = 144;
    colvals[2298] = 146;
    colvals[2299] = 147;
    colvals[2300] = 149;
    colvals[2301] = 150;
    colvals[2302] = 154;
    colvals[2303] = 156;
    colvals[2304] = 158;
    colvals[2305] = 161;
    colvals[2306] = 163;
    colvals[2307] = 164;
    colvals[2308] = 167;
    colvals[2309] = 172;
    colvals[2310] = 175;
    colvals[2311] = 179;
    colvals[2312] = 184;
    colvals[2313] = 186;
    colvals[2314] = 191;
    colvals[2315] = 192;
    colvals[2316] = 194;
    colvals[2317] = 195;
    colvals[2318] = 196;
    colvals[2319] = 198;
    colvals[2320] = 199;
    colvals[2321] = 201;
    colvals[2322] = 202;
    colvals[2323] = 203;
    colvals[2324] = 205;
    colvals[2325] = 206;
    colvals[2326] = 207;
    colvals[2327] = 208;
    colvals[2328] = 210;
    colvals[2329] = 211;
    colvals[2330] = 213;
    colvals[2331] = 214;
    colvals[2332] = 145;
    colvals[2333] = 151;
    colvals[2334] = 154;
    colvals[2335] = 159;
    colvals[2336] = 161;
    colvals[2337] = 163;
    colvals[2338] = 164;
    colvals[2339] = 165;
    colvals[2340] = 167;
    colvals[2341] = 171;
    colvals[2342] = 172;
    colvals[2343] = 177;
    colvals[2344] = 181;
    colvals[2345] = 183;
    colvals[2346] = 184;
    colvals[2347] = 187;
    colvals[2348] = 189;
    colvals[2349] = 190;
    colvals[2350] = 192;
    colvals[2351] = 193;
    colvals[2352] = 194;
    colvals[2353] = 195;
    colvals[2354] = 196;
    colvals[2355] = 198;
    colvals[2356] = 200;
    colvals[2357] = 201;
    colvals[2358] = 202;
    colvals[2359] = 203;
    colvals[2360] = 206;
    colvals[2361] = 207;
    colvals[2362] = 208;
    colvals[2363] = 210;
    colvals[2364] = 211;
    colvals[2365] = 212;
    colvals[2366] = 213;
    colvals[2367] = 122;
    colvals[2368] = 143;
    colvals[2369] = 144;
    colvals[2370] = 145;
    colvals[2371] = 146;
    colvals[2372] = 149;
    colvals[2373] = 150;
    colvals[2374] = 151;
    colvals[2375] = 153;
    colvals[2376] = 154;
    colvals[2377] = 155;
    colvals[2378] = 158;
    colvals[2379] = 161;
    colvals[2380] = 164;
    colvals[2381] = 165;
    colvals[2382] = 166;
    colvals[2383] = 167;
    colvals[2384] = 168;
    colvals[2385] = 169;
    colvals[2386] = 170;
    colvals[2387] = 172;
    colvals[2388] = 174;
    colvals[2389] = 176;
    colvals[2390] = 177;
    colvals[2391] = 178;
    colvals[2392] = 179;
    colvals[2393] = 182;
    colvals[2394] = 184;
    colvals[2395] = 185;
    colvals[2396] = 186;
    colvals[2397] = 187;
    colvals[2398] = 189;
    colvals[2399] = 190;
    colvals[2400] = 191;
    colvals[2401] = 192;
    colvals[2402] = 193;
    colvals[2403] = 194;
    colvals[2404] = 195;
    colvals[2405] = 196;
    colvals[2406] = 197;
    colvals[2407] = 199;
    colvals[2408] = 200;
    colvals[2409] = 201;
    colvals[2410] = 203;
    colvals[2411] = 205;
    colvals[2412] = 206;
    colvals[2413] = 207;
    colvals[2414] = 208;
    colvals[2415] = 209;
    colvals[2416] = 210;
    colvals[2417] = 212;
    colvals[2418] = 213;
    colvals[2419] = 214;
    colvals[2420] = 66;
    colvals[2421] = 92;
    colvals[2422] = 134;
    colvals[2423] = 138;
    colvals[2424] = 140;
    colvals[2425] = 145;
    colvals[2426] = 151;
    colvals[2427] = 154;
    colvals[2428] = 156;
    colvals[2429] = 157;
    colvals[2430] = 158;
    colvals[2431] = 161;
    colvals[2432] = 163;
    colvals[2433] = 164;
    colvals[2434] = 165;
    colvals[2435] = 166;
    colvals[2436] = 167;
    colvals[2437] = 172;
    colvals[2438] = 177;
    colvals[2439] = 178;
    colvals[2440] = 179;
    colvals[2441] = 183;
    colvals[2442] = 184;
    colvals[2443] = 187;
    colvals[2444] = 190;
    colvals[2445] = 192;
    colvals[2446] = 194;
    colvals[2447] = 195;
    colvals[2448] = 196;
    colvals[2449] = 198;
    colvals[2450] = 200;
    colvals[2451] = 201;
    colvals[2452] = 202;
    colvals[2453] = 203;
    colvals[2454] = 206;
    colvals[2455] = 207;
    colvals[2456] = 208;
    colvals[2457] = 213;
    colvals[2458] = 107;
    colvals[2459] = 118;
    colvals[2460] = 134;
    colvals[2461] = 140;
    colvals[2462] = 147;
    colvals[2463] = 151;
    colvals[2464] = 163;
    colvals[2465] = 165;
    colvals[2466] = 167;
    colvals[2467] = 171;
    colvals[2468] = 177;
    colvals[2469] = 181;
    colvals[2470] = 183;
    colvals[2471] = 184;
    colvals[2472] = 187;
    colvals[2473] = 189;
    colvals[2474] = 190;
    colvals[2475] = 191;
    colvals[2476] = 192;
    colvals[2477] = 193;
    colvals[2478] = 194;
    colvals[2479] = 195;
    colvals[2480] = 196;
    colvals[2481] = 198;
    colvals[2482] = 201;
    colvals[2483] = 202;
    colvals[2484] = 203;
    colvals[2485] = 210;
    colvals[2486] = 211;
    colvals[2487] = 212;
    colvals[2488] = 213;
    colvals[2489] = 57;
    colvals[2490] = 143;
    colvals[2491] = 144;
    colvals[2492] = 145;
    colvals[2493] = 146;
    colvals[2494] = 149;
    colvals[2495] = 150;
    colvals[2496] = 151;
    colvals[2497] = 154;
    colvals[2498] = 158;
    colvals[2499] = 159;
    colvals[2500] = 161;
    colvals[2501] = 164;
    colvals[2502] = 165;
    colvals[2503] = 166;
    colvals[2504] = 167;
    colvals[2505] = 168;
    colvals[2506] = 171;
    colvals[2507] = 172;
    colvals[2508] = 177;
    colvals[2509] = 178;
    colvals[2510] = 179;
    colvals[2511] = 183;
    colvals[2512] = 184;
    colvals[2513] = 187;
    colvals[2514] = 190;
    colvals[2515] = 192;
    colvals[2516] = 193;
    colvals[2517] = 194;
    colvals[2518] = 195;
    colvals[2519] = 196;
    colvals[2520] = 198;
    colvals[2521] = 200;
    colvals[2522] = 202;
    colvals[2523] = 203;
    colvals[2524] = 205;
    colvals[2525] = 206;
    colvals[2526] = 207;
    colvals[2527] = 209;
    colvals[2528] = 210;
    colvals[2529] = 212;
    colvals[2530] = 213;
    colvals[2531] = 64;
    colvals[2532] = 95;
    colvals[2533] = 128;
    colvals[2534] = 137;
    colvals[2535] = 143;
    colvals[2536] = 144;
    colvals[2537] = 145;
    colvals[2538] = 146;
    colvals[2539] = 149;
    colvals[2540] = 150;
    colvals[2541] = 151;
    colvals[2542] = 154;
    colvals[2543] = 157;
    colvals[2544] = 158;
    colvals[2545] = 159;
    colvals[2546] = 161;
    colvals[2547] = 164;
    colvals[2548] = 165;
    colvals[2549] = 167;
    colvals[2550] = 168;
    colvals[2551] = 169;
    colvals[2552] = 171;
    colvals[2553] = 172;
    colvals[2554] = 174;
    colvals[2555] = 175;
    colvals[2556] = 178;
    colvals[2557] = 179;
    colvals[2558] = 180;
    colvals[2559] = 183;
    colvals[2560] = 184;
    colvals[2561] = 185;
    colvals[2562] = 186;
    colvals[2563] = 187;
    colvals[2564] = 190;
    colvals[2565] = 191;
    colvals[2566] = 192;
    colvals[2567] = 193;
    colvals[2568] = 194;
    colvals[2569] = 195;
    colvals[2570] = 196;
    colvals[2571] = 197;
    colvals[2572] = 199;
    colvals[2573] = 200;
    colvals[2574] = 202;
    colvals[2575] = 203;
    colvals[2576] = 205;
    colvals[2577] = 206;
    colvals[2578] = 207;
    colvals[2579] = 208;
    colvals[2580] = 209;
    colvals[2581] = 210;
    colvals[2582] = 212;
    colvals[2583] = 213;
    colvals[2584] = 75;
    colvals[2585] = 79;
    colvals[2586] = 96;
    colvals[2587] = 113;
    colvals[2588] = 125;
    colvals[2589] = 126;
    colvals[2590] = 130;
    colvals[2591] = 132;
    colvals[2592] = 135;
    colvals[2593] = 138;
    colvals[2594] = 144;
    colvals[2595] = 146;
    colvals[2596] = 148;
    colvals[2597] = 149;
    colvals[2598] = 150;
    colvals[2599] = 153;
    colvals[2600] = 155;
    colvals[2601] = 162;
    colvals[2602] = 164;
    colvals[2603] = 165;
    colvals[2604] = 168;
    colvals[2605] = 169;
    colvals[2606] = 170;
    colvals[2607] = 174;
    colvals[2608] = 175;
    colvals[2609] = 176;
    colvals[2610] = 177;
    colvals[2611] = 178;
    colvals[2612] = 179;
    colvals[2613] = 180;
    colvals[2614] = 182;
    colvals[2615] = 186;
    colvals[2616] = 187;
    colvals[2617] = 190;
    colvals[2618] = 191;
    colvals[2619] = 192;
    colvals[2620] = 195;
    colvals[2621] = 196;
    colvals[2622] = 197;
    colvals[2623] = 201;
    colvals[2624] = 205;
    colvals[2625] = 209;
    colvals[2626] = 210;
    colvals[2627] = 212;
    colvals[2628] = 213;
    colvals[2629] = 13;
    colvals[2630] = 64;
    colvals[2631] = 95;
    colvals[2632] = 107;
    colvals[2633] = 117;
    colvals[2634] = 144;
    colvals[2635] = 145;
    colvals[2636] = 146;
    colvals[2637] = 149;
    colvals[2638] = 150;
    colvals[2639] = 151;
    colvals[2640] = 154;
    colvals[2641] = 157;
    colvals[2642] = 158;
    colvals[2643] = 159;
    colvals[2644] = 160;
    colvals[2645] = 161;
    colvals[2646] = 164;
    colvals[2647] = 165;
    colvals[2648] = 167;
    colvals[2649] = 168;
    colvals[2650] = 169;
    colvals[2651] = 171;
    colvals[2652] = 172;
    colvals[2653] = 174;
    colvals[2654] = 177;
    colvals[2655] = 178;
    colvals[2656] = 179;
    colvals[2657] = 182;
    colvals[2658] = 184;
    colvals[2659] = 186;
    colvals[2660] = 189;
    colvals[2661] = 193;
    colvals[2662] = 194;
    colvals[2663] = 196;
    colvals[2664] = 198;
    colvals[2665] = 199;
    colvals[2666] = 200;
    colvals[2667] = 201;
    colvals[2668] = 202;
    colvals[2669] = 203;
    colvals[2670] = 205;
    colvals[2671] = 206;
    colvals[2672] = 207;
    colvals[2673] = 208;
    colvals[2674] = 209;
    colvals[2675] = 210;
    colvals[2676] = 212;
    colvals[2677] = 213;
    colvals[2678] = 214;
    colvals[2679] = 101;
    colvals[2680] = 114;
    colvals[2681] = 121;
    colvals[2682] = 133;
    colvals[2683] = 134;
    colvals[2684] = 136;
    colvals[2685] = 137;
    colvals[2686] = 145;
    colvals[2687] = 147;
    colvals[2688] = 151;
    colvals[2689] = 157;
    colvals[2690] = 161;
    colvals[2691] = 163;
    colvals[2692] = 165;
    colvals[2693] = 166;
    colvals[2694] = 167;
    colvals[2695] = 171;
    colvals[2696] = 172;
    colvals[2697] = 177;
    colvals[2698] = 179;
    colvals[2699] = 181;
    colvals[2700] = 183;
    colvals[2701] = 184;
    colvals[2702] = 186;
    colvals[2703] = 187;
    colvals[2704] = 189;
    colvals[2705] = 190;
    colvals[2706] = 191;
    colvals[2707] = 192;
    colvals[2708] = 193;
    colvals[2709] = 194;
    colvals[2710] = 195;
    colvals[2711] = 196;
    colvals[2712] = 198;
    colvals[2713] = 199;
    colvals[2714] = 201;
    colvals[2715] = 203;
    colvals[2716] = 206;
    colvals[2717] = 207;
    colvals[2718] = 208;
    colvals[2719] = 210;
    colvals[2720] = 211;
    colvals[2721] = 212;
    colvals[2722] = 213;
    colvals[2723] = 214;
    colvals[2724] = 88;
    colvals[2725] = 101;
    colvals[2726] = 102;
    colvals[2727] = 122;
    colvals[2728] = 124;
    colvals[2729] = 134;
    colvals[2730] = 140;
    colvals[2731] = 143;
    colvals[2732] = 145;
    colvals[2733] = 148;
    colvals[2734] = 149;
    colvals[2735] = 151;
    colvals[2736] = 154;
    colvals[2737] = 156;
    colvals[2738] = 158;
    colvals[2739] = 161;
    colvals[2740] = 162;
    colvals[2741] = 163;
    colvals[2742] = 164;
    colvals[2743] = 166;
    colvals[2744] = 167;
    colvals[2745] = 168;
    colvals[2746] = 169;
    colvals[2747] = 172;
    colvals[2748] = 173;
    colvals[2749] = 174;
    colvals[2750] = 175;
    colvals[2751] = 176;
    colvals[2752] = 178;
    colvals[2753] = 179;
    colvals[2754] = 180;
    colvals[2755] = 181;
    colvals[2756] = 182;
    colvals[2757] = 186;
    colvals[2758] = 187;
    colvals[2759] = 188;
    colvals[2760] = 189;
    colvals[2761] = 190;
    colvals[2762] = 193;
    colvals[2763] = 194;
    colvals[2764] = 198;
    colvals[2765] = 199;
    colvals[2766] = 202;
    colvals[2767] = 203;
    colvals[2768] = 205;
    colvals[2769] = 206;
    colvals[2770] = 207;
    colvals[2771] = 211;
    colvals[2772] = 213;
    colvals[2773] = 134;
    colvals[2774] = 138;
    colvals[2775] = 140;
    colvals[2776] = 144;
    colvals[2777] = 145;
    colvals[2778] = 146;
    colvals[2779] = 149;
    colvals[2780] = 150;
    colvals[2781] = 151;
    colvals[2782] = 154;
    colvals[2783] = 156;
    colvals[2784] = 158;
    colvals[2785] = 159;
    colvals[2786] = 161;
    colvals[2787] = 163;
    colvals[2788] = 164;
    colvals[2789] = 165;
    colvals[2790] = 166;
    colvals[2791] = 167;
    colvals[2792] = 168;
    colvals[2793] = 171;
    colvals[2794] = 172;
    colvals[2795] = 174;
    colvals[2796] = 175;
    colvals[2797] = 178;
    colvals[2798] = 179;
    colvals[2799] = 180;
    colvals[2800] = 182;
    colvals[2801] = 183;
    colvals[2802] = 184;
    colvals[2803] = 186;
    colvals[2804] = 187;
    colvals[2805] = 190;
    colvals[2806] = 191;
    colvals[2807] = 192;
    colvals[2808] = 193;
    colvals[2809] = 194;
    colvals[2810] = 195;
    colvals[2811] = 196;
    colvals[2812] = 198;
    colvals[2813] = 199;
    colvals[2814] = 200;
    colvals[2815] = 201;
    colvals[2816] = 203;
    colvals[2817] = 205;
    colvals[2818] = 206;
    colvals[2819] = 207;
    colvals[2820] = 209;
    colvals[2821] = 210;
    colvals[2822] = 213;
    colvals[2823] = 78;
    colvals[2824] = 93;
    colvals[2825] = 126;
    colvals[2826] = 137;
    colvals[2827] = 138;
    colvals[2828] = 143;
    colvals[2829] = 144;
    colvals[2830] = 145;
    colvals[2831] = 146;
    colvals[2832] = 147;
    colvals[2833] = 149;
    colvals[2834] = 150;
    colvals[2835] = 151;
    colvals[2836] = 153;
    colvals[2837] = 154;
    colvals[2838] = 155;
    colvals[2839] = 158;
    colvals[2840] = 161;
    colvals[2841] = 162;
    colvals[2842] = 163;
    colvals[2843] = 164;
    colvals[2844] = 166;
    colvals[2845] = 167;
    colvals[2846] = 168;
    colvals[2847] = 169;
    colvals[2848] = 172;
    colvals[2849] = 174;
    colvals[2850] = 175;
    colvals[2851] = 177;
    colvals[2852] = 178;
    colvals[2853] = 179;
    colvals[2854] = 180;
    colvals[2855] = 184;
    colvals[2856] = 186;
    colvals[2857] = 190;
    colvals[2858] = 191;
    colvals[2859] = 192;
    colvals[2860] = 194;
    colvals[2861] = 195;
    colvals[2862] = 196;
    colvals[2863] = 197;
    colvals[2864] = 198;
    colvals[2865] = 199;
    colvals[2866] = 200;
    colvals[2867] = 202;
    colvals[2868] = 203;
    colvals[2869] = 205;
    colvals[2870] = 206;
    colvals[2871] = 207;
    colvals[2872] = 208;
    colvals[2873] = 209;
    colvals[2874] = 210;
    colvals[2875] = 212;
    colvals[2876] = 213;
    colvals[2877] = 214;
    colvals[2878] = 119;
    colvals[2879] = 138;
    colvals[2880] = 140;
    colvals[2881] = 142;
    colvals[2882] = 144;
    colvals[2883] = 146;
    colvals[2884] = 148;
    colvals[2885] = 149;
    colvals[2886] = 150;
    colvals[2887] = 151;
    colvals[2888] = 153;
    colvals[2889] = 154;
    colvals[2890] = 155;
    colvals[2891] = 156;
    colvals[2892] = 158;
    colvals[2893] = 161;
    colvals[2894] = 162;
    colvals[2895] = 164;
    colvals[2896] = 165;
    colvals[2897] = 166;
    colvals[2898] = 167;
    colvals[2899] = 168;
    colvals[2900] = 169;
    colvals[2901] = 172;
    colvals[2902] = 174;
    colvals[2903] = 176;
    colvals[2904] = 177;
    colvals[2905] = 178;
    colvals[2906] = 179;
    colvals[2907] = 182;
    colvals[2908] = 183;
    colvals[2909] = 184;
    colvals[2910] = 186;
    colvals[2911] = 187;
    colvals[2912] = 188;
    colvals[2913] = 190;
    colvals[2914] = 191;
    colvals[2915] = 192;
    colvals[2916] = 194;
    colvals[2917] = 195;
    colvals[2918] = 196;
    colvals[2919] = 197;
    colvals[2920] = 199;
    colvals[2921] = 201;
    colvals[2922] = 203;
    colvals[2923] = 205;
    colvals[2924] = 206;
    colvals[2925] = 207;
    colvals[2926] = 209;
    colvals[2927] = 210;
    colvals[2928] = 212;
    colvals[2929] = 213;
    colvals[2930] = 48;
    colvals[2931] = 96;
    colvals[2932] = 98;
    colvals[2933] = 119;
    colvals[2934] = 129;
    colvals[2935] = 135;
    colvals[2936] = 138;
    colvals[2937] = 140;
    colvals[2938] = 142;
    colvals[2939] = 148;
    colvals[2940] = 150;
    colvals[2941] = 151;
    colvals[2942] = 152;
    colvals[2943] = 153;
    colvals[2944] = 154;
    colvals[2945] = 155;
    colvals[2946] = 156;
    colvals[2947] = 159;
    colvals[2948] = 161;
    colvals[2949] = 162;
    colvals[2950] = 166;
    colvals[2951] = 167;
    colvals[2952] = 168;
    colvals[2953] = 172;
    colvals[2954] = 175;
    colvals[2955] = 176;
    colvals[2956] = 177;
    colvals[2957] = 178;
    colvals[2958] = 179;
    colvals[2959] = 180;
    colvals[2960] = 182;
    colvals[2961] = 185;
    colvals[2962] = 186;
    colvals[2963] = 188;
    colvals[2964] = 190;
    colvals[2965] = 192;
    colvals[2966] = 193;
    colvals[2967] = 194;
    colvals[2968] = 196;
    colvals[2969] = 197;
    colvals[2970] = 199;
    colvals[2971] = 200;
    colvals[2972] = 201;
    colvals[2973] = 203;
    colvals[2974] = 205;
    colvals[2975] = 206;
    colvals[2976] = 207;
    colvals[2977] = 209;
    colvals[2978] = 210;
    colvals[2979] = 212;
    colvals[2980] = 213;
    colvals[2981] = 214;
    colvals[2982] = 121;
    colvals[2983] = 140;
    colvals[2984] = 144;
    colvals[2985] = 146;
    colvals[2986] = 147;
    colvals[2987] = 149;
    colvals[2988] = 150;
    colvals[2989] = 151;
    colvals[2990] = 154;
    colvals[2991] = 156;
    colvals[2992] = 157;
    colvals[2993] = 158;
    colvals[2994] = 159;
    colvals[2995] = 161;
    colvals[2996] = 164;
    colvals[2997] = 165;
    colvals[2998] = 167;
    colvals[2999] = 171;
    colvals[3000] = 172;
    colvals[3001] = 177;
    colvals[3002] = 178;
    colvals[3003] = 179;
    colvals[3004] = 183;
    colvals[3005] = 184;
    colvals[3006] = 187;
    colvals[3007] = 190;
    colvals[3008] = 191;
    colvals[3009] = 192;
    colvals[3010] = 193;
    colvals[3011] = 194;
    colvals[3012] = 195;
    colvals[3013] = 196;
    colvals[3014] = 198;
    colvals[3015] = 200;
    colvals[3016] = 201;
    colvals[3017] = 202;
    colvals[3018] = 203;
    colvals[3019] = 205;
    colvals[3020] = 206;
    colvals[3021] = 207;
    colvals[3022] = 208;
    colvals[3023] = 209;
    colvals[3024] = 210;
    colvals[3025] = 211;
    colvals[3026] = 212;
    colvals[3027] = 213;
    colvals[3028] = 123;
    colvals[3029] = 133;
    colvals[3030] = 134;
    colvals[3031] = 143;
    colvals[3032] = 144;
    colvals[3033] = 145;
    colvals[3034] = 146;
    colvals[3035] = 151;
    colvals[3036] = 154;
    colvals[3037] = 156;
    colvals[3038] = 159;
    colvals[3039] = 161;
    colvals[3040] = 163;
    colvals[3041] = 164;
    colvals[3042] = 165;
    colvals[3043] = 167;
    colvals[3044] = 171;
    colvals[3045] = 172;
    colvals[3046] = 177;
    colvals[3047] = 178;
    colvals[3048] = 179;
    colvals[3049] = 181;
    colvals[3050] = 183;
    colvals[3051] = 184;
    colvals[3052] = 187;
    colvals[3053] = 189;
    colvals[3054] = 190;
    colvals[3055] = 191;
    colvals[3056] = 192;
    colvals[3057] = 193;
    colvals[3058] = 194;
    colvals[3059] = 195;
    colvals[3060] = 196;
    colvals[3061] = 198;
    colvals[3062] = 200;
    colvals[3063] = 201;
    colvals[3064] = 202;
    colvals[3065] = 203;
    colvals[3066] = 205;
    colvals[3067] = 206;
    colvals[3068] = 207;
    colvals[3069] = 208;
    colvals[3070] = 210;
    colvals[3071] = 211;
    colvals[3072] = 212;
    colvals[3073] = 213;
    colvals[3074] = 104;
    colvals[3075] = 117;
    colvals[3076] = 127;
    colvals[3077] = 134;
    colvals[3078] = 136;
    colvals[3079] = 137;
    colvals[3080] = 138;
    colvals[3081] = 140;
    colvals[3082] = 144;
    colvals[3083] = 146;
    colvals[3084] = 147;
    colvals[3085] = 149;
    colvals[3086] = 150;
    colvals[3087] = 151;
    colvals[3088] = 152;
    colvals[3089] = 153;
    colvals[3090] = 154;
    colvals[3091] = 158;
    colvals[3092] = 164;
    colvals[3093] = 167;
    colvals[3094] = 168;
    colvals[3095] = 169;
    colvals[3096] = 172;
    colvals[3097] = 174;
    colvals[3098] = 175;
    colvals[3099] = 176;
    colvals[3100] = 177;
    colvals[3101] = 178;
    colvals[3102] = 179;
    colvals[3103] = 180;
    colvals[3104] = 184;
    colvals[3105] = 185;
    colvals[3106] = 186;
    colvals[3107] = 187;
    colvals[3108] = 190;
    colvals[3109] = 191;
    colvals[3110] = 192;
    colvals[3111] = 193;
    colvals[3112] = 194;
    colvals[3113] = 195;
    colvals[3114] = 196;
    colvals[3115] = 197;
    colvals[3116] = 198;
    colvals[3117] = 199;
    colvals[3118] = 200;
    colvals[3119] = 201;
    colvals[3120] = 203;
    colvals[3121] = 205;
    colvals[3122] = 206;
    colvals[3123] = 207;
    colvals[3124] = 208;
    colvals[3125] = 209;
    colvals[3126] = 210;
    colvals[3127] = 212;
    colvals[3128] = 213;
    colvals[3129] = 214;
    colvals[3130] = 34;
    colvals[3131] = 101;
    colvals[3132] = 107;
    colvals[3133] = 114;
    colvals[3134] = 117;
    colvals[3135] = 118;
    colvals[3136] = 140;
    colvals[3137] = 144;
    colvals[3138] = 146;
    colvals[3139] = 147;
    colvals[3140] = 149;
    colvals[3141] = 151;
    colvals[3142] = 154;
    colvals[3143] = 159;
    colvals[3144] = 163;
    colvals[3145] = 164;
    colvals[3146] = 165;
    colvals[3147] = 167;
    colvals[3148] = 171;
    colvals[3149] = 172;
    colvals[3150] = 177;
    colvals[3151] = 179;
    colvals[3152] = 181;
    colvals[3153] = 183;
    colvals[3154] = 184;
    colvals[3155] = 187;
    colvals[3156] = 189;
    colvals[3157] = 190;
    colvals[3158] = 191;
    colvals[3159] = 192;
    colvals[3160] = 193;
    colvals[3161] = 194;
    colvals[3162] = 195;
    colvals[3163] = 196;
    colvals[3164] = 198;
    colvals[3165] = 200;
    colvals[3166] = 201;
    colvals[3167] = 202;
    colvals[3168] = 203;
    colvals[3169] = 205;
    colvals[3170] = 207;
    colvals[3171] = 208;
    colvals[3172] = 210;
    colvals[3173] = 211;
    colvals[3174] = 213;
    colvals[3175] = 68;
    colvals[3176] = 89;
    colvals[3177] = 117;
    colvals[3178] = 127;
    colvals[3179] = 128;
    colvals[3180] = 137;
    colvals[3181] = 138;
    colvals[3182] = 140;
    colvals[3183] = 143;
    colvals[3184] = 144;
    colvals[3185] = 146;
    colvals[3186] = 149;
    colvals[3187] = 150;
    colvals[3188] = 154;
    colvals[3189] = 156;
    colvals[3190] = 157;
    colvals[3191] = 158;
    colvals[3192] = 159;
    colvals[3193] = 164;
    colvals[3194] = 165;
    colvals[3195] = 166;
    colvals[3196] = 168;
    colvals[3197] = 169;
    colvals[3198] = 171;
    colvals[3199] = 172;
    colvals[3200] = 174;
    colvals[3201] = 176;
    colvals[3202] = 177;
    colvals[3203] = 178;
    colvals[3204] = 179;
    colvals[3205] = 180;
    colvals[3206] = 182;
    colvals[3207] = 183;
    colvals[3208] = 185;
    colvals[3209] = 186;
    colvals[3210] = 190;
    colvals[3211] = 191;
    colvals[3212] = 192;
    colvals[3213] = 193;
    colvals[3214] = 194;
    colvals[3215] = 195;
    colvals[3216] = 196;
    colvals[3217] = 199;
    colvals[3218] = 202;
    colvals[3219] = 203;
    colvals[3220] = 205;
    colvals[3221] = 206;
    colvals[3222] = 207;
    colvals[3223] = 209;
    colvals[3224] = 210;
    colvals[3225] = 212;
    colvals[3226] = 213;
    colvals[3227] = 31;
    colvals[3228] = 61;
    colvals[3229] = 64;
    colvals[3230] = 69;
    colvals[3231] = 73;
    colvals[3232] = 80;
    colvals[3233] = 82;
    colvals[3234] = 83;
    colvals[3235] = 91;
    colvals[3236] = 106;
    colvals[3237] = 108;
    colvals[3238] = 124;
    colvals[3239] = 129;
    colvals[3240] = 139;
    colvals[3241] = 143;
    colvals[3242] = 144;
    colvals[3243] = 145;
    colvals[3244] = 146;
    colvals[3245] = 149;
    colvals[3246] = 150;
    colvals[3247] = 151;
    colvals[3248] = 154;
    colvals[3249] = 157;
    colvals[3250] = 158;
    colvals[3251] = 159;
    colvals[3252] = 161;
    colvals[3253] = 164;
    colvals[3254] = 165;
    colvals[3255] = 166;
    colvals[3256] = 167;
    colvals[3257] = 168;
    colvals[3258] = 169;
    colvals[3259] = 171;
    colvals[3260] = 172;
    colvals[3261] = 174;
    colvals[3262] = 176;
    colvals[3263] = 178;
    colvals[3264] = 179;
    colvals[3265] = 182;
    colvals[3266] = 183;
    colvals[3267] = 184;
    colvals[3268] = 186;
    colvals[3269] = 188;
    colvals[3270] = 189;
    colvals[3271] = 190;
    colvals[3272] = 192;
    colvals[3273] = 193;
    colvals[3274] = 194;
    colvals[3275] = 195;
    colvals[3276] = 196;
    colvals[3277] = 197;
    colvals[3278] = 198;
    colvals[3279] = 199;
    colvals[3280] = 200;
    colvals[3281] = 201;
    colvals[3282] = 202;
    colvals[3283] = 203;
    colvals[3284] = 205;
    colvals[3285] = 206;
    colvals[3286] = 207;
    colvals[3287] = 208;
    colvals[3288] = 209;
    colvals[3289] = 211;
    colvals[3290] = 213;
    colvals[3291] = 214;
    colvals[3292] = 78;
    colvals[3293] = 93;
    colvals[3294] = 99;
    colvals[3295] = 101;
    colvals[3296] = 113;
    colvals[3297] = 122;
    colvals[3298] = 124;
    colvals[3299] = 134;
    colvals[3300] = 137;
    colvals[3301] = 138;
    colvals[3302] = 143;
    colvals[3303] = 144;
    colvals[3304] = 145;
    colvals[3305] = 146;
    colvals[3306] = 149;
    colvals[3307] = 150;
    colvals[3308] = 151;
    colvals[3309] = 153;
    colvals[3310] = 154;
    colvals[3311] = 155;
    colvals[3312] = 157;
    colvals[3313] = 158;
    colvals[3314] = 161;
    colvals[3315] = 164;
    colvals[3316] = 166;
    colvals[3317] = 167;
    colvals[3318] = 168;
    colvals[3319] = 169;
    colvals[3320] = 172;
    colvals[3321] = 174;
    colvals[3322] = 175;
    colvals[3323] = 176;
    colvals[3324] = 178;
    colvals[3325] = 179;
    colvals[3326] = 180;
    colvals[3327] = 181;
    colvals[3328] = 184;
    colvals[3329] = 185;
    colvals[3330] = 186;
    colvals[3331] = 188;
    colvals[3332] = 189;
    colvals[3333] = 190;
    colvals[3334] = 191;
    colvals[3335] = 192;
    colvals[3336] = 194;
    colvals[3337] = 195;
    colvals[3338] = 196;
    colvals[3339] = 197;
    colvals[3340] = 198;
    colvals[3341] = 199;
    colvals[3342] = 200;
    colvals[3343] = 201;
    colvals[3344] = 202;
    colvals[3345] = 203;
    colvals[3346] = 205;
    colvals[3347] = 206;
    colvals[3348] = 207;
    colvals[3349] = 208;
    colvals[3350] = 209;
    colvals[3351] = 212;
    colvals[3352] = 213;
    colvals[3353] = 214;
    colvals[3354] = 78;
    colvals[3355] = 81;
    colvals[3356] = 99;
    colvals[3357] = 100;
    colvals[3358] = 101;
    colvals[3359] = 103;
    colvals[3360] = 113;
    colvals[3361] = 117;
    colvals[3362] = 122;
    colvals[3363] = 128;
    colvals[3364] = 131;
    colvals[3365] = 134;
    colvals[3366] = 137;
    colvals[3367] = 138;
    colvals[3368] = 140;
    colvals[3369] = 143;
    colvals[3370] = 147;
    colvals[3371] = 150;
    colvals[3372] = 157;
    colvals[3373] = 158;
    colvals[3374] = 160;
    colvals[3375] = 161;
    colvals[3376] = 163;
    colvals[3377] = 165;
    colvals[3378] = 167;
    colvals[3379] = 169;
    colvals[3380] = 174;
    colvals[3381] = 175;
    colvals[3382] = 176;
    colvals[3383] = 177;
    colvals[3384] = 178;
    colvals[3385] = 180;
    colvals[3386] = 184;
    colvals[3387] = 185;
    colvals[3388] = 187;
    colvals[3389] = 188;
    colvals[3390] = 189;
    colvals[3391] = 190;
    colvals[3392] = 191;
    colvals[3393] = 192;
    colvals[3394] = 194;
    colvals[3395] = 195;
    colvals[3396] = 196;
    colvals[3397] = 198;
    colvals[3398] = 199;
    colvals[3399] = 200;
    colvals[3400] = 201;
    colvals[3401] = 202;
    colvals[3402] = 203;
    colvals[3403] = 205;
    colvals[3404] = 206;
    colvals[3405] = 207;
    colvals[3406] = 208;
    colvals[3407] = 210;
    colvals[3408] = 211;
    colvals[3409] = 212;
    colvals[3410] = 213;
    colvals[3411] = 214;
    colvals[3412] = 130;
    colvals[3413] = 134;
    colvals[3414] = 140;
    colvals[3415] = 143;
    colvals[3416] = 144;
    colvals[3417] = 145;
    colvals[3418] = 146;
    colvals[3419] = 147;
    colvals[3420] = 149;
    colvals[3421] = 150;
    colvals[3422] = 151;
    colvals[3423] = 154;
    colvals[3424] = 156;
    colvals[3425] = 157;
    colvals[3426] = 158;
    colvals[3427] = 159;
    colvals[3428] = 161;
    colvals[3429] = 163;
    colvals[3430] = 164;
    colvals[3431] = 165;
    colvals[3432] = 166;
    colvals[3433] = 167;
    colvals[3434] = 168;
    colvals[3435] = 169;
    colvals[3436] = 171;
    colvals[3437] = 172;
    colvals[3438] = 174;
    colvals[3439] = 175;
    colvals[3440] = 177;
    colvals[3441] = 178;
    colvals[3442] = 179;
    colvals[3443] = 180;
    colvals[3444] = 182;
    colvals[3445] = 183;
    colvals[3446] = 184;
    colvals[3447] = 185;
    colvals[3448] = 186;
    colvals[3449] = 187;
    colvals[3450] = 189;
    colvals[3451] = 190;
    colvals[3452] = 191;
    colvals[3453] = 192;
    colvals[3454] = 193;
    colvals[3455] = 194;
    colvals[3456] = 195;
    colvals[3457] = 196;
    colvals[3458] = 198;
    colvals[3459] = 199;
    colvals[3460] = 200;
    colvals[3461] = 201;
    colvals[3462] = 202;
    colvals[3463] = 203;
    colvals[3464] = 205;
    colvals[3465] = 206;
    colvals[3466] = 207;
    colvals[3467] = 208;
    colvals[3468] = 209;
    colvals[3469] = 210;
    colvals[3470] = 212;
    colvals[3471] = 213;
    colvals[3472] = 214;
    colvals[3473] = 57;
    colvals[3474] = 67;
    colvals[3475] = 99;
    colvals[3476] = 101;
    colvals[3477] = 103;
    colvals[3478] = 114;
    colvals[3479] = 118;
    colvals[3480] = 122;
    colvals[3481] = 128;
    colvals[3482] = 134;
    colvals[3483] = 143;
    colvals[3484] = 144;
    colvals[3485] = 145;
    colvals[3486] = 146;
    colvals[3487] = 149;
    colvals[3488] = 150;
    colvals[3489] = 151;
    colvals[3490] = 152;
    colvals[3491] = 154;
    colvals[3492] = 158;
    colvals[3493] = 159;
    colvals[3494] = 161;
    colvals[3495] = 164;
    colvals[3496] = 165;
    colvals[3497] = 166;
    colvals[3498] = 167;
    colvals[3499] = 168;
    colvals[3500] = 171;
    colvals[3501] = 172;
    colvals[3502] = 174;
    colvals[3503] = 176;
    colvals[3504] = 177;
    colvals[3505] = 178;
    colvals[3506] = 179;
    colvals[3507] = 180;
    colvals[3508] = 181;
    colvals[3509] = 183;
    colvals[3510] = 184;
    colvals[3511] = 185;
    colvals[3512] = 186;
    colvals[3513] = 187;
    colvals[3514] = 188;
    colvals[3515] = 189;
    colvals[3516] = 190;
    colvals[3517] = 191;
    colvals[3518] = 192;
    colvals[3519] = 193;
    colvals[3520] = 194;
    colvals[3521] = 195;
    colvals[3522] = 196;
    colvals[3523] = 198;
    colvals[3524] = 199;
    colvals[3525] = 200;
    colvals[3526] = 201;
    colvals[3527] = 202;
    colvals[3528] = 203;
    colvals[3529] = 205;
    colvals[3530] = 206;
    colvals[3531] = 207;
    colvals[3532] = 208;
    colvals[3533] = 209;
    colvals[3534] = 210;
    colvals[3535] = 211;
    colvals[3536] = 212;
    colvals[3537] = 213;
    colvals[3538] = 214;
    colvals[3539] = 87;
    colvals[3540] = 93;
    colvals[3541] = 108;
    colvals[3542] = 112;
    colvals[3543] = 115;
    colvals[3544] = 118;
    colvals[3545] = 121;
    colvals[3546] = 123;
    colvals[3547] = 131;
    colvals[3548] = 136;
    colvals[3549] = 137;
    colvals[3550] = 139;
    colvals[3551] = 140;
    colvals[3552] = 143;
    colvals[3553] = 145;
    colvals[3554] = 147;
    colvals[3555] = 148;
    colvals[3556] = 151;
    colvals[3557] = 152;
    colvals[3558] = 154;
    colvals[3559] = 156;
    colvals[3560] = 158;
    colvals[3561] = 161;
    colvals[3562] = 162;
    colvals[3563] = 164;
    colvals[3564] = 166;
    colvals[3565] = 167;
    colvals[3566] = 168;
    colvals[3567] = 169;
    colvals[3568] = 172;
    colvals[3569] = 174;
    colvals[3570] = 177;
    colvals[3571] = 178;
    colvals[3572] = 179;
    colvals[3573] = 183;
    colvals[3574] = 184;
    colvals[3575] = 185;
    colvals[3576] = 186;
    colvals[3577] = 187;
    colvals[3578] = 188;
    colvals[3579] = 190;
    colvals[3580] = 191;
    colvals[3581] = 192;
    colvals[3582] = 194;
    colvals[3583] = 195;
    colvals[3584] = 196;
    colvals[3585] = 198;
    colvals[3586] = 199;
    colvals[3587] = 200;
    colvals[3588] = 201;
    colvals[3589] = 203;
    colvals[3590] = 206;
    colvals[3591] = 207;
    colvals[3592] = 208;
    colvals[3593] = 212;
    colvals[3594] = 213;
    colvals[3595] = 214;
    colvals[3596] = 49;
    colvals[3597] = 68;
    colvals[3598] = 80;
    colvals[3599] = 95;
    colvals[3600] = 101;
    colvals[3601] = 105;
    colvals[3602] = 106;
    colvals[3603] = 107;
    colvals[3604] = 108;
    colvals[3605] = 114;
    colvals[3606] = 117;
    colvals[3607] = 118;
    colvals[3608] = 122;
    colvals[3609] = 127;
    colvals[3610] = 129;
    colvals[3611] = 137;
    colvals[3612] = 139;
    colvals[3613] = 143;
    colvals[3614] = 144;
    colvals[3615] = 145;
    colvals[3616] = 149;
    colvals[3617] = 151;
    colvals[3618] = 153;
    colvals[3619] = 154;
    colvals[3620] = 157;
    colvals[3621] = 158;
    colvals[3622] = 159;
    colvals[3623] = 163;
    colvals[3624] = 164;
    colvals[3625] = 165;
    colvals[3626] = 167;
    colvals[3627] = 169;
    colvals[3628] = 171;
    colvals[3629] = 172;
    colvals[3630] = 179;
    colvals[3631] = 181;
    colvals[3632] = 182;
    colvals[3633] = 183;
    colvals[3634] = 184;
    colvals[3635] = 185;
    colvals[3636] = 186;
    colvals[3637] = 187;
    colvals[3638] = 189;
    colvals[3639] = 190;
    colvals[3640] = 191;
    colvals[3641] = 192;
    colvals[3642] = 193;
    colvals[3643] = 194;
    colvals[3644] = 195;
    colvals[3645] = 196;
    colvals[3646] = 198;
    colvals[3647] = 199;
    colvals[3648] = 200;
    colvals[3649] = 201;
    colvals[3650] = 202;
    colvals[3651] = 203;
    colvals[3652] = 205;
    colvals[3653] = 207;
    colvals[3654] = 208;
    colvals[3655] = 210;
    colvals[3656] = 211;
    colvals[3657] = 212;
    colvals[3658] = 213;
    colvals[3659] = 214;
    colvals[3660] = 42;
    colvals[3661] = 81;
    colvals[3662] = 99;
    colvals[3663] = 101;
    colvals[3664] = 102;
    colvals[3665] = 103;
    colvals[3666] = 113;
    colvals[3667] = 122;
    colvals[3668] = 124;
    colvals[3669] = 128;
    colvals[3670] = 134;
    colvals[3671] = 135;
    colvals[3672] = 137;
    colvals[3673] = 138;
    colvals[3674] = 140;
    colvals[3675] = 142;
    colvals[3676] = 144;
    colvals[3677] = 145;
    colvals[3678] = 146;
    colvals[3679] = 149;
    colvals[3680] = 150;
    colvals[3681] = 151;
    colvals[3682] = 153;
    colvals[3683] = 154;
    colvals[3684] = 155;
    colvals[3685] = 156;
    colvals[3686] = 158;
    colvals[3687] = 159;
    colvals[3688] = 161;
    colvals[3689] = 164;
    colvals[3690] = 165;
    colvals[3691] = 166;
    colvals[3692] = 167;
    colvals[3693] = 168;
    colvals[3694] = 169;
    colvals[3695] = 172;
    colvals[3696] = 174;
    colvals[3697] = 175;
    colvals[3698] = 176;
    colvals[3699] = 177;
    colvals[3700] = 178;
    colvals[3701] = 179;
    colvals[3702] = 180;
    colvals[3703] = 182;
    colvals[3704] = 184;
    colvals[3705] = 185;
    colvals[3706] = 186;
    colvals[3707] = 188;
    colvals[3708] = 189;
    colvals[3709] = 190;
    colvals[3710] = 192;
    colvals[3711] = 193;
    colvals[3712] = 194;
    colvals[3713] = 195;
    colvals[3714] = 196;
    colvals[3715] = 197;
    colvals[3716] = 198;
    colvals[3717] = 199;
    colvals[3718] = 200;
    colvals[3719] = 201;
    colvals[3720] = 203;
    colvals[3721] = 205;
    colvals[3722] = 206;
    colvals[3723] = 207;
    colvals[3724] = 209;
    colvals[3725] = 210;
    colvals[3726] = 213;
    colvals[3727] = 214;
    colvals[3728] = 62;
    colvals[3729] = 81;
    colvals[3730] = 85;
    colvals[3731] = 99;
    colvals[3732] = 104;
    colvals[3733] = 122;
    colvals[3734] = 127;
    colvals[3735] = 128;
    colvals[3736] = 130;
    colvals[3737] = 131;
    colvals[3738] = 134;
    colvals[3739] = 138;
    colvals[3740] = 140;
    colvals[3741] = 143;
    colvals[3742] = 146;
    colvals[3743] = 147;
    colvals[3744] = 149;
    colvals[3745] = 150;
    colvals[3746] = 151;
    colvals[3747] = 154;
    colvals[3748] = 157;
    colvals[3749] = 158;
    colvals[3750] = 159;
    colvals[3751] = 161;
    colvals[3752] = 162;
    colvals[3753] = 163;
    colvals[3754] = 165;
    colvals[3755] = 167;
    colvals[3756] = 169;
    colvals[3757] = 171;
    colvals[3758] = 172;
    colvals[3759] = 174;
    colvals[3760] = 176;
    colvals[3761] = 177;
    colvals[3762] = 178;
    colvals[3763] = 179;
    colvals[3764] = 180;
    colvals[3765] = 182;
    colvals[3766] = 183;
    colvals[3767] = 184;
    colvals[3768] = 185;
    colvals[3769] = 186;
    colvals[3770] = 187;
    colvals[3771] = 188;
    colvals[3772] = 189;
    colvals[3773] = 190;
    colvals[3774] = 191;
    colvals[3775] = 192;
    colvals[3776] = 193;
    colvals[3777] = 195;
    colvals[3778] = 196;
    colvals[3779] = 198;
    colvals[3780] = 199;
    colvals[3781] = 200;
    colvals[3782] = 201;
    colvals[3783] = 203;
    colvals[3784] = 205;
    colvals[3785] = 206;
    colvals[3786] = 207;
    colvals[3787] = 208;
    colvals[3788] = 210;
    colvals[3789] = 211;
    colvals[3790] = 212;
    colvals[3791] = 213;
    colvals[3792] = 214;
    colvals[3793] = 72;
    colvals[3794] = 99;
    colvals[3795] = 122;
    colvals[3796] = 124;
    colvals[3797] = 128;
    colvals[3798] = 134;
    colvals[3799] = 137;
    colvals[3800] = 140;
    colvals[3801] = 142;
    colvals[3802] = 143;
    colvals[3803] = 144;
    colvals[3804] = 145;
    colvals[3805] = 146;
    colvals[3806] = 147;
    colvals[3807] = 148;
    colvals[3808] = 149;
    colvals[3809] = 150;
    colvals[3810] = 151;
    colvals[3811] = 154;
    colvals[3812] = 155;
    colvals[3813] = 157;
    colvals[3814] = 158;
    colvals[3815] = 159;
    colvals[3816] = 161;
    colvals[3817] = 163;
    colvals[3818] = 164;
    colvals[3819] = 165;
    colvals[3820] = 166;
    colvals[3821] = 167;
    colvals[3822] = 168;
    colvals[3823] = 169;
    colvals[3824] = 171;
    colvals[3825] = 172;
    colvals[3826] = 174;
    colvals[3827] = 175;
    colvals[3828] = 176;
    colvals[3829] = 178;
    colvals[3830] = 179;
    colvals[3831] = 180;
    colvals[3832] = 182;
    colvals[3833] = 183;
    colvals[3834] = 184;
    colvals[3835] = 185;
    colvals[3836] = 186;
    colvals[3837] = 188;
    colvals[3838] = 189;
    colvals[3839] = 190;
    colvals[3840] = 192;
    colvals[3841] = 193;
    colvals[3842] = 194;
    colvals[3843] = 195;
    colvals[3844] = 196;
    colvals[3845] = 198;
    colvals[3846] = 199;
    colvals[3847] = 200;
    colvals[3848] = 201;
    colvals[3849] = 202;
    colvals[3850] = 203;
    colvals[3851] = 205;
    colvals[3852] = 206;
    colvals[3853] = 207;
    colvals[3854] = 208;
    colvals[3855] = 209;
    colvals[3856] = 210;
    colvals[3857] = 213;
    colvals[3858] = 214;
    colvals[3859] = 45;
    colvals[3860] = 67;
    colvals[3861] = 68;
    colvals[3862] = 83;
    colvals[3863] = 95;
    colvals[3864] = 96;
    colvals[3865] = 107;
    colvals[3866] = 109;
    colvals[3867] = 113;
    colvals[3868] = 114;
    colvals[3869] = 117;
    colvals[3870] = 122;
    colvals[3871] = 127;
    colvals[3872] = 129;
    colvals[3873] = 135;
    colvals[3874] = 137;
    colvals[3875] = 144;
    colvals[3876] = 145;
    colvals[3877] = 146;
    colvals[3878] = 149;
    colvals[3879] = 150;
    colvals[3880] = 151;
    colvals[3881] = 153;
    colvals[3882] = 154;
    colvals[3883] = 155;
    colvals[3884] = 157;
    colvals[3885] = 158;
    colvals[3886] = 159;
    colvals[3887] = 161;
    colvals[3888] = 162;
    colvals[3889] = 164;
    colvals[3890] = 165;
    colvals[3891] = 167;
    colvals[3892] = 168;
    colvals[3893] = 169;
    colvals[3894] = 171;
    colvals[3895] = 172;
    colvals[3896] = 174;
    colvals[3897] = 177;
    colvals[3898] = 178;
    colvals[3899] = 179;
    colvals[3900] = 180;
    colvals[3901] = 181;
    colvals[3902] = 182;
    colvals[3903] = 183;
    colvals[3904] = 184;
    colvals[3905] = 185;
    colvals[3906] = 186;
    colvals[3907] = 187;
    colvals[3908] = 189;
    colvals[3909] = 190;
    colvals[3910] = 191;
    colvals[3911] = 192;
    colvals[3912] = 193;
    colvals[3913] = 194;
    colvals[3914] = 195;
    colvals[3915] = 196;
    colvals[3916] = 197;
    colvals[3917] = 198;
    colvals[3918] = 199;
    colvals[3919] = 200;
    colvals[3920] = 201;
    colvals[3921] = 202;
    colvals[3922] = 203;
    colvals[3923] = 205;
    colvals[3924] = 206;
    colvals[3925] = 207;
    colvals[3926] = 208;
    colvals[3927] = 209;
    colvals[3928] = 210;
    colvals[3929] = 212;
    colvals[3930] = 213;
    colvals[3931] = 214;
    colvals[3932] = 21;
    colvals[3933] = 53;
    colvals[3934] = 101;
    colvals[3935] = 102;
    colvals[3936] = 114;
    colvals[3937] = 118;
    colvals[3938] = 122;
    colvals[3939] = 124;
    colvals[3940] = 134;
    colvals[3941] = 136;
    colvals[3942] = 140;
    colvals[3943] = 143;
    colvals[3944] = 144;
    colvals[3945] = 145;
    colvals[3946] = 147;
    colvals[3947] = 148;
    colvals[3948] = 149;
    colvals[3949] = 151;
    colvals[3950] = 154;
    colvals[3951] = 156;
    colvals[3952] = 157;
    colvals[3953] = 158;
    colvals[3954] = 159;
    colvals[3955] = 161;
    colvals[3956] = 162;
    colvals[3957] = 163;
    colvals[3958] = 164;
    colvals[3959] = 165;
    colvals[3960] = 166;
    colvals[3961] = 167;
    colvals[3962] = 168;
    colvals[3963] = 169;
    colvals[3964] = 171;
    colvals[3965] = 173;
    colvals[3966] = 174;
    colvals[3967] = 175;
    colvals[3968] = 176;
    colvals[3969] = 178;
    colvals[3970] = 179;
    colvals[3971] = 180;
    colvals[3972] = 181;
    colvals[3973] = 182;
    colvals[3974] = 183;
    colvals[3975] = 184;
    colvals[3976] = 185;
    colvals[3977] = 186;
    colvals[3978] = 187;
    colvals[3979] = 188;
    colvals[3980] = 189;
    colvals[3981] = 190;
    colvals[3982] = 191;
    colvals[3983] = 192;
    colvals[3984] = 193;
    colvals[3985] = 194;
    colvals[3986] = 195;
    colvals[3987] = 196;
    colvals[3988] = 198;
    colvals[3989] = 199;
    colvals[3990] = 200;
    colvals[3991] = 201;
    colvals[3992] = 202;
    colvals[3993] = 203;
    colvals[3994] = 205;
    colvals[3995] = 206;
    colvals[3996] = 207;
    colvals[3997] = 208;
    colvals[3998] = 210;
    colvals[3999] = 211;
    colvals[4000] = 213;
    colvals[4001] = 214;
    colvals[4002] = 95;
    colvals[4003] = 99;
    colvals[4004] = 122;
    colvals[4005] = 124;
    colvals[4006] = 126;
    colvals[4007] = 132;
    colvals[4008] = 134;
    colvals[4009] = 136;
    colvals[4010] = 137;
    colvals[4011] = 140;
    colvals[4012] = 143;
    colvals[4013] = 144;
    colvals[4014] = 145;
    colvals[4015] = 146;
    colvals[4016] = 147;
    colvals[4017] = 149;
    colvals[4018] = 150;
    colvals[4019] = 151;
    colvals[4020] = 152;
    colvals[4021] = 153;
    colvals[4022] = 154;
    colvals[4023] = 155;
    colvals[4024] = 156;
    colvals[4025] = 157;
    colvals[4026] = 158;
    colvals[4027] = 160;
    colvals[4028] = 161;
    colvals[4029] = 162;
    colvals[4030] = 163;
    colvals[4031] = 164;
    colvals[4032] = 165;
    colvals[4033] = 166;
    colvals[4034] = 167;
    colvals[4035] = 168;
    colvals[4036] = 169;
    colvals[4037] = 171;
    colvals[4038] = 172;
    colvals[4039] = 174;
    colvals[4040] = 175;
    colvals[4041] = 176;
    colvals[4042] = 177;
    colvals[4043] = 178;
    colvals[4044] = 179;
    colvals[4045] = 180;
    colvals[4046] = 181;
    colvals[4047] = 182;
    colvals[4048] = 183;
    colvals[4049] = 184;
    colvals[4050] = 185;
    colvals[4051] = 186;
    colvals[4052] = 187;
    colvals[4053] = 188;
    colvals[4054] = 189;
    colvals[4055] = 190;
    colvals[4056] = 191;
    colvals[4057] = 192;
    colvals[4058] = 193;
    colvals[4059] = 194;
    colvals[4060] = 195;
    colvals[4061] = 196;
    colvals[4062] = 197;
    colvals[4063] = 198;
    colvals[4064] = 199;
    colvals[4065] = 200;
    colvals[4066] = 201;
    colvals[4067] = 202;
    colvals[4068] = 203;
    colvals[4069] = 205;
    colvals[4070] = 206;
    colvals[4071] = 207;
    colvals[4072] = 208;
    colvals[4073] = 209;
    colvals[4074] = 212;
    colvals[4075] = 213;
    colvals[4076] = 214;
    colvals[4077] = 52;
    colvals[4078] = 75;
    colvals[4079] = 79;
    colvals[4080] = 96;
    colvals[4081] = 113;
    colvals[4082] = 119;
    colvals[4083] = 125;
    colvals[4084] = 126;
    colvals[4085] = 129;
    colvals[4086] = 130;
    colvals[4087] = 132;
    colvals[4088] = 135;
    colvals[4089] = 138;
    colvals[4090] = 140;
    colvals[4091] = 142;
    colvals[4092] = 144;
    colvals[4093] = 146;
    colvals[4094] = 148;
    colvals[4095] = 149;
    colvals[4096] = 150;
    colvals[4097] = 151;
    colvals[4098] = 153;
    colvals[4099] = 154;
    colvals[4100] = 155;
    colvals[4101] = 156;
    colvals[4102] = 158;
    colvals[4103] = 161;
    colvals[4104] = 162;
    colvals[4105] = 164;
    colvals[4106] = 165;
    colvals[4107] = 166;
    colvals[4108] = 167;
    colvals[4109] = 168;
    colvals[4110] = 169;
    colvals[4111] = 170;
    colvals[4112] = 172;
    colvals[4113] = 174;
    colvals[4114] = 175;
    colvals[4115] = 176;
    colvals[4116] = 177;
    colvals[4117] = 178;
    colvals[4118] = 179;
    colvals[4119] = 180;
    colvals[4120] = 182;
    colvals[4121] = 185;
    colvals[4122] = 186;
    colvals[4123] = 187;
    colvals[4124] = 188;
    colvals[4125] = 189;
    colvals[4126] = 191;
    colvals[4127] = 192;
    colvals[4128] = 194;
    colvals[4129] = 196;
    colvals[4130] = 197;
    colvals[4131] = 199;
    colvals[4132] = 201;
    colvals[4133] = 203;
    colvals[4134] = 205;
    colvals[4135] = 206;
    colvals[4136] = 207;
    colvals[4137] = 209;
    colvals[4138] = 212;
    colvals[4139] = 213;
    colvals[4140] = 214;
    colvals[4141] = 70;
    colvals[4142] = 71;
    colvals[4143] = 74;
    colvals[4144] = 79;
    colvals[4145] = 81;
    colvals[4146] = 86;
    colvals[4147] = 87;
    colvals[4148] = 89;
    colvals[4149] = 104;
    colvals[4150] = 110;
    colvals[4151] = 115;
    colvals[4152] = 117;
    colvals[4153] = 123;
    colvals[4154] = 126;
    colvals[4155] = 130;
    colvals[4156] = 133;
    colvals[4157] = 134;
    colvals[4158] = 137;
    colvals[4159] = 138;
    colvals[4160] = 139;
    colvals[4161] = 144;
    colvals[4162] = 146;
    colvals[4163] = 148;
    colvals[4164] = 149;
    colvals[4165] = 150;
    colvals[4166] = 154;
    colvals[4167] = 155;
    colvals[4168] = 156;
    colvals[4169] = 158;
    colvals[4170] = 159;
    colvals[4171] = 162;
    colvals[4172] = 164;
    colvals[4173] = 165;
    colvals[4174] = 168;
    colvals[4175] = 174;
    colvals[4176] = 177;
    colvals[4177] = 178;
    colvals[4178] = 179;
    colvals[4179] = 182;
    colvals[4180] = 183;
    colvals[4181] = 184;
    colvals[4182] = 186;
    colvals[4183] = 187;
    colvals[4184] = 190;
    colvals[4185] = 191;
    colvals[4186] = 192;
    colvals[4187] = 193;
    colvals[4188] = 195;
    colvals[4189] = 196;
    colvals[4190] = 197;
    colvals[4191] = 201;
    colvals[4192] = 205;
    colvals[4193] = 207;
    colvals[4194] = 208;
    colvals[4195] = 209;
    colvals[4196] = 210;
    colvals[4197] = 212;
    colvals[4198] = 213;
    colvals[4199] = 22;
    colvals[4200] = 66;
    colvals[4201] = 92;
    colvals[4202] = 97;
    colvals[4203] = 99;
    colvals[4204] = 101;
    colvals[4205] = 114;
    colvals[4206] = 120;
    colvals[4207] = 121;
    colvals[4208] = 122;
    colvals[4209] = 124;
    colvals[4210] = 128;
    colvals[4211] = 130;
    colvals[4212] = 131;
    colvals[4213] = 134;
    colvals[4214] = 136;
    colvals[4215] = 138;
    colvals[4216] = 140;
    colvals[4217] = 143;
    colvals[4218] = 144;
    colvals[4219] = 145;
    colvals[4220] = 151;
    colvals[4221] = 154;
    colvals[4222] = 156;
    colvals[4223] = 157;
    colvals[4224] = 158;
    colvals[4225] = 159;
    colvals[4226] = 161;
    colvals[4227] = 163;
    colvals[4228] = 164;
    colvals[4229] = 165;
    colvals[4230] = 166;
    colvals[4231] = 167;
    colvals[4232] = 168;
    colvals[4233] = 171;
    colvals[4234] = 172;
    colvals[4235] = 174;
    colvals[4236] = 175;
    colvals[4237] = 177;
    colvals[4238] = 178;
    colvals[4239] = 179;
    colvals[4240] = 180;
    colvals[4241] = 181;
    colvals[4242] = 183;
    colvals[4243] = 184;
    colvals[4244] = 185;
    colvals[4245] = 186;
    colvals[4246] = 187;
    colvals[4247] = 188;
    colvals[4248] = 189;
    colvals[4249] = 190;
    colvals[4250] = 191;
    colvals[4251] = 192;
    colvals[4252] = 193;
    colvals[4253] = 194;
    colvals[4254] = 195;
    colvals[4255] = 196;
    colvals[4256] = 198;
    colvals[4257] = 199;
    colvals[4258] = 200;
    colvals[4259] = 201;
    colvals[4260] = 202;
    colvals[4261] = 203;
    colvals[4262] = 205;
    colvals[4263] = 206;
    colvals[4264] = 207;
    colvals[4265] = 208;
    colvals[4266] = 210;
    colvals[4267] = 211;
    colvals[4268] = 212;
    colvals[4269] = 213;
    colvals[4270] = 214;
    colvals[4271] = 73;
    colvals[4272] = 89;
    colvals[4273] = 91;
    colvals[4274] = 93;
    colvals[4275] = 95;
    colvals[4276] = 104;
    colvals[4277] = 106;
    colvals[4278] = 107;
    colvals[4279] = 108;
    colvals[4280] = 110;
    colvals[4281] = 112;
    colvals[4282] = 114;
    colvals[4283] = 116;
    colvals[4284] = 118;
    colvals[4285] = 121;
    colvals[4286] = 123;
    colvals[4287] = 133;
    colvals[4288] = 134;
    colvals[4289] = 136;
    colvals[4290] = 139;
    colvals[4291] = 140;
    colvals[4292] = 143;
    colvals[4293] = 145;
    colvals[4294] = 147;
    colvals[4295] = 151;
    colvals[4296] = 152;
    colvals[4297] = 156;
    colvals[4298] = 159;
    colvals[4299] = 161;
    colvals[4300] = 163;
    colvals[4301] = 165;
    colvals[4302] = 166;
    colvals[4303] = 171;
    colvals[4304] = 175;
    colvals[4305] = 177;
    colvals[4306] = 183;
    colvals[4307] = 184;
    colvals[4308] = 185;
    colvals[4309] = 186;
    colvals[4310] = 187;
    colvals[4311] = 189;
    colvals[4312] = 190;
    colvals[4313] = 191;
    colvals[4314] = 192;
    colvals[4315] = 193;
    colvals[4316] = 194;
    colvals[4317] = 195;
    colvals[4318] = 196;
    colvals[4319] = 198;
    colvals[4320] = 199;
    colvals[4321] = 200;
    colvals[4322] = 201;
    colvals[4323] = 202;
    colvals[4324] = 203;
    colvals[4325] = 207;
    colvals[4326] = 208;
    colvals[4327] = 210;
    colvals[4328] = 211;
    colvals[4329] = 212;
    colvals[4330] = 213;
    colvals[4331] = 214;
    colvals[4332] = 87;
    colvals[4333] = 88;
    colvals[4334] = 92;
    colvals[4335] = 93;
    colvals[4336] = 97;
    colvals[4337] = 98;
    colvals[4338] = 102;
    colvals[4339] = 108;
    colvals[4340] = 109;
    colvals[4341] = 112;
    colvals[4342] = 115;
    colvals[4343] = 118;
    colvals[4344] = 119;
    colvals[4345] = 120;
    colvals[4346] = 121;
    colvals[4347] = 123;
    colvals[4348] = 124;
    colvals[4349] = 131;
    colvals[4350] = 132;
    colvals[4351] = 135;
    colvals[4352] = 136;
    colvals[4353] = 139;
    colvals[4354] = 140;
    colvals[4355] = 142;
    colvals[4356] = 143;
    colvals[4357] = 144;
    colvals[4358] = 145;
    colvals[4359] = 146;
    colvals[4360] = 147;
    colvals[4361] = 148;
    colvals[4362] = 149;
    colvals[4363] = 150;
    colvals[4364] = 151;
    colvals[4365] = 152;
    colvals[4366] = 156;
    colvals[4367] = 158;
    colvals[4368] = 159;
    colvals[4369] = 161;
    colvals[4370] = 162;
    colvals[4371] = 164;
    colvals[4372] = 166;
    colvals[4373] = 168;
    colvals[4374] = 169;
    colvals[4375] = 174;
    colvals[4376] = 175;
    colvals[4377] = 177;
    colvals[4378] = 178;
    colvals[4379] = 179;
    colvals[4380] = 180;
    colvals[4381] = 183;
    colvals[4382] = 184;
    colvals[4383] = 185;
    colvals[4384] = 186;
    colvals[4385] = 187;
    colvals[4386] = 188;
    colvals[4387] = 189;
    colvals[4388] = 191;
    colvals[4389] = 192;
    colvals[4390] = 193;
    colvals[4391] = 194;
    colvals[4392] = 195;
    colvals[4393] = 196;
    colvals[4394] = 198;
    colvals[4395] = 199;
    colvals[4396] = 200;
    colvals[4397] = 201;
    colvals[4398] = 202;
    colvals[4399] = 203;
    colvals[4400] = 205;
    colvals[4401] = 206;
    colvals[4402] = 207;
    colvals[4403] = 208;
    colvals[4404] = 209;
    colvals[4405] = 210;
    colvals[4406] = 212;
    colvals[4407] = 213;
    colvals[4408] = 214;
    colvals[4409] = 75;
    colvals[4410] = 79;
    colvals[4411] = 86;
    colvals[4412] = 92;
    colvals[4413] = 99;
    colvals[4414] = 101;
    colvals[4415] = 103;
    colvals[4416] = 104;
    colvals[4417] = 107;
    colvals[4418] = 113;
    colvals[4419] = 114;
    colvals[4420] = 120;
    colvals[4421] = 121;
    colvals[4422] = 122;
    colvals[4423] = 125;
    colvals[4424] = 127;
    colvals[4425] = 128;
    colvals[4426] = 129;
    colvals[4427] = 130;
    colvals[4428] = 134;
    colvals[4429] = 136;
    colvals[4430] = 137;
    colvals[4431] = 138;
    colvals[4432] = 139;
    colvals[4433] = 143;
    colvals[4434] = 144;
    colvals[4435] = 145;
    colvals[4436] = 146;
    colvals[4437] = 148;
    colvals[4438] = 149;
    colvals[4439] = 150;
    colvals[4440] = 151;
    colvals[4441] = 152;
    colvals[4442] = 154;
    colvals[4443] = 155;
    colvals[4444] = 156;
    colvals[4445] = 157;
    colvals[4446] = 158;
    colvals[4447] = 159;
    colvals[4448] = 160;
    colvals[4449] = 161;
    colvals[4450] = 162;
    colvals[4451] = 163;
    colvals[4452] = 164;
    colvals[4453] = 165;
    colvals[4454] = 167;
    colvals[4455] = 168;
    colvals[4456] = 171;
    colvals[4457] = 172;
    colvals[4458] = 175;
    colvals[4459] = 176;
    colvals[4460] = 177;
    colvals[4461] = 178;
    colvals[4462] = 179;
    colvals[4463] = 180;
    colvals[4464] = 183;
    colvals[4465] = 184;
    colvals[4466] = 185;
    colvals[4467] = 186;
    colvals[4468] = 187;
    colvals[4469] = 188;
    colvals[4470] = 189;
    colvals[4471] = 190;
    colvals[4472] = 191;
    colvals[4473] = 192;
    colvals[4474] = 193;
    colvals[4475] = 194;
    colvals[4476] = 195;
    colvals[4477] = 196;
    colvals[4478] = 197;
    colvals[4479] = 198;
    colvals[4480] = 199;
    colvals[4481] = 200;
    colvals[4482] = 201;
    colvals[4483] = 202;
    colvals[4484] = 203;
    colvals[4485] = 205;
    colvals[4486] = 206;
    colvals[4487] = 207;
    colvals[4488] = 208;
    colvals[4489] = 209;
    colvals[4490] = 210;
    colvals[4491] = 211;
    colvals[4492] = 212;
    colvals[4493] = 213;
    colvals[4494] = 214;
    colvals[4495] = 49;
    colvals[4496] = 50;
    colvals[4497] = 64;
    colvals[4498] = 67;
    colvals[4499] = 68;
    colvals[4500] = 80;
    colvals[4501] = 88;
    colvals[4502] = 99;
    colvals[4503] = 101;
    colvals[4504] = 103;
    colvals[4505] = 105;
    colvals[4506] = 106;
    colvals[4507] = 107;
    colvals[4508] = 108;
    colvals[4509] = 109;
    colvals[4510] = 115;
    colvals[4511] = 118;
    colvals[4512] = 122;
    colvals[4513] = 124;
    colvals[4514] = 128;
    colvals[4515] = 129;
    colvals[4516] = 136;
    colvals[4517] = 137;
    colvals[4518] = 139;
    colvals[4519] = 140;
    colvals[4520] = 142;
    colvals[4521] = 143;
    colvals[4522] = 145;
    colvals[4523] = 146;
    colvals[4524] = 147;
    colvals[4525] = 148;
    colvals[4526] = 151;
    colvals[4527] = 152;
    colvals[4528] = 154;
    colvals[4529] = 156;
    colvals[4530] = 159;
    colvals[4531] = 161;
    colvals[4532] = 162;
    colvals[4533] = 163;
    colvals[4534] = 164;
    colvals[4535] = 165;
    colvals[4536] = 166;
    colvals[4537] = 167;
    colvals[4538] = 168;
    colvals[4539] = 169;
    colvals[4540] = 171;
    colvals[4541] = 172;
    colvals[4542] = 173;
    colvals[4543] = 174;
    colvals[4544] = 175;
    colvals[4545] = 176;
    colvals[4546] = 177;
    colvals[4547] = 178;
    colvals[4548] = 179;
    colvals[4549] = 181;
    colvals[4550] = 182;
    colvals[4551] = 183;
    colvals[4552] = 184;
    colvals[4553] = 185;
    colvals[4554] = 186;
    colvals[4555] = 187;
    colvals[4556] = 189;
    colvals[4557] = 190;
    colvals[4558] = 191;
    colvals[4559] = 192;
    colvals[4560] = 193;
    colvals[4561] = 194;
    colvals[4562] = 195;
    colvals[4563] = 196;
    colvals[4564] = 198;
    colvals[4565] = 199;
    colvals[4566] = 200;
    colvals[4567] = 201;
    colvals[4568] = 202;
    colvals[4569] = 203;
    colvals[4570] = 207;
    colvals[4571] = 208;
    colvals[4572] = 210;
    colvals[4573] = 211;
    colvals[4574] = 212;
    colvals[4575] = 213;
    colvals[4576] = 214;
    colvals[4577] = 51;
    colvals[4578] = 54;
    colvals[4579] = 64;
    colvals[4580] = 73;
    colvals[4581] = 78;
    colvals[4582] = 80;
    colvals[4583] = 81;
    colvals[4584] = 85;
    colvals[4585] = 87;
    colvals[4586] = 89;
    colvals[4587] = 91;
    colvals[4588] = 93;
    colvals[4589] = 95;
    colvals[4590] = 106;
    colvals[4591] = 107;
    colvals[4592] = 108;
    colvals[4593] = 109;
    colvals[4594] = 110;
    colvals[4595] = 112;
    colvals[4596] = 114;
    colvals[4597] = 115;
    colvals[4598] = 116;
    colvals[4599] = 117;
    colvals[4600] = 118;
    colvals[4601] = 121;
    colvals[4602] = 122;
    colvals[4603] = 123;
    colvals[4604] = 127;
    colvals[4605] = 128;
    colvals[4606] = 133;
    colvals[4607] = 134;
    colvals[4608] = 136;
    colvals[4609] = 137;
    colvals[4610] = 139;
    colvals[4611] = 147;
    colvals[4612] = 152;
    colvals[4613] = 156;
    colvals[4614] = 157;
    colvals[4615] = 159;
    colvals[4616] = 163;
    colvals[4617] = 165;
    colvals[4618] = 171;
    colvals[4619] = 177;
    colvals[4620] = 181;
    colvals[4621] = 183;
    colvals[4622] = 184;
    colvals[4623] = 185;
    colvals[4624] = 187;
    colvals[4625] = 189;
    colvals[4626] = 190;
    colvals[4627] = 191;
    colvals[4628] = 192;
    colvals[4629] = 193;
    colvals[4630] = 194;
    colvals[4631] = 195;
    colvals[4632] = 196;
    colvals[4633] = 198;
    colvals[4634] = 201;
    colvals[4635] = 203;
    colvals[4636] = 204;
    colvals[4637] = 208;
    colvals[4638] = 210;
    colvals[4639] = 211;
    colvals[4640] = 212;
    colvals[4641] = 213;
    colvals[4642] = 214;
    colvals[4643] = 51;
    colvals[4644] = 54;
    colvals[4645] = 56;
    colvals[4646] = 64;
    colvals[4647] = 73;
    colvals[4648] = 78;
    colvals[4649] = 80;
    colvals[4650] = 81;
    colvals[4651] = 85;
    colvals[4652] = 87;
    colvals[4653] = 89;
    colvals[4654] = 91;
    colvals[4655] = 93;
    colvals[4656] = 95;
    colvals[4657] = 106;
    colvals[4658] = 107;
    colvals[4659] = 108;
    colvals[4660] = 109;
    colvals[4661] = 110;
    colvals[4662] = 112;
    colvals[4663] = 114;
    colvals[4664] = 115;
    colvals[4665] = 116;
    colvals[4666] = 117;
    colvals[4667] = 118;
    colvals[4668] = 121;
    colvals[4669] = 122;
    colvals[4670] = 123;
    colvals[4671] = 127;
    colvals[4672] = 128;
    colvals[4673] = 133;
    colvals[4674] = 134;
    colvals[4675] = 136;
    colvals[4676] = 137;
    colvals[4677] = 139;
    colvals[4678] = 147;
    colvals[4679] = 152;
    colvals[4680] = 154;
    colvals[4681] = 156;
    colvals[4682] = 157;
    colvals[4683] = 159;
    colvals[4684] = 163;
    colvals[4685] = 165;
    colvals[4686] = 171;
    colvals[4687] = 177;
    colvals[4688] = 181;
    colvals[4689] = 183;
    colvals[4690] = 184;
    colvals[4691] = 185;
    colvals[4692] = 187;
    colvals[4693] = 189;
    colvals[4694] = 190;
    colvals[4695] = 191;
    colvals[4696] = 192;
    colvals[4697] = 193;
    colvals[4698] = 194;
    colvals[4699] = 195;
    colvals[4700] = 196;
    colvals[4701] = 198;
    colvals[4702] = 201;
    colvals[4703] = 203;
    colvals[4704] = 204;
    colvals[4705] = 206;
    colvals[4706] = 208;
    colvals[4707] = 210;
    colvals[4708] = 211;
    colvals[4709] = 212;
    colvals[4710] = 213;
    colvals[4711] = 214;
    colvals[4712] = 56;
    colvals[4713] = 65;
    colvals[4714] = 81;
    colvals[4715] = 85;
    colvals[4716] = 87;
    colvals[4717] = 93;
    colvals[4718] = 101;
    colvals[4719] = 104;
    colvals[4720] = 106;
    colvals[4721] = 109;
    colvals[4722] = 110;
    colvals[4723] = 112;
    colvals[4724] = 115;
    colvals[4725] = 116;
    colvals[4726] = 117;
    colvals[4727] = 118;
    colvals[4728] = 121;
    colvals[4729] = 122;
    colvals[4730] = 123;
    colvals[4731] = 127;
    colvals[4732] = 133;
    colvals[4733] = 134;
    colvals[4734] = 136;
    colvals[4735] = 139;
    colvals[4736] = 140;
    colvals[4737] = 144;
    colvals[4738] = 147;
    colvals[4739] = 152;
    colvals[4740] = 154;
    colvals[4741] = 156;
    colvals[4742] = 159;
    colvals[4743] = 163;
    colvals[4744] = 164;
    colvals[4745] = 165;
    colvals[4746] = 171;
    colvals[4747] = 177;
    colvals[4748] = 181;
    colvals[4749] = 183;
    colvals[4750] = 184;
    colvals[4751] = 185;
    colvals[4752] = 187;
    colvals[4753] = 189;
    colvals[4754] = 190;
    colvals[4755] = 191;
    colvals[4756] = 192;
    colvals[4757] = 193;
    colvals[4758] = 194;
    colvals[4759] = 195;
    colvals[4760] = 196;
    colvals[4761] = 198;
    colvals[4762] = 200;
    colvals[4763] = 201;
    colvals[4764] = 205;
    colvals[4765] = 207;
    colvals[4766] = 208;
    colvals[4767] = 210;
    colvals[4768] = 211;
    colvals[4769] = 212;
    colvals[4770] = 213;
    colvals[4771] = 51;
    colvals[4772] = 54;
    colvals[4773] = 57;
    colvals[4774] = 65;
    colvals[4775] = 66;
    colvals[4776] = 73;
    colvals[4777] = 85;
    colvals[4778] = 87;
    colvals[4779] = 91;
    colvals[4780] = 93;
    colvals[4781] = 101;
    colvals[4782] = 106;
    colvals[4783] = 108;
    colvals[4784] = 109;
    colvals[4785] = 110;
    colvals[4786] = 112;
    colvals[4787] = 115;
    colvals[4788] = 116;
    colvals[4789] = 117;
    colvals[4790] = 118;
    colvals[4791] = 121;
    colvals[4792] = 122;
    colvals[4793] = 123;
    colvals[4794] = 127;
    colvals[4795] = 128;
    colvals[4796] = 133;
    colvals[4797] = 134;
    colvals[4798] = 136;
    colvals[4799] = 137;
    colvals[4800] = 139;
    colvals[4801] = 140;
    colvals[4802] = 145;
    colvals[4803] = 147;
    colvals[4804] = 152;
    colvals[4805] = 154;
    colvals[4806] = 156;
    colvals[4807] = 157;
    colvals[4808] = 158;
    colvals[4809] = 159;
    colvals[4810] = 161;
    colvals[4811] = 162;
    colvals[4812] = 163;
    colvals[4813] = 164;
    colvals[4814] = 165;
    colvals[4815] = 171;
    colvals[4816] = 172;
    colvals[4817] = 175;
    colvals[4818] = 177;
    colvals[4819] = 183;
    colvals[4820] = 184;
    colvals[4821] = 185;
    colvals[4822] = 186;
    colvals[4823] = 187;
    colvals[4824] = 190;
    colvals[4825] = 191;
    colvals[4826] = 192;
    colvals[4827] = 193;
    colvals[4828] = 194;
    colvals[4829] = 195;
    colvals[4830] = 196;
    colvals[4831] = 198;
    colvals[4832] = 200;
    colvals[4833] = 201;
    colvals[4834] = 203;
    colvals[4835] = 204;
    colvals[4836] = 205;
    colvals[4837] = 206;
    colvals[4838] = 207;
    colvals[4839] = 210;
    colvals[4840] = 212;
    colvals[4841] = 213;
    colvals[4842] = 214;
    colvals[4843] = 49;
    colvals[4844] = 67;
    colvals[4845] = 73;
    colvals[4846] = 79;
    colvals[4847] = 80;
    colvals[4848] = 88;
    colvals[4849] = 91;
    colvals[4850] = 92;
    colvals[4851] = 99;
    colvals[4852] = 101;
    colvals[4853] = 103;
    colvals[4854] = 105;
    colvals[4855] = 106;
    colvals[4856] = 108;
    colvals[4857] = 109;
    colvals[4858] = 110;
    colvals[4859] = 111;
    colvals[4860] = 114;
    colvals[4861] = 115;
    colvals[4862] = 116;
    colvals[4863] = 118;
    colvals[4864] = 119;
    colvals[4865] = 120;
    colvals[4866] = 121;
    colvals[4867] = 122;
    colvals[4868] = 123;
    colvals[4869] = 124;
    colvals[4870] = 126;
    colvals[4871] = 127;
    colvals[4872] = 128;
    colvals[4873] = 130;
    colvals[4874] = 131;
    colvals[4875] = 132;
    colvals[4876] = 133;
    colvals[4877] = 136;
    colvals[4878] = 137;
    colvals[4879] = 138;
    colvals[4880] = 139;
    colvals[4881] = 142;
    colvals[4882] = 143;
    colvals[4883] = 144;
    colvals[4884] = 145;
    colvals[4885] = 146;
    colvals[4886] = 147;
    colvals[4887] = 148;
    colvals[4888] = 150;
    colvals[4889] = 151;
    colvals[4890] = 152;
    colvals[4891] = 154;
    colvals[4892] = 156;
    colvals[4893] = 157;
    colvals[4894] = 159;
    colvals[4895] = 160;
    colvals[4896] = 161;
    colvals[4897] = 162;
    colvals[4898] = 163;
    colvals[4899] = 164;
    colvals[4900] = 165;
    colvals[4901] = 166;
    colvals[4902] = 167;
    colvals[4903] = 168;
    colvals[4904] = 169;
    colvals[4905] = 171;
    colvals[4906] = 172;
    colvals[4907] = 173;
    colvals[4908] = 174;
    colvals[4909] = 175;
    colvals[4910] = 176;
    colvals[4911] = 177;
    colvals[4912] = 178;
    colvals[4913] = 179;
    colvals[4914] = 180;
    colvals[4915] = 181;
    colvals[4916] = 182;
    colvals[4917] = 183;
    colvals[4918] = 184;
    colvals[4919] = 185;
    colvals[4920] = 186;
    colvals[4921] = 187;
    colvals[4922] = 188;
    colvals[4923] = 189;
    colvals[4924] = 190;
    colvals[4925] = 191;
    colvals[4926] = 192;
    colvals[4927] = 193;
    colvals[4928] = 194;
    colvals[4929] = 195;
    colvals[4930] = 196;
    colvals[4931] = 197;
    colvals[4932] = 198;
    colvals[4933] = 199;
    colvals[4934] = 200;
    colvals[4935] = 201;
    colvals[4936] = 202;
    colvals[4937] = 203;
    colvals[4938] = 205;
    colvals[4939] = 206;
    colvals[4940] = 207;
    colvals[4941] = 208;
    colvals[4942] = 210;
    colvals[4943] = 211;
    colvals[4944] = 212;
    colvals[4945] = 213;
    colvals[4946] = 214;
    colvals[4947] = 50;
    colvals[4948] = 57;
    colvals[4949] = 61;
    colvals[4950] = 67;
    colvals[4951] = 69;
    colvals[4952] = 73;
    colvals[4953] = 82;
    colvals[4954] = 83;
    colvals[4955] = 91;
    colvals[4956] = 103;
    colvals[4957] = 105;
    colvals[4958] = 106;
    colvals[4959] = 107;
    colvals[4960] = 108;
    colvals[4961] = 110;
    colvals[4962] = 112;
    colvals[4963] = 114;
    colvals[4964] = 115;
    colvals[4965] = 116;
    colvals[4966] = 118;
    colvals[4967] = 119;
    colvals[4968] = 121;
    colvals[4969] = 123;
    colvals[4970] = 124;
    colvals[4971] = 126;
    colvals[4972] = 128;
    colvals[4973] = 129;
    colvals[4974] = 130;
    colvals[4975] = 131;
    colvals[4976] = 134;
    colvals[4977] = 136;
    colvals[4978] = 139;
    colvals[4979] = 140;
    colvals[4980] = 143;
    colvals[4981] = 144;
    colvals[4982] = 145;
    colvals[4983] = 146;
    colvals[4984] = 147;
    colvals[4985] = 148;
    colvals[4986] = 149;
    colvals[4987] = 150;
    colvals[4988] = 151;
    colvals[4989] = 152;
    colvals[4990] = 154;
    colvals[4991] = 156;
    colvals[4992] = 158;
    colvals[4993] = 159;
    colvals[4994] = 161;
    colvals[4995] = 162;
    colvals[4996] = 163;
    colvals[4997] = 164;
    colvals[4998] = 165;
    colvals[4999] = 166;
    colvals[5000] = 167;
    colvals[5001] = 169;
    colvals[5002] = 171;
    colvals[5003] = 172;
    colvals[5004] = 175;
    colvals[5005] = 176;
    colvals[5006] = 177;
    colvals[5007] = 178;
    colvals[5008] = 179;
    colvals[5009] = 180;
    colvals[5010] = 181;
    colvals[5011] = 183;
    colvals[5012] = 184;
    colvals[5013] = 185;
    colvals[5014] = 186;
    colvals[5015] = 187;
    colvals[5016] = 188;
    colvals[5017] = 189;
    colvals[5018] = 190;
    colvals[5019] = 192;
    colvals[5020] = 193;
    colvals[5021] = 194;
    colvals[5022] = 195;
    colvals[5023] = 196;
    colvals[5024] = 197;
    colvals[5025] = 198;
    colvals[5026] = 199;
    colvals[5027] = 200;
    colvals[5028] = 201;
    colvals[5029] = 202;
    colvals[5030] = 203;
    colvals[5031] = 205;
    colvals[5032] = 207;
    colvals[5033] = 208;
    colvals[5034] = 209;
    colvals[5035] = 210;
    colvals[5036] = 211;
    colvals[5037] = 212;
    colvals[5038] = 213;
    colvals[5039] = 214;
    colvals[5040] = 59;
    colvals[5041] = 69;
    colvals[5042] = 71;
    colvals[5043] = 74;
    colvals[5044] = 81;
    colvals[5045] = 87;
    colvals[5046] = 89;
    colvals[5047] = 93;
    colvals[5048] = 104;
    colvals[5049] = 110;
    colvals[5050] = 112;
    colvals[5051] = 115;
    colvals[5052] = 117;
    colvals[5053] = 118;
    colvals[5054] = 123;
    colvals[5055] = 124;
    colvals[5056] = 127;
    colvals[5057] = 129;
    colvals[5058] = 130;
    colvals[5059] = 132;
    colvals[5060] = 133;
    colvals[5061] = 134;
    colvals[5062] = 136;
    colvals[5063] = 138;
    colvals[5064] = 139;
    colvals[5065] = 140;
    colvals[5066] = 142;
    colvals[5067] = 143;
    colvals[5068] = 144;
    colvals[5069] = 145;
    colvals[5070] = 146;
    colvals[5071] = 147;
    colvals[5072] = 148;
    colvals[5073] = 149;
    colvals[5074] = 150;
    colvals[5075] = 151;
    colvals[5076] = 152;
    colvals[5077] = 154;
    colvals[5078] = 156;
    colvals[5079] = 157;
    colvals[5080] = 158;
    colvals[5081] = 159;
    colvals[5082] = 161;
    colvals[5083] = 163;
    colvals[5084] = 164;
    colvals[5085] = 165;
    colvals[5086] = 166;
    colvals[5087] = 167;
    colvals[5088] = 168;
    colvals[5089] = 169;
    colvals[5090] = 171;
    colvals[5091] = 172;
    colvals[5092] = 174;
    colvals[5093] = 175;
    colvals[5094] = 176;
    colvals[5095] = 177;
    colvals[5096] = 178;
    colvals[5097] = 179;
    colvals[5098] = 180;
    colvals[5099] = 182;
    colvals[5100] = 183;
    colvals[5101] = 184;
    colvals[5102] = 186;
    colvals[5103] = 187;
    colvals[5104] = 188;
    colvals[5105] = 190;
    colvals[5106] = 191;
    colvals[5107] = 192;
    colvals[5108] = 193;
    colvals[5109] = 195;
    colvals[5110] = 196;
    colvals[5111] = 197;
    colvals[5112] = 198;
    colvals[5113] = 199;
    colvals[5114] = 200;
    colvals[5115] = 201;
    colvals[5116] = 203;
    colvals[5117] = 205;
    colvals[5118] = 206;
    colvals[5119] = 207;
    colvals[5120] = 208;
    colvals[5121] = 209;
    colvals[5122] = 210;
    colvals[5123] = 211;
    colvals[5124] = 212;
    colvals[5125] = 213;
    colvals[5126] = 47;
    colvals[5127] = 69;
    colvals[5128] = 70;
    colvals[5129] = 71;
    colvals[5130] = 74;
    colvals[5131] = 75;
    colvals[5132] = 79;
    colvals[5133] = 81;
    colvals[5134] = 86;
    colvals[5135] = 87;
    colvals[5136] = 89;
    colvals[5137] = 96;
    colvals[5138] = 99;
    colvals[5139] = 103;
    colvals[5140] = 104;
    colvals[5141] = 110;
    colvals[5142] = 113;
    colvals[5143] = 115;
    colvals[5144] = 117;
    colvals[5145] = 122;
    colvals[5146] = 123;
    colvals[5147] = 126;
    colvals[5148] = 128;
    colvals[5149] = 129;
    colvals[5150] = 130;
    colvals[5151] = 133;
    colvals[5152] = 134;
    colvals[5153] = 137;
    colvals[5154] = 138;
    colvals[5155] = 139;
    colvals[5156] = 140;
    colvals[5157] = 142;
    colvals[5158] = 143;
    colvals[5159] = 144;
    colvals[5160] = 145;
    colvals[5161] = 146;
    colvals[5162] = 147;
    colvals[5163] = 148;
    colvals[5164] = 149;
    colvals[5165] = 150;
    colvals[5166] = 151;
    colvals[5167] = 154;
    colvals[5168] = 155;
    colvals[5169] = 156;
    colvals[5170] = 157;
    colvals[5171] = 158;
    colvals[5172] = 159;
    colvals[5173] = 160;
    colvals[5174] = 161;
    colvals[5175] = 162;
    colvals[5176] = 164;
    colvals[5177] = 165;
    colvals[5178] = 167;
    colvals[5179] = 168;
    colvals[5180] = 171;
    colvals[5181] = 172;
    colvals[5182] = 174;
    colvals[5183] = 175;
    colvals[5184] = 176;
    colvals[5185] = 177;
    colvals[5186] = 178;
    colvals[5187] = 179;
    colvals[5188] = 180;
    colvals[5189] = 182;
    colvals[5190] = 183;
    colvals[5191] = 184;
    colvals[5192] = 185;
    colvals[5193] = 186;
    colvals[5194] = 187;
    colvals[5195] = 190;
    colvals[5196] = 191;
    colvals[5197] = 192;
    colvals[5198] = 193;
    colvals[5199] = 194;
    colvals[5200] = 195;
    colvals[5201] = 196;
    colvals[5202] = 197;
    colvals[5203] = 198;
    colvals[5204] = 199;
    colvals[5205] = 200;
    colvals[5206] = 201;
    colvals[5207] = 203;
    colvals[5208] = 205;
    colvals[5209] = 206;
    colvals[5210] = 207;
    colvals[5211] = 209;
    colvals[5212] = 210;
    colvals[5213] = 212;
    colvals[5214] = 213;
    colvals[5215] = 214;
    colvals[5216] = 16;
    colvals[5217] = 49;
    colvals[5218] = 57;
    colvals[5219] = 59;
    colvals[5220] = 71;
    colvals[5221] = 73;
    colvals[5222] = 74;
    colvals[5223] = 78;
    colvals[5224] = 80;
    colvals[5225] = 81;
    colvals[5226] = 87;
    colvals[5227] = 89;
    colvals[5228] = 91;
    colvals[5229] = 92;
    colvals[5230] = 93;
    colvals[5231] = 99;
    colvals[5232] = 101;
    colvals[5233] = 104;
    colvals[5234] = 106;
    colvals[5235] = 108;
    colvals[5236] = 109;
    colvals[5237] = 110;
    colvals[5238] = 112;
    colvals[5239] = 114;
    colvals[5240] = 115;
    colvals[5241] = 117;
    colvals[5242] = 118;
    colvals[5243] = 119;
    colvals[5244] = 121;
    colvals[5245] = 122;
    colvals[5246] = 123;
    colvals[5247] = 124;
    colvals[5248] = 127;
    colvals[5249] = 130;
    colvals[5250] = 133;
    colvals[5251] = 134;
    colvals[5252] = 136;
    colvals[5253] = 138;
    colvals[5254] = 139;
    colvals[5255] = 142;
    colvals[5256] = 143;
    colvals[5257] = 144;
    colvals[5258] = 145;
    colvals[5259] = 146;
    colvals[5260] = 147;
    colvals[5261] = 149;
    colvals[5262] = 150;
    colvals[5263] = 151;
    colvals[5264] = 152;
    colvals[5265] = 154;
    colvals[5266] = 155;
    colvals[5267] = 156;
    colvals[5268] = 157;
    colvals[5269] = 158;
    colvals[5270] = 159;
    colvals[5271] = 160;
    colvals[5272] = 161;
    colvals[5273] = 163;
    colvals[5274] = 164;
    colvals[5275] = 165;
    colvals[5276] = 166;
    colvals[5277] = 167;
    colvals[5278] = 169;
    colvals[5279] = 171;
    colvals[5280] = 172;
    colvals[5281] = 174;
    colvals[5282] = 175;
    colvals[5283] = 176;
    colvals[5284] = 177;
    colvals[5285] = 178;
    colvals[5286] = 179;
    colvals[5287] = 180;
    colvals[5288] = 183;
    colvals[5289] = 184;
    colvals[5290] = 185;
    colvals[5291] = 186;
    colvals[5292] = 187;
    colvals[5293] = 188;
    colvals[5294] = 189;
    colvals[5295] = 190;
    colvals[5296] = 191;
    colvals[5297] = 192;
    colvals[5298] = 193;
    colvals[5299] = 194;
    colvals[5300] = 195;
    colvals[5301] = 196;
    colvals[5302] = 198;
    colvals[5303] = 199;
    colvals[5304] = 200;
    colvals[5305] = 201;
    colvals[5306] = 202;
    colvals[5307] = 203;
    colvals[5308] = 205;
    colvals[5309] = 206;
    colvals[5310] = 207;
    colvals[5311] = 208;
    colvals[5312] = 209;
    colvals[5313] = 210;
    colvals[5314] = 211;
    colvals[5315] = 213;
    colvals[5316] = 214;
    colvals[5317] = 53;
    colvals[5318] = 54;
    colvals[5319] = 55;
    colvals[5320] = 56;
    colvals[5321] = 59;
    colvals[5322] = 65;
    colvals[5323] = 66;
    colvals[5324] = 67;
    colvals[5325] = 70;
    colvals[5326] = 74;
    colvals[5327] = 81;
    colvals[5328] = 85;
    colvals[5329] = 87;
    colvals[5330] = 89;
    colvals[5331] = 90;
    colvals[5332] = 92;
    colvals[5333] = 93;
    colvals[5334] = 99;
    colvals[5335] = 101;
    colvals[5336] = 103;
    colvals[5337] = 104;
    colvals[5338] = 106;
    colvals[5339] = 109;
    colvals[5340] = 110;
    colvals[5341] = 111;
    colvals[5342] = 112;
    colvals[5343] = 113;
    colvals[5344] = 115;
    colvals[5345] = 116;
    colvals[5346] = 117;
    colvals[5347] = 118;
    colvals[5348] = 120;
    colvals[5349] = 121;
    colvals[5350] = 122;
    colvals[5351] = 123;
    colvals[5352] = 124;
    colvals[5353] = 126;
    colvals[5354] = 127;
    colvals[5355] = 128;
    colvals[5356] = 131;
    colvals[5357] = 133;
    colvals[5358] = 134;
    colvals[5359] = 135;
    colvals[5360] = 136;
    colvals[5361] = 137;
    colvals[5362] = 138;
    colvals[5363] = 139;
    colvals[5364] = 140;
    colvals[5365] = 143;
    colvals[5366] = 144;
    colvals[5367] = 145;
    colvals[5368] = 147;
    colvals[5369] = 148;
    colvals[5370] = 150;
    colvals[5371] = 151;
    colvals[5372] = 152;
    colvals[5373] = 154;
    colvals[5374] = 155;
    colvals[5375] = 156;
    colvals[5376] = 157;
    colvals[5377] = 158;
    colvals[5378] = 159;
    colvals[5379] = 160;
    colvals[5380] = 161;
    colvals[5381] = 162;
    colvals[5382] = 163;
    colvals[5383] = 164;
    colvals[5384] = 165;
    colvals[5385] = 167;
    colvals[5386] = 168;
    colvals[5387] = 169;
    colvals[5388] = 170;
    colvals[5389] = 171;
    colvals[5390] = 172;
    colvals[5391] = 174;
    colvals[5392] = 175;
    colvals[5393] = 176;
    colvals[5394] = 177;
    colvals[5395] = 178;
    colvals[5396] = 179;
    colvals[5397] = 180;
    colvals[5398] = 181;
    colvals[5399] = 182;
    colvals[5400] = 183;
    colvals[5401] = 184;
    colvals[5402] = 185;
    colvals[5403] = 186;
    colvals[5404] = 187;
    colvals[5405] = 188;
    colvals[5406] = 189;
    colvals[5407] = 190;
    colvals[5408] = 191;
    colvals[5409] = 192;
    colvals[5410] = 193;
    colvals[5411] = 194;
    colvals[5412] = 195;
    colvals[5413] = 196;
    colvals[5414] = 197;
    colvals[5415] = 198;
    colvals[5416] = 199;
    colvals[5417] = 200;
    colvals[5418] = 201;
    colvals[5419] = 202;
    colvals[5420] = 203;
    colvals[5421] = 205;
    colvals[5422] = 206;
    colvals[5423] = 207;
    colvals[5424] = 208;
    colvals[5425] = 210;
    colvals[5426] = 211;
    colvals[5427] = 212;
    colvals[5428] = 213;
    colvals[5429] = 214;
    colvals[5430] = 53;
    colvals[5431] = 55;
    colvals[5432] = 56;
    colvals[5433] = 58;
    colvals[5434] = 59;
    colvals[5435] = 60;
    colvals[5436] = 61;
    colvals[5437] = 65;
    colvals[5438] = 66;
    colvals[5439] = 68;
    colvals[5440] = 69;
    colvals[5441] = 70;
    colvals[5442] = 71;
    colvals[5443] = 72;
    colvals[5444] = 74;
    colvals[5445] = 75;
    colvals[5446] = 76;
    colvals[5447] = 77;
    colvals[5448] = 79;
    colvals[5449] = 82;
    colvals[5450] = 83;
    colvals[5451] = 84;
    colvals[5452] = 85;
    colvals[5453] = 86;
    colvals[5454] = 87;
    colvals[5455] = 88;
    colvals[5456] = 90;
    colvals[5457] = 92;
    colvals[5458] = 94;
    colvals[5459] = 96;
    colvals[5460] = 97;
    colvals[5461] = 98;
    colvals[5462] = 100;
    colvals[5463] = 102;
    colvals[5464] = 105;
    colvals[5465] = 107;
    colvals[5466] = 109;
    colvals[5467] = 110;
    colvals[5468] = 111;
    colvals[5469] = 113;
    colvals[5470] = 115;
    colvals[5471] = 116;
    colvals[5472] = 119;
    colvals[5473] = 120;
    colvals[5474] = 124;
    colvals[5475] = 125;
    colvals[5476] = 126;
    colvals[5477] = 129;
    colvals[5478] = 130;
    colvals[5479] = 131;
    colvals[5480] = 132;
    colvals[5481] = 133;
    colvals[5482] = 134;
    colvals[5483] = 135;
    colvals[5484] = 136;
    colvals[5485] = 138;
    colvals[5486] = 139;
    colvals[5487] = 140;
    colvals[5488] = 141;
    colvals[5489] = 142;
    colvals[5490] = 143;
    colvals[5491] = 144;
    colvals[5492] = 145;
    colvals[5493] = 146;
    colvals[5494] = 147;
    colvals[5495] = 148;
    colvals[5496] = 149;
    colvals[5497] = 150;
    colvals[5498] = 151;
    colvals[5499] = 153;
    colvals[5500] = 154;
    colvals[5501] = 155;
    colvals[5502] = 156;
    colvals[5503] = 157;
    colvals[5504] = 158;
    colvals[5505] = 160;
    colvals[5506] = 161;
    colvals[5507] = 162;
    colvals[5508] = 164;
    colvals[5509] = 165;
    colvals[5510] = 166;
    colvals[5511] = 167;
    colvals[5512] = 168;
    colvals[5513] = 169;
    colvals[5514] = 170;
    colvals[5515] = 171;
    colvals[5516] = 172;
    colvals[5517] = 173;
    colvals[5518] = 174;
    colvals[5519] = 175;
    colvals[5520] = 176;
    colvals[5521] = 177;
    colvals[5522] = 178;
    colvals[5523] = 179;
    colvals[5524] = 180;
    colvals[5525] = 182;
    colvals[5526] = 183;
    colvals[5527] = 184;
    colvals[5528] = 185;
    colvals[5529] = 186;
    colvals[5530] = 187;
    colvals[5531] = 188;
    colvals[5532] = 190;
    colvals[5533] = 191;
    colvals[5534] = 192;
    colvals[5535] = 194;
    colvals[5536] = 195;
    colvals[5537] = 196;
    colvals[5538] = 197;
    colvals[5539] = 198;
    colvals[5540] = 199;
    colvals[5541] = 200;
    colvals[5542] = 201;
    colvals[5543] = 202;
    colvals[5544] = 203;
    colvals[5545] = 204;
    colvals[5546] = 205;
    colvals[5547] = 206;
    colvals[5548] = 207;
    colvals[5549] = 208;
    colvals[5550] = 209;
    colvals[5551] = 210;
    colvals[5552] = 211;
    colvals[5553] = 212;
    colvals[5554] = 213;
    colvals[5555] = 214;
    colvals[5556] = 50;
    colvals[5557] = 51;
    colvals[5558] = 53;
    colvals[5559] = 54;
    colvals[5560] = 55;
    colvals[5561] = 56;
    colvals[5562] = 58;
    colvals[5563] = 59;
    colvals[5564] = 60;
    colvals[5565] = 64;
    colvals[5566] = 65;
    colvals[5567] = 66;
    colvals[5568] = 67;
    colvals[5569] = 68;
    colvals[5570] = 70;
    colvals[5571] = 71;
    colvals[5572] = 72;
    colvals[5573] = 73;
    colvals[5574] = 74;
    colvals[5575] = 75;
    colvals[5576] = 76;
    colvals[5577] = 77;
    colvals[5578] = 78;
    colvals[5579] = 79;
    colvals[5580] = 83;
    colvals[5581] = 84;
    colvals[5582] = 85;
    colvals[5583] = 86;
    colvals[5584] = 87;
    colvals[5585] = 90;
    colvals[5586] = 91;
    colvals[5587] = 92;
    colvals[5588] = 93;
    colvals[5589] = 94;
    colvals[5590] = 95;
    colvals[5591] = 96;
    colvals[5592] = 97;
    colvals[5593] = 98;
    colvals[5594] = 99;
    colvals[5595] = 100;
    colvals[5596] = 101;
    colvals[5597] = 103;
    colvals[5598] = 106;
    colvals[5599] = 107;
    colvals[5600] = 108;
    colvals[5601] = 109;
    colvals[5602] = 110;
    colvals[5603] = 111;
    colvals[5604] = 112;
    colvals[5605] = 113;
    colvals[5606] = 114;
    colvals[5607] = 115;
    colvals[5608] = 116;
    colvals[5609] = 118;
    colvals[5610] = 120;
    colvals[5611] = 121;
    colvals[5612] = 122;
    colvals[5613] = 123;
    colvals[5614] = 124;
    colvals[5615] = 125;
    colvals[5616] = 126;
    colvals[5617] = 127;
    colvals[5618] = 128;
    colvals[5619] = 130;
    colvals[5620] = 131;
    colvals[5621] = 132;
    colvals[5622] = 133;
    colvals[5623] = 134;
    colvals[5624] = 135;
    colvals[5625] = 136;
    colvals[5626] = 137;
    colvals[5627] = 138;
    colvals[5628] = 139;
    colvals[5629] = 140;
    colvals[5630] = 143;
    colvals[5631] = 144;
    colvals[5632] = 145;
    colvals[5633] = 146;
    colvals[5634] = 147;
    colvals[5635] = 148;
    colvals[5636] = 149;
    colvals[5637] = 150;
    colvals[5638] = 151;
    colvals[5639] = 152;
    colvals[5640] = 153;
    colvals[5641] = 154;
    colvals[5642] = 155;
    colvals[5643] = 156;
    colvals[5644] = 157;
    colvals[5645] = 158;
    colvals[5646] = 159;
    colvals[5647] = 160;
    colvals[5648] = 161;
    colvals[5649] = 162;
    colvals[5650] = 163;
    colvals[5651] = 164;
    colvals[5652] = 165;
    colvals[5653] = 166;
    colvals[5654] = 167;
    colvals[5655] = 168;
    colvals[5656] = 169;
    colvals[5657] = 170;
    colvals[5658] = 171;
    colvals[5659] = 172;
    colvals[5660] = 174;
    colvals[5661] = 175;
    colvals[5662] = 176;
    colvals[5663] = 177;
    colvals[5664] = 178;
    colvals[5665] = 179;
    colvals[5666] = 180;
    colvals[5667] = 181;
    colvals[5668] = 182;
    colvals[5669] = 183;
    colvals[5670] = 184;
    colvals[5671] = 185;
    colvals[5672] = 186;
    colvals[5673] = 187;
    colvals[5674] = 188;
    colvals[5675] = 189;
    colvals[5676] = 190;
    colvals[5677] = 191;
    colvals[5678] = 192;
    colvals[5679] = 193;
    colvals[5680] = 194;
    colvals[5681] = 195;
    colvals[5682] = 196;
    colvals[5683] = 197;
    colvals[5684] = 198;
    colvals[5685] = 199;
    colvals[5686] = 200;
    colvals[5687] = 201;
    colvals[5688] = 202;
    colvals[5689] = 203;
    colvals[5690] = 204;
    colvals[5691] = 205;
    colvals[5692] = 206;
    colvals[5693] = 207;
    colvals[5694] = 208;
    colvals[5695] = 209;
    colvals[5696] = 210;
    colvals[5697] = 211;
    colvals[5698] = 212;
    colvals[5699] = 213;
    colvals[5700] = 214;
    
    // value of each non-zero element
    data[0] = 0.0 - k[2339] - k[2340] - k[2341] - k[2342];
    data[1] = 0.0 + k[2146];
    data[2] = 0.0 - k[2351] - k[2352] - k[2353] - k[2354];
    data[3] = 0.0 + k[2300];
    data[4] = 0.0 - k[2359] - k[2360] - k[2361] - k[2362];
    data[5] = 0.0 + k[2301];
    data[6] = 0.0 - k[2399] - k[2400] - k[2401] - k[2402];
    data[7] = 0.0 + k[2199];
    data[8] = 0.0 - k[2463] - k[2464] - k[2465] - k[2466];
    data[9] = 0.0 + k[2200];
    data[10] = 0.0 - k[2415] - k[2416] - k[2417] - k[2418];
    data[11] = 0.0 + k[2147];
    data[12] = 0.0 - k[2411] - k[2412] - k[2413] - k[2414];
    data[13] = 0.0 + k[2299];
    data[14] = 0.0 - k[2431] - k[2432] - k[2433] - k[2434];
    data[15] = 0.0 - k[2355] - k[2356] - k[2357] - k[2358];
    data[16] = 0.0 + k[2275];
    data[17] = 0.0 - k[2335] - k[2336] - k[2337] - k[2338];
    data[18] = 0.0 + k[2297];
    data[19] = 0.0 - k[2447] - k[2448] - k[2449] - k[2450];
    data[20] = 0.0 + k[2270];
    data[21] = 0.0 - k[2427] - k[2428] - k[2429] - k[2430];
    data[22] = 0.0 - k[2155];
    data[23] = 0.0 + k[2419] + k[2420] + k[2421] + k[2422];
    data[24] = 0.0 - k[2323] - k[2324] - k[2325] - k[2326];
    data[25] = 0.0 + k[2242];
    data[26] = 0.0 + k[2224];
    data[27] = 0.0 - k[2327] - k[2328] - k[2329] - k[2330];
    data[28] = 0.0 + k[2298];
    data[29] = 0.0 + k[2277];
    data[30] = 0.0 - k[2491] - k[2492] - k[2493] - k[2494];
    data[31] = 0.0 + k[2166];
    data[32] = 0.0 + k[2165];
    data[33] = 0.0 - k[2343] - k[2344] - k[2345] - k[2346];
    data[34] = 0.0 + k[2234];
    data[35] = 0.0 + k[2207];
    data[36] = 0.0 - k[2467] - k[2468] - k[2469] - k[2470];
    data[37] = 0.0 + k[2161];
    data[38] = 0.0 + k[2145];
    data[39] = 0.0 - k[2423] - k[2424] - k[2425] - k[2426];
    data[40] = 0.0 + k[2157];
    data[41] = 0.0 + k[2158];
    data[42] = 0.0 - k[2319] - k[2320] - k[2321] - k[2322];
    data[43] = 0.0 + k[2287];
    data[44] = 0.0 + k[2286];
    data[45] = 0.0 - k[2471] - k[2472] - k[2473] - k[2474];
    data[46] = 0.0 + k[2163];
    data[47] = 0.0 + k[2164];
    data[48] = 0.0 - k[2363] - k[2364] - k[2365] - k[2366];
    data[49] = 0.0 + k[2236];
    data[50] = 0.0 + k[2212];
    data[51] = 0.0 - k[2375] - k[2376] - k[2377] - k[2378];
    data[52] = 0.0 + k[2229];
    data[53] = 0.0 + k[2253];
    data[54] = 0.0 - k[2387] - k[2388] - k[2389] - k[2390];
    data[55] = 0.0 + k[2294];
    data[56] = 0.0 + k[2274];
    data[57] = 0.0 - k[2407] - k[2408] - k[2409] - k[2410];
    data[58] = 0.0 + k[2183];
    data[59] = 0.0 + k[2181];
    data[60] = 0.0 - k[2475] - k[2476] - k[2477] - k[2478];
    data[61] = 0.0 + k[2184];
    data[62] = 0.0 + k[2182];
    data[63] = 0.0 - k[2495] - k[2496] - k[2497] - k[2498];
    data[64] = 0.0 + k[2186];
    data[65] = 0.0 + k[2185];
    data[66] = 0.0 - k[2439] - k[2440] - k[2441] - k[2442];
    data[67] = 0.0 + k[2151];
    data[68] = 0.0 + k[2150];
    data[69] = 0.0 - k[2403] - k[2404] - k[2405] - k[2406];
    data[70] = 0.0 + k[2148];
    data[71] = 0.0 + k[2149];
    data[72] = 0.0 - k[2435] - k[2436] - k[2437] - k[2438];
    data[73] = 0.0 + k[2247];
    data[74] = 0.0 + k[2215];
    data[75] = 0.0 - k[2479] - k[2480] - k[2481] - k[2482];
    data[76] = 0.0 + k[2167];
    data[77] = 0.0 + k[2153];
    data[78] = 0.0 - k[2315] - k[2316] - k[2317] - k[2318];
    data[79] = 0.0 + k[2292];
    data[80] = 0.0 + k[2228];
    data[81] = 0.0 + k[2209];
    data[82] = 0.0 - k[2455] - k[2456] - k[2457] - k[2458];
    data[83] = 0.0 + k[2192];
    data[84] = 0.0 + k[2187];
    data[85] = 0.0 + k[2188];
    data[86] = 0.0 + k[2171];
    data[87] = 0.0 - k[2371] - k[2372] - k[2373] - k[2374];
    data[88] = 0.0 + k[2273];
    data[89] = 0.0 + k[2271];
    data[90] = 0.0 + k[2272];
    data[91] = 0.0 - k[2347] - k[2348] - k[2349] - k[2350];
    data[92] = 0.0 + k[2290];
    data[93] = 0.0 + k[2230];
    data[94] = 0.0 + k[2219];
    data[95] = 0.0 - k[2451] - k[2452] - k[2453] - k[2454];
    data[96] = 0.0 + k[2291];
    data[97] = 0.0 + k[2276];
    data[98] = 0.0 + k[2262];
    data[99] = 0.0 - k[2487] - k[2488] - k[2489] - k[2490];
    data[100] = 0.0 + k[2285];
    data[101] = 0.0 + k[2269];
    data[102] = 0.0 + k[2259];
    data[103] = 0.0 - k[2483] - k[2484] - k[2485] - k[2486];
    data[104] = 0.0 + k[2191];
    data[105] = 0.0 + k[2190];
    data[106] = 0.0 + k[2189];
    data[107] = 0.0 - k[2459] - k[2460] - k[2461] - k[2462];
    data[108] = 0.0 + k[2283];
    data[109] = 0.0 + k[2256];
    data[110] = 0.0 + k[2267];
    data[111] = 0.0 - k[2499] - k[2500] - k[2501] - k[2502];
    data[112] = 0.0 + k[2295];
    data[113] = 0.0 + k[2284];
    data[114] = 0.0 + k[2260];
    data[115] = 0.0 + k[2155];
    data[116] = 0.0 - k[2419] - k[2420] - k[2421] - k[2422];
    data[117] = 0.0 + k[2162];
    data[118] = 0.0 + k[2156];
    data[119] = 0.0 + k[2159];
    data[120] = 0.0 + k[2160];
    data[121] = 0.0 - k[2379] - k[2380] - k[2381] - k[2382];
    data[122] = 0.0 + k[2152];
    data[123] = 0.0 + k[2154];
    data[124] = 0.0 + k[2202];
    data[125] = 0.0 + k[2296];
    data[126] = 0.0 - k[2367] - k[2368] - k[2369] - k[2370];
    data[127] = 0.0 + k[2168];
    data[128] = 0.0 + k[2244];
    data[129] = 0.0 + k[2208];
    data[130] = 0.0 + k[2218];
    data[131] = 0.0 + k[2240];
    data[132] = 0.0 + k[2169];
    data[133] = 0.0 - k[2503] - k[2504] - k[2505] - k[2506];
    data[134] = 0.0 + k[2203];
    data[135] = 0.0 + k[2280];
    data[136] = 0.0 + k[2204];
    data[137] = 0.0 + k[2205];
    data[138] = 0.0 + k[2281];
    data[139] = 0.0 + k[2201];
    data[140] = 0.0 - k[2395] - k[2396] - k[2397] - k[2398];
    data[141] = 0.0 + k[2197];
    data[142] = 0.0 + k[2194];
    data[143] = 0.0 + k[2196];
    data[144] = 0.0 + k[2198];
    data[145] = 0.0 + k[2195];
    data[146] = 0.0 - k[2331] - k[2332] - k[2333] - k[2334];
    data[147] = 0.0 + k[2235];
    data[148] = 0.0 + k[2289];
    data[149] = 0.0 + k[2241];
    data[150] = 0.0 + k[2220];
    data[151] = 0.0 + k[2223];
    data[152] = 0.0 - k[2443] - k[2444] - k[2445] - k[2446];
    data[153] = 0.0 + k[2282];
    data[154] = 0.0 + k[2263];
    data[155] = 0.0 + k[2279];
    data[156] = 0.0 + k[2258];
    data[157] = 0.0 + k[2266];
    data[158] = 0.0 + k[2268];
    data[159] = 0.0 + k[2255];
    data[160] = 0.0 - k[2311] - k[2312] - k[2313] - k[2314];
    data[161] = 0.0 + k[2227];
    data[162] = 0.0 + k[2239];
    data[163] = 0.0 + k[2233];
    data[164] = 0.0 + k[2246];
    data[165] = 0.0 + k[2211];
    data[166] = 0.0 + k[2252];
    data[167] = 0.0 + k[2214];
    data[168] = 0.0 - k[2391] - k[2392] - k[2393] - k[2394];
    data[169] = 0.0 + k[2278];
    data[170] = 0.0 + k[2293];
    data[171] = 0.0 + k[2254];
    data[172] = 0.0 + k[2265];
    data[173] = 0.0 + k[2257];
    data[174] = 0.0 + k[2264];
    data[175] = 0.0 + k[2261];
    data[176] = 0.0 + k[2491] + k[2492] + k[2493] + k[2494];
    data[177] = 0.0 - k[1800]*y[IDX_NI] - k[1878]*y[IDX_OI] - k[2166];
    data[178] = 0.0 + k[1799]*y[IDX_NI];
    data[179] = 0.0 + k[1799]*y[IDX_C4HI] - k[1800]*y[IDX_C4NI];
    data[180] = 0.0 - k[1878]*y[IDX_C4NI];
    data[181] = 0.0 + k[2399] + k[2400] + k[2401] + k[2402];
    data[182] = 0.0 - k[1585]*y[IDX_CI] - k[1797]*y[IDX_NI] - k[2199];
    data[183] = 0.0 + k[1582]*y[IDX_CI];
    data[184] = 0.0 + k[1674]*y[IDX_CHI];
    data[185] = 0.0 + k[1674]*y[IDX_C2H2I];
    data[186] = 0.0 - k[1797]*y[IDX_C3H2I];
    data[187] = 0.0 + k[1582]*y[IDX_C2H3I] - k[1585]*y[IDX_C3H2I];
    data[188] = 0.0 + k[2503] + k[2504] + k[2505] + k[2506];
    data[189] = 0.0 - k[122]*y[IDX_HII] - k[402] - k[1224]*y[IDX_HeII] -
        k[1225]*y[IDX_HeII] - k[2019] - k[2203];
    data[190] = 0.0 - k[1224]*y[IDX_H2S2I] - k[1225]*y[IDX_H2S2I];
    data[191] = 0.0 - k[122]*y[IDX_H2S2I];
    data[192] = 0.0 - k[2307] - k[2308] - k[2309] - k[2310];
    data[193] = 0.0 + k[2232];
    data[194] = 0.0 + k[2250];
    data[195] = 0.0 + k[2226];
    data[196] = 0.0 + k[2238];
    data[197] = 0.0 + k[2288];
    data[198] = 0.0 + k[2243];
    data[199] = 0.0 + k[2222];
    data[200] = 0.0 + k[2225];
    data[201] = 0.0 + k[2251];
    data[202] = 0.0 - k[510]*y[IDX_EM] - k[511]*y[IDX_EM] - k[2273];
    data[203] = 0.0 + k[1051]*y[IDX_H3II];
    data[204] = 0.0 + k[1390]*y[IDX_O2I];
    data[205] = 0.0 + k[1390]*y[IDX_NH2II];
    data[206] = 0.0 + k[1051]*y[IDX_HNOI];
    data[207] = 0.0 - k[510]*y[IDX_H2NOII] - k[511]*y[IDX_H2NOII];
    data[208] = 0.0 + k[2455] + k[2456] + k[2457] + k[2458];
    data[209] = 0.0 - k[405] - k[896]*y[IDX_HII] - k[1228]*y[IDX_HeII] - k[2023] -
        k[2024] - k[2192];
    data[210] = 0.0 + k[1924]*y[IDX_OI];
    data[211] = 0.0 - k[1228]*y[IDX_H2SiOI];
    data[212] = 0.0 - k[896]*y[IDX_H2SiOI];
    data[213] = 0.0 + k[1924]*y[IDX_SiH3I];
    data[214] = 0.0 - k[548]*y[IDX_EM] - k[948]*y[IDX_H2I] - k[2197];
    data[215] = 0.0 + k[1040]*y[IDX_H3II];
    data[216] = 0.0 + k[944]*y[IDX_H2I];
    data[217] = 0.0 + k[126]*y[IDX_HII] + k[2035];
    data[218] = 0.0 + k[1040]*y[IDX_ClI];
    data[219] = 0.0 + k[126]*y[IDX_HClI];
    data[220] = 0.0 + k[944]*y[IDX_ClII] - k[948]*y[IDX_HClII];
    data[221] = 0.0 - k[548]*y[IDX_HClII];
    data[222] = 0.0 - k[563]*y[IDX_EM] - k[951]*y[IDX_H2I] - k[1106]*y[IDX_HI];
    data[223] = 0.0 + k[926]*y[IDX_HeI];
    data[224] = 0.0 + k[1236]*y[IDX_HeII];
    data[225] = 0.0 + k[1236]*y[IDX_HCOI];
    data[226] = 0.0 + k[926]*y[IDX_H2II] + k[2111]*y[IDX_HII];
    data[227] = 0.0 + k[2111]*y[IDX_HeI];
    data[228] = 0.0 - k[951]*y[IDX_HeHII];
    data[229] = 0.0 - k[563]*y[IDX_HeHII];
    data[230] = 0.0 - k[1106]*y[IDX_HeHII];
    data[231] = 0.0 + k[2423] + k[2424] + k[2425] + k[2426];
    data[232] = 0.0 - k[415] - k[900]*y[IDX_HII] - k[1788]*y[IDX_CI] - k[2037] - k[2157];
    data[233] = 0.0 + k[1631]*y[IDX_NOI];
    data[234] = 0.0 + k[1631]*y[IDX_CH2I];
    data[235] = 0.0 - k[900]*y[IDX_HNCOI];
    data[236] = 0.0 - k[1788]*y[IDX_HNCOI];
    data[237] = 0.0 - k[550]*y[IDX_EM] - k[2291];
    data[238] = 0.0 + k[1061]*y[IDX_H3II] + k[1148]*y[IDX_HCOII];
    data[239] = 0.0 + k[1392]*y[IDX_SI];
    data[240] = 0.0 + k[1392]*y[IDX_NH2II];
    data[241] = 0.0 + k[1061]*y[IDX_NSI];
    data[242] = 0.0 + k[1148]*y[IDX_NSI];
    data[243] = 0.0 - k[550]*y[IDX_HNSII];
    data[244] = 0.0 - k[5]*y[IDX_H2I] - k[551]*y[IDX_EM] - k[2168];
    data[245] = 0.0 + k[942]*y[IDX_H2I];
    data[246] = 0.0 + k[621]*y[IDX_H2OI];
    data[247] = 0.0 + k[1038]*y[IDX_COI];
    data[248] = 0.0 + k[621]*y[IDX_CII];
    data[249] = 0.0 + k[1038]*y[IDX_H3II];
    data[250] = 0.0 - k[5]*y[IDX_HOCII] + k[942]*y[IDX_COII];
    data[251] = 0.0 - k[551]*y[IDX_HOCII];
    data[252] = 0.0 - k[557]*y[IDX_EM] - k[2283];
    data[253] = 0.0 + k[1070]*y[IDX_H3II] + k[1152]*y[IDX_HCOII];
    data[254] = 0.0 + k[988]*y[IDX_SI];
    data[255] = 0.0 + k[988]*y[IDX_H2OII];
    data[256] = 0.0 + k[1070]*y[IDX_SOI];
    data[257] = 0.0 + k[1152]*y[IDX_SOI];
    data[258] = 0.0 - k[557]*y[IDX_HSOII];
    data[259] = 0.0 - k[590]*y[IDX_EM] - k[591]*y[IDX_EM] - k[2186];
    data[260] = 0.0 + k[28]*y[IDX_CII] + k[145]*y[IDX_HII];
    data[261] = 0.0 + k[28]*y[IDX_SiC3I];
    data[262] = 0.0 + k[145]*y[IDX_SiC3I];
    data[263] = 0.0 - k[590]*y[IDX_SiC3II] - k[591]*y[IDX_SiC3II];
    data[264] = 0.0 - k[2303] - k[2304] - k[2305] - k[2306];
    data[265] = 0.0 + k[2249];
    data[266] = 0.0 + k[2248];
    data[267] = 0.0 + k[2237];
    data[268] = 0.0 + k[2245];
    data[269] = 0.0 + k[2213];
    data[270] = 0.0 + k[2216];
    data[271] = 0.0 + k[2231];
    data[272] = 0.0 + k[2217];
    data[273] = 0.0 + k[2210];
    data[274] = 0.0 + k[2221];
    data[275] = 0.0 + k[2206];
    data[276] = 0.0 - k[2383] - k[2384] - k[2385] - k[2386];
    data[277] = 0.0 + k[2193];
    data[278] = 0.0 + k[2180];
    data[279] = 0.0 + k[2175];
    data[280] = 0.0 + k[2178];
    data[281] = 0.0 + k[2177];
    data[282] = 0.0 + k[2176];
    data[283] = 0.0 + k[2172];
    data[284] = 0.0 + k[2174];
    data[285] = 0.0 + k[2179];
    data[286] = 0.0 + k[2170];
    data[287] = 0.0 + k[2173];
    data[288] = 0.0 + k[2463] + k[2464] + k[2465] + k[2466];
    data[289] = 0.0 + k[1585]*y[IDX_CI];
    data[290] = 0.0 - k[378] - k[1191]*y[IDX_HeII] - k[1799]*y[IDX_NI] - k[1975] -
        k[2200];
    data[291] = 0.0 + k[1570]*y[IDX_C2I];
    data[292] = 0.0 + k[1570]*y[IDX_C2H2I];
    data[293] = 0.0 - k[1799]*y[IDX_C4HI];
    data[294] = 0.0 - k[1191]*y[IDX_C4HI];
    data[295] = 0.0 + k[1585]*y[IDX_C3H2I];
    data[296] = 0.0 + k[548]*y[IDX_EM];
    data[297] = 0.0 - k[109]*y[IDX_HII] - k[358] - k[397] - k[1040]*y[IDX_H3II] -
        k[1720]*y[IDX_H2I] - k[2008] - k[2194];
    data[298] = 0.0 + k[108]*y[IDX_HI] + k[324]*y[IDX_O2I] + k[2134]*y[IDX_EM];
    data[299] = 0.0 + k[508]*y[IDX_EM];
    data[300] = 0.0 + k[413] + k[1787]*y[IDX_HI] + k[2034];
    data[301] = 0.0 + k[324]*y[IDX_ClII];
    data[302] = 0.0 - k[1040]*y[IDX_ClI];
    data[303] = 0.0 - k[109]*y[IDX_ClI];
    data[304] = 0.0 - k[1720]*y[IDX_ClI];
    data[305] = 0.0 + k[508]*y[IDX_H2ClII] + k[548]*y[IDX_HClII] + k[2134]*y[IDX_ClII];
    data[306] = 0.0 + k[108]*y[IDX_ClII] + k[1787]*y[IDX_HClI];
    data[307] = 0.0 + k[109]*y[IDX_HII] + k[358] + k[397] + k[2008];
    data[308] = 0.0 - k[108]*y[IDX_HI] - k[324]*y[IDX_O2I] - k[944]*y[IDX_H2I] -
        k[2134]*y[IDX_EM] - k[2196];
    data[309] = 0.0 + k[1241]*y[IDX_HeII];
    data[310] = 0.0 - k[324]*y[IDX_ClII];
    data[311] = 0.0 + k[1241]*y[IDX_HClI];
    data[312] = 0.0 + k[109]*y[IDX_ClI];
    data[313] = 0.0 - k[944]*y[IDX_ClII];
    data[314] = 0.0 - k[2134]*y[IDX_ClII];
    data[315] = 0.0 - k[108]*y[IDX_ClII];
    data[316] = 0.0 + k[2355] + k[2356] + k[2357] + k[2358];
    data[317] = 0.0 - k[398] - k[1593]*y[IDX_CI] - k[1746]*y[IDX_HI] - k[1810]*y[IDX_NI]
        - k[1885]*y[IDX_OI] - k[2010] - k[2275];
    data[318] = 0.0 + k[1794]*y[IDX_NI];
    data[319] = 0.0 + k[1804]*y[IDX_NI];
    data[320] = 0.0 + k[1794]*y[IDX_C2H5I] + k[1804]*y[IDX_CH3I] - k[1810]*y[IDX_H2CNI];
    data[321] = 0.0 - k[1885]*y[IDX_H2CNI];
    data[322] = 0.0 - k[1593]*y[IDX_H2CNI];
    data[323] = 0.0 - k[1746]*y[IDX_H2CNI];
    data[324] = 0.0 - k[470]*y[IDX_EM] - k[471]*y[IDX_EM] - k[675]*y[IDX_C2H2I] -
        k[1097]*y[IDX_HI] - k[1118]*y[IDX_HCNI] - k[2163];
    data[325] = 0.0 + k[1305]*y[IDX_NII] + k[2046];
    data[326] = 0.0 + k[866]*y[IDX_HCNI];
    data[327] = 0.0 - k[675]*y[IDX_C2N2II];
    data[328] = 0.0 + k[1305]*y[IDX_NCCNI];
    data[329] = 0.0 + k[866]*y[IDX_CNII] - k[1118]*y[IDX_C2N2II];
    data[330] = 0.0 - k[470]*y[IDX_C2N2II] - k[471]*y[IDX_C2N2II];
    data[331] = 0.0 - k[1097]*y[IDX_C2N2II];
    data[332] = 0.0 - k[475]*y[IDX_EM] - k[476]*y[IDX_EM] - k[994]*y[IDX_H2OI] - k[2165];
    data[333] = 0.0 + k[1119]*y[IDX_HCNI];
    data[334] = 0.0 + k[625]*y[IDX_CII];
    data[335] = 0.0 + k[1119]*y[IDX_C3II];
    data[336] = 0.0 + k[625]*y[IDX_HC3NI];
    data[337] = 0.0 - k[994]*y[IDX_C4NII];
    data[338] = 0.0 - k[475]*y[IDX_C4NII] - k[476]*y[IDX_C4NII];
    data[339] = 0.0 - k[600]*y[IDX_EM] - k[601]*y[IDX_EM] - k[1016]*y[IDX_H2OI] -
        k[2193];
    data[340] = 0.0 + k[963]*y[IDX_H2I];
    data[341] = 0.0 + k[2120]*y[IDX_H2I];
    data[342] = 0.0 + k[1074]*y[IDX_H3II] + k[1154]*y[IDX_HCOII];
    data[343] = 0.0 + k[1074]*y[IDX_SiH4I];
    data[344] = 0.0 + k[1154]*y[IDX_SiH4I];
    data[345] = 0.0 - k[1016]*y[IDX_SiH5II];
    data[346] = 0.0 + k[963]*y[IDX_SiH4II] + k[2120]*y[IDX_SiH3II];
    data[347] = 0.0 - k[600]*y[IDX_SiH5II] - k[601]*y[IDX_SiH5II];
    data[348] = 0.0 + k[948]*y[IDX_H2I];
    data[349] = 0.0 - k[508]*y[IDX_EM] - k[509]*y[IDX_EM] - k[874]*y[IDX_COI] -
        k[999]*y[IDX_H2OI] - k[2198];
    data[350] = 0.0 + k[832]*y[IDX_CH5II] + k[1049]*y[IDX_H3II];
    data[351] = 0.0 + k[832]*y[IDX_HClI];
    data[352] = 0.0 + k[1049]*y[IDX_HClI];
    data[353] = 0.0 - k[999]*y[IDX_H2ClII];
    data[354] = 0.0 - k[874]*y[IDX_H2ClII];
    data[355] = 0.0 + k[948]*y[IDX_HClII];
    data[356] = 0.0 - k[508]*y[IDX_H2ClII] - k[509]*y[IDX_H2ClII];
    data[357] = 0.0 - k[536]*y[IDX_EM] - k[537]*y[IDX_EM] - k[2167];
    data[358] = 0.0 + k[1047]*y[IDX_H3II] + k[1089]*y[IDX_H3OII] + k[1145]*y[IDX_HCOII];
    data[359] = 0.0 + k[969]*y[IDX_H2COI];
    data[360] = 0.0 + k[969]*y[IDX_CH3OH2II];
    data[361] = 0.0 + k[1089]*y[IDX_HCOOCH3I];
    data[362] = 0.0 + k[1047]*y[IDX_HCOOCH3I];
    data[363] = 0.0 + k[1145]*y[IDX_HCOOCH3I];
    data[364] = 0.0 - k[536]*y[IDX_H5C2O2II] - k[537]*y[IDX_H5C2O2II];
    data[365] = 0.0 + k[2495] + k[2496] + k[2497] + k[2498];
    data[366] = 0.0 - k[28]*y[IDX_CII] - k[145]*y[IDX_HII] - k[450] -
        k[1275]*y[IDX_HeII] - k[1919]*y[IDX_OI] - k[2079] - k[2080] - k[2185];
    data[367] = 0.0 - k[28]*y[IDX_SiC3I];
    data[368] = 0.0 - k[1275]*y[IDX_SiC3I];
    data[369] = 0.0 - k[145]*y[IDX_SiC3I];
    data[370] = 0.0 - k[1919]*y[IDX_SiC3I];
    data[371] = 0.0 - k[598]*y[IDX_EM] - k[599]*y[IDX_EM] - k[880]*y[IDX_COI] -
        k[963]*y[IDX_H2I] - k[1015]*y[IDX_H2OI] - k[2180];
    data[372] = 0.0 + k[1073]*y[IDX_H3II];
    data[373] = 0.0 + k[149]*y[IDX_HII];
    data[374] = 0.0 + k[1073]*y[IDX_SiH3I];
    data[375] = 0.0 + k[149]*y[IDX_SiH4I];
    data[376] = 0.0 - k[1015]*y[IDX_SiH4II];
    data[377] = 0.0 - k[880]*y[IDX_SiH4II];
    data[378] = 0.0 - k[963]*y[IDX_SiH4II];
    data[379] = 0.0 - k[598]*y[IDX_SiH4II] - k[599]*y[IDX_SiH4II];
    data[380] = 0.0 - k[464]*y[IDX_EM] - k[465]*y[IDX_EM] - k[466]*y[IDX_EM] -
        k[467]*y[IDX_EM] - k[1420]*y[IDX_NH3I] - k[2151];
    data[381] = 0.0 + k[1081]*y[IDX_H3OII] + k[1136]*y[IDX_HCOII] + k[1164]*y[IDX_HCSII];
    data[382] = 0.0 + k[1164]*y[IDX_C2H5OHI];
    data[383] = 0.0 + k[2121]*y[IDX_H3OII];
    data[384] = 0.0 - k[1420]*y[IDX_C2H5OH2II];
    data[385] = 0.0 + k[1081]*y[IDX_C2H5OHI] + k[2121]*y[IDX_C2H4I];
    data[386] = 0.0 + k[1136]*y[IDX_C2H5OHI];
    data[387] = 0.0 - k[464]*y[IDX_C2H5OH2II] - k[465]*y[IDX_C2H5OH2II] -
        k[466]*y[IDX_C2H5OH2II] - k[467]*y[IDX_C2H5OH2II];
    data[388] = 0.0 - k[472]*y[IDX_EM] - k[2162];
    data[389] = 0.0 + k[1026]*y[IDX_H3II];
    data[390] = 0.0 + k[992]*y[IDX_H2OI];
    data[391] = 0.0 + k[1394]*y[IDX_NH2I];
    data[392] = 0.0 + k[1394]*y[IDX_C2II];
    data[393] = 0.0 + k[1329]*y[IDX_NI];
    data[394] = 0.0 + k[724]*y[IDX_HCNI];
    data[395] = 0.0 + k[724]*y[IDX_CHII];
    data[396] = 0.0 + k[1329]*y[IDX_C2H2II];
    data[397] = 0.0 + k[1026]*y[IDX_C2NI];
    data[398] = 0.0 + k[992]*y[IDX_C2NII];
    data[399] = 0.0 - k[472]*y[IDX_C2NHII];
    data[400] = 0.0 - k[474]*y[IDX_EM] - k[2148];
    data[401] = 0.0 + k[1082]*y[IDX_H3OII] + k[1137]*y[IDX_HCOII];
    data[402] = 0.0 + k[665]*y[IDX_C2H2II];
    data[403] = 0.0 + k[774]*y[IDX_CH3II];
    data[404] = 0.0 + k[774]*y[IDX_C2H4I];
    data[405] = 0.0 + k[665]*y[IDX_CH3CNI] + k[806]*y[IDX_CH4I];
    data[406] = 0.0 + k[806]*y[IDX_C2H2II];
    data[407] = 0.0 + k[1082]*y[IDX_CH3CCHI];
    data[408] = 0.0 + k[1137]*y[IDX_CH3CCHI];
    data[409] = 0.0 - k[474]*y[IDX_C3H5II];
    data[410] = 0.0 + k[2415] + k[2416] + k[2417] + k[2418];
    data[411] = 0.0 - k[383] - k[1194]*y[IDX_HeII] - k[1195]*y[IDX_HeII] -
        k[1740]*y[IDX_HI] - k[1983] - k[2147];
    data[412] = 0.0 + k[1869]*y[IDX_OI];
    data[413] = 0.0 + k[1871]*y[IDX_OI];
    data[414] = 0.0 + k[1928]*y[IDX_OHI];
    data[415] = 0.0 + k[1928]*y[IDX_C2H2I];
    data[416] = 0.0 - k[1194]*y[IDX_CH2COI] - k[1195]*y[IDX_CH2COI];
    data[417] = 0.0 + k[1869]*y[IDX_C2H3I] + k[1871]*y[IDX_C2H4I];
    data[418] = 0.0 - k[1740]*y[IDX_CH2COI];
    data[419] = 0.0 - k[558]*y[IDX_EM] - k[559]*y[IDX_EM] - k[560]*y[IDX_EM] -
        k[1008]*y[IDX_H2OI] - k[1438]*y[IDX_NH3I] - k[2295];
    data[420] = 0.0 + k[962]*y[IDX_H2I];
    data[421] = 0.0 + k[989]*y[IDX_H2OII] + k[1069]*y[IDX_H3II];
    data[422] = 0.0 + k[989]*y[IDX_SO2I];
    data[423] = 0.0 - k[1438]*y[IDX_HSO2II];
    data[424] = 0.0 + k[1069]*y[IDX_SO2I];
    data[425] = 0.0 - k[1008]*y[IDX_HSO2II];
    data[426] = 0.0 + k[962]*y[IDX_SO2II];
    data[427] = 0.0 - k[558]*y[IDX_HSO2II] - k[559]*y[IDX_HSO2II] - k[560]*y[IDX_HSO2II];
    data[428] = 0.0 + k[1800]*y[IDX_NI] + k[1878]*y[IDX_OI];
    data[429] = 0.0 + k[476]*y[IDX_EM];
    data[430] = 0.0 - k[377] - k[1190]*y[IDX_HeII] - k[1798]*y[IDX_NI] -
        k[1877]*y[IDX_OI] - k[1974] - k[2161];
    data[431] = 0.0 + k[1571]*y[IDX_HCNI];
    data[432] = 0.0 + k[1571]*y[IDX_C2I];
    data[433] = 0.0 - k[1798]*y[IDX_C3NI] + k[1800]*y[IDX_C4NI];
    data[434] = 0.0 - k[1190]*y[IDX_C3NI];
    data[435] = 0.0 - k[1877]*y[IDX_C3NI] + k[1878]*y[IDX_C4NI];
    data[436] = 0.0 + k[476]*y[IDX_C4NII];
    data[437] = 0.0 + k[2479] + k[2480] + k[2481] + k[2482];
    data[438] = 0.0 + k[537]*y[IDX_EM];
    data[439] = 0.0 - k[411] - k[1047]*y[IDX_H3II] - k[1089]*y[IDX_H3OII] -
        k[1145]*y[IDX_HCOII] - k[1238]*y[IDX_HeII] - k[2032] - k[2153];
    data[440] = 0.0 - k[1089]*y[IDX_HCOOCH3I];
    data[441] = 0.0 - k[1238]*y[IDX_HCOOCH3I];
    data[442] = 0.0 - k[1047]*y[IDX_HCOOCH3I];
    data[443] = 0.0 - k[1145]*y[IDX_HCOOCH3I];
    data[444] = 0.0 + k[537]*y[IDX_H5C2O2II];
    data[445] = 0.0 + k[1275]*y[IDX_HeII];
    data[446] = 0.0 - k[588]*y[IDX_EM] - k[589]*y[IDX_EM] - k[2184];
    data[447] = 0.0 + k[27]*y[IDX_CII] + k[144]*y[IDX_HII];
    data[448] = 0.0 + k[670]*y[IDX_C2H2II];
    data[449] = 0.0 + k[684]*y[IDX_C2HI];
    data[450] = 0.0 + k[684]*y[IDX_SiII];
    data[451] = 0.0 + k[670]*y[IDX_SiI];
    data[452] = 0.0 + k[27]*y[IDX_SiC2I];
    data[453] = 0.0 + k[1275]*y[IDX_SiC3I];
    data[454] = 0.0 + k[144]*y[IDX_SiC2I];
    data[455] = 0.0 - k[588]*y[IDX_SiC2II] - k[589]*y[IDX_SiC2II];
    data[456] = 0.0 - k[473]*y[IDX_EM] - k[1119]*y[IDX_HCNI] - k[2292];
    data[457] = 0.0 + k[1197]*y[IDX_HeII];
    data[458] = 0.0 + k[624]*y[IDX_CII];
    data[459] = 0.0 + k[647]*y[IDX_C2I] + k[837]*y[IDX_CHI];
    data[460] = 0.0 + k[685]*y[IDX_CI];
    data[461] = 0.0 + k[607]*y[IDX_CII] + k[706]*y[IDX_CHII];
    data[462] = 0.0 + k[647]*y[IDX_C2II] + k[705]*y[IDX_CHII];
    data[463] = 0.0 + k[705]*y[IDX_C2I] + k[706]*y[IDX_C2HI];
    data[464] = 0.0 - k[1119]*y[IDX_C3II];
    data[465] = 0.0 + k[837]*y[IDX_C2II];
    data[466] = 0.0 + k[607]*y[IDX_C2HI] + k[624]*y[IDX_HC3NI];
    data[467] = 0.0 + k[1197]*y[IDX_CH3CCHI];
    data[468] = 0.0 + k[685]*y[IDX_C2HII];
    data[469] = 0.0 - k[473]*y[IDX_C3II];
    data[470] = 0.0 + k[122]*y[IDX_HII];
    data[471] = 0.0 - k[517]*y[IDX_EM] - k[518]*y[IDX_EM] - k[2280];
    data[472] = 0.0 + k[1052]*y[IDX_H3II] + k[1091]*y[IDX_H3OII] + k[1146]*y[IDX_HCOII];
    data[473] = 0.0 + k[793]*y[IDX_CH3OHI];
    data[474] = 0.0 + k[793]*y[IDX_S2II];
    data[475] = 0.0 + k[1553]*y[IDX_SI];
    data[476] = 0.0 + k[1091]*y[IDX_HS2I];
    data[477] = 0.0 + k[1553]*y[IDX_H3SII];
    data[478] = 0.0 + k[1052]*y[IDX_HS2I];
    data[479] = 0.0 + k[122]*y[IDX_H2S2I];
    data[480] = 0.0 + k[1146]*y[IDX_HS2I];
    data[481] = 0.0 - k[517]*y[IDX_H2S2II] - k[518]*y[IDX_H2S2II];
    data[482] = 0.0 + k[2395] + k[2396] + k[2397] + k[2398];
    data[483] = 0.0 + k[1720]*y[IDX_H2I];
    data[484] = 0.0 + k[509]*y[IDX_EM] + k[874]*y[IDX_COI] + k[999]*y[IDX_H2OI];
    data[485] = 0.0 - k[126]*y[IDX_HII] - k[413] - k[832]*y[IDX_CH5II] -
        k[1049]*y[IDX_H3II] - k[1241]*y[IDX_HeII] - k[1787]*y[IDX_HI] - k[2034]
        - k[2035] - k[2195];
    data[486] = 0.0 - k[832]*y[IDX_HClI];
    data[487] = 0.0 - k[1241]*y[IDX_HClI];
    data[488] = 0.0 - k[1049]*y[IDX_HClI];
    data[489] = 0.0 - k[126]*y[IDX_HClI];
    data[490] = 0.0 + k[999]*y[IDX_H2ClII];
    data[491] = 0.0 + k[874]*y[IDX_H2ClII];
    data[492] = 0.0 + k[1720]*y[IDX_ClI];
    data[493] = 0.0 + k[509]*y[IDX_H2ClII];
    data[494] = 0.0 - k[1787]*y[IDX_HClI];
    data[495] = 0.0 - k[552]*y[IDX_EM] - k[553]*y[IDX_EM] - k[1006]*y[IDX_H2OI] -
        k[2285];
    data[496] = 0.0 + k[788]*y[IDX_CH3II];
    data[497] = 0.0 + k[802]*y[IDX_OCSI];
    data[498] = 0.0 + k[737]*y[IDX_CHII] + k[802]*y[IDX_CH4II] + k[1065]*y[IDX_H3II] +
        k[1149]*y[IDX_HCOII];
    data[499] = 0.0 + k[788]*y[IDX_SOI];
    data[500] = 0.0 + k[737]*y[IDX_OCSI];
    data[501] = 0.0 + k[1065]*y[IDX_OCSI];
    data[502] = 0.0 + k[1149]*y[IDX_OCSI];
    data[503] = 0.0 - k[1006]*y[IDX_HOCSII];
    data[504] = 0.0 - k[552]*y[IDX_HOCSII] - k[553]*y[IDX_HOCSII];
    data[505] = 0.0 + k[517]*y[IDX_EM];
    data[506] = 0.0 - k[127]*y[IDX_HII] - k[417] - k[1052]*y[IDX_H3II] -
        k[1091]*y[IDX_H3OII] - k[1146]*y[IDX_HCOII] - k[1247]*y[IDX_HeII] -
        k[1248]*y[IDX_HeII] - k[2041] - k[2042] - k[2204];
    data[507] = 0.0 - k[1091]*y[IDX_HS2I];
    data[508] = 0.0 - k[1247]*y[IDX_HS2I] - k[1248]*y[IDX_HS2I];
    data[509] = 0.0 - k[1052]*y[IDX_HS2I];
    data[510] = 0.0 - k[127]*y[IDX_HS2I];
    data[511] = 0.0 - k[1146]*y[IDX_HS2I];
    data[512] = 0.0 + k[517]*y[IDX_H2S2II];
    data[513] = 0.0 - k[576]*y[IDX_EM] - k[1509]*y[IDX_OI] - k[2276];
    data[514] = 0.0 + k[23]*y[IDX_CII] + k[134]*y[IDX_HII];
    data[515] = 0.0 + k[1341]*y[IDX_NI];
    data[516] = 0.0 + k[1335]*y[IDX_NI];
    data[517] = 0.0 + k[1336]*y[IDX_NI];
    data[518] = 0.0 + k[1373]*y[IDX_SI];
    data[519] = 0.0 + k[1461]*y[IDX_SII];
    data[520] = 0.0 + k[1461]*y[IDX_NHI];
    data[521] = 0.0 + k[23]*y[IDX_NSI];
    data[522] = 0.0 + k[1373]*y[IDX_NHII];
    data[523] = 0.0 + k[1335]*y[IDX_H2SII] + k[1336]*y[IDX_HSII] + k[1341]*y[IDX_SOII];
    data[524] = 0.0 + k[134]*y[IDX_NSI];
    data[525] = 0.0 - k[1509]*y[IDX_NSII];
    data[526] = 0.0 - k[576]*y[IDX_NSII];
    data[527] = 0.0 + k[2403] + k[2404] + k[2405] + k[2406];
    data[528] = 0.0 + k[474]*y[IDX_EM];
    data[529] = 0.0 - k[611]*y[IDX_CII] - k[1082]*y[IDX_H3OII] - k[1137]*y[IDX_HCOII] -
        k[1197]*y[IDX_HeII] - k[2149];
    data[530] = 0.0 + k[1583]*y[IDX_CI];
    data[531] = 0.0 + k[1675]*y[IDX_CHI];
    data[532] = 0.0 + k[1675]*y[IDX_C2H4I];
    data[533] = 0.0 - k[1082]*y[IDX_CH3CCHI];
    data[534] = 0.0 - k[611]*y[IDX_CH3CCHI];
    data[535] = 0.0 - k[1197]*y[IDX_CH3CCHI];
    data[536] = 0.0 + k[1583]*y[IDX_C2H5I];
    data[537] = 0.0 - k[1137]*y[IDX_CH3CCHI];
    data[538] = 0.0 + k[474]*y[IDX_C3H5II];
    data[539] = 0.0 - k[526]*y[IDX_EM] - k[527]*y[IDX_EM] - k[2282];
    data[540] = 0.0 + k[1042]*y[IDX_H3II] + k[1142]*y[IDX_HCOII];
    data[541] = 0.0 + k[1561]*y[IDX_SOII];
    data[542] = 0.0 + k[1561]*y[IDX_C2H4I];
    data[543] = 0.0 + k[785]*y[IDX_CH3II];
    data[544] = 0.0 + k[814]*y[IDX_CH4I];
    data[545] = 0.0 + k[744]*y[IDX_H2SI];
    data[546] = 0.0 + k[744]*y[IDX_CH2II] + k[778]*y[IDX_CH3II];
    data[547] = 0.0 + k[778]*y[IDX_H2SI] + k[785]*y[IDX_OCSI];
    data[548] = 0.0 + k[821]*y[IDX_CH4I];
    data[549] = 0.0 + k[814]*y[IDX_HSII] + k[821]*y[IDX_SII];
    data[550] = 0.0 + k[1042]*y[IDX_H2CSI];
    data[551] = 0.0 + k[1142]*y[IDX_H2CSI];
    data[552] = 0.0 - k[526]*y[IDX_H3CSII] - k[527]*y[IDX_H3CSII];
    data[553] = 0.0 + k[2475] + k[2476] + k[2477] + k[2478];
    data[554] = 0.0 + k[590]*y[IDX_EM];
    data[555] = 0.0 + k[450] + k[1919]*y[IDX_OI] + k[2080];
    data[556] = 0.0 - k[27]*y[IDX_CII] - k[144]*y[IDX_HII] - k[449] -
        k[1274]*y[IDX_HeII] - k[1918]*y[IDX_OI] - k[2078] - k[2182];
    data[557] = 0.0 + k[1575]*y[IDX_C2H2I];
    data[558] = 0.0 + k[1575]*y[IDX_SiI];
    data[559] = 0.0 - k[27]*y[IDX_SiC2I];
    data[560] = 0.0 - k[1274]*y[IDX_SiC2I];
    data[561] = 0.0 - k[144]*y[IDX_SiC2I];
    data[562] = 0.0 - k[1918]*y[IDX_SiC2I] + k[1919]*y[IDX_SiC3I];
    data[563] = 0.0 + k[590]*y[IDX_SiC3II];
    data[564] = 0.0 - k[325]*y[IDX_O2I] - k[585]*y[IDX_EM] - k[586]*y[IDX_EM] -
        k[879]*y[IDX_COI] - k[962]*y[IDX_H2I] - k[1107]*y[IDX_HI] - k[2284];
    data[565] = 0.0 + k[141]*y[IDX_HII] + k[218]*y[IDX_HeII] + k[320]*y[IDX_OII];
    data[566] = 0.0 + k[320]*y[IDX_SO2I];
    data[567] = 0.0 - k[325]*y[IDX_SO2II];
    data[568] = 0.0 + k[218]*y[IDX_SO2I];
    data[569] = 0.0 + k[141]*y[IDX_SO2I];
    data[570] = 0.0 - k[879]*y[IDX_SO2II];
    data[571] = 0.0 - k[962]*y[IDX_SO2II];
    data[572] = 0.0 - k[585]*y[IDX_SO2II] - k[586]*y[IDX_SO2II];
    data[573] = 0.0 - k[1107]*y[IDX_SO2II];
    data[574] = 0.0 + k[2443] + k[2444] + k[2445] + k[2446];
    data[575] = 0.0 + k[527]*y[IDX_EM];
    data[576] = 0.0 - k[120]*y[IDX_HII] - k[400] - k[619]*y[IDX_CII] -
        k[1042]*y[IDX_H3II] - k[1142]*y[IDX_HCOII] - k[1219]*y[IDX_HeII] -
        k[1220]*y[IDX_HeII] - k[1221]*y[IDX_HeII] - k[2015] - k[2263];
    data[577] = 0.0 + k[2137]*y[IDX_EM];
    data[578] = 0.0 + k[1669]*y[IDX_SI];
    data[579] = 0.0 - k[619]*y[IDX_H2CSI];
    data[580] = 0.0 + k[1669]*y[IDX_CH3I];
    data[581] = 0.0 - k[1219]*y[IDX_H2CSI] - k[1220]*y[IDX_H2CSI] - k[1221]*y[IDX_H2CSI];
    data[582] = 0.0 - k[1042]*y[IDX_H2CSI];
    data[583] = 0.0 - k[120]*y[IDX_H2CSI];
    data[584] = 0.0 - k[1142]*y[IDX_H2CSI];
    data[585] = 0.0 + k[527]*y[IDX_H3CSII] + k[2137]*y[IDX_H2CSII];
    data[586] = 0.0 + k[120]*y[IDX_HII];
    data[587] = 0.0 - k[506]*y[IDX_EM] - k[507]*y[IDX_EM] - k[2137]*y[IDX_EM] - k[2279];
    data[588] = 0.0 + k[1048]*y[IDX_H3II];
    data[589] = 0.0 + k[1559]*y[IDX_SOII];
    data[590] = 0.0 + k[1556]*y[IDX_C2H2I] + k[1559]*y[IDX_C2H4I];
    data[591] = 0.0 + k[751]*y[IDX_CH2II];
    data[592] = 0.0 + k[780]*y[IDX_CH3II];
    data[593] = 0.0 + k[1556]*y[IDX_SOII];
    data[594] = 0.0 + k[751]*y[IDX_OCSI];
    data[595] = 0.0 + k[780]*y[IDX_HSI];
    data[596] = 0.0 + k[790]*y[IDX_SII];
    data[597] = 0.0 + k[790]*y[IDX_CH3I];
    data[598] = 0.0 + k[1048]*y[IDX_HCSI];
    data[599] = 0.0 + k[120]*y[IDX_H2CSI];
    data[600] = 0.0 - k[506]*y[IDX_H2CSII] - k[507]*y[IDX_H2CSII] -
        k[2137]*y[IDX_H2CSII];
    data[601] = 0.0 + k[2467] + k[2468] + k[2469] + k[2470];
    data[602] = 0.0 + k[1797]*y[IDX_NI];
    data[603] = 0.0 + k[994]*y[IDX_H2OI];
    data[604] = 0.0 - k[407] - k[623]*y[IDX_CII] - k[624]*y[IDX_CII] - k[625]*y[IDX_CII]
        - k[1229]*y[IDX_HeII] - k[1230]*y[IDX_HeII] - k[2027] - k[2145];
    data[605] = 0.0 + k[1580]*y[IDX_C2HI];
    data[606] = 0.0 + k[1701]*y[IDX_CNI];
    data[607] = 0.0 + k[1579]*y[IDX_C2HI];
    data[608] = 0.0 + k[1578]*y[IDX_HCNI] + k[1579]*y[IDX_HNCI] + k[1580]*y[IDX_NCCNI] +
        k[2100]*y[IDX_CNI];
    data[609] = 0.0 + k[1701]*y[IDX_C2H2I] + k[2100]*y[IDX_C2HI];
    data[610] = 0.0 + k[1578]*y[IDX_C2HI];
    data[611] = 0.0 - k[623]*y[IDX_HC3NI] - k[624]*y[IDX_HC3NI] - k[625]*y[IDX_HC3NI];
    data[612] = 0.0 + k[1797]*y[IDX_C3H2I];
    data[613] = 0.0 - k[1229]*y[IDX_HC3NI] - k[1230]*y[IDX_HC3NI];
    data[614] = 0.0 + k[994]*y[IDX_C4NII];
    data[615] = 0.0 - k[561]*y[IDX_EM] - k[562]*y[IDX_EM] - k[1009]*y[IDX_H2OI] -
        k[1020]*y[IDX_H2SI] - k[1127]*y[IDX_HCNI] - k[1439]*y[IDX_NH3I] -
        k[2191];
    data[616] = 0.0 + k[1077]*y[IDX_H3II] + k[1157]*y[IDX_HCOII];
    data[617] = 0.0 + k[1568]*y[IDX_SI];
    data[618] = 0.0 - k[1020]*y[IDX_HSiSII];
    data[619] = 0.0 - k[1127]*y[IDX_HSiSII];
    data[620] = 0.0 - k[1439]*y[IDX_HSiSII];
    data[621] = 0.0 + k[1568]*y[IDX_SiH2II];
    data[622] = 0.0 + k[1077]*y[IDX_SiSI];
    data[623] = 0.0 + k[1157]*y[IDX_SiSI];
    data[624] = 0.0 - k[1009]*y[IDX_HSiSII];
    data[625] = 0.0 - k[561]*y[IDX_HSiSII] - k[562]*y[IDX_HSiSII];
    data[626] = 0.0 - k[605]*y[IDX_EM] - k[1109]*y[IDX_HI] - k[1487]*y[IDX_O2I] -
        k[1488]*y[IDX_O2I] - k[2190];
    data[627] = 0.0 + k[32]*y[IDX_CII] + k[152]*y[IDX_HII] + k[344]*y[IDX_SII];
    data[628] = 0.0 + k[1569]*y[IDX_SII];
    data[629] = 0.0 + k[1565]*y[IDX_SiII];
    data[630] = 0.0 + k[1565]*y[IDX_OCSI];
    data[631] = 0.0 + k[344]*y[IDX_SiSI] + k[1569]*y[IDX_SiHI];
    data[632] = 0.0 - k[1487]*y[IDX_SiSII] - k[1488]*y[IDX_SiSII];
    data[633] = 0.0 + k[32]*y[IDX_SiSI];
    data[634] = 0.0 + k[152]*y[IDX_SiSI];
    data[635] = 0.0 - k[605]*y[IDX_SiSII];
    data[636] = 0.0 - k[1109]*y[IDX_SiSII];
    data[637] = 0.0 + k[1225]*y[IDX_HeII];
    data[638] = 0.0 + k[127]*y[IDX_HII] + k[2041];
    data[639] = 0.0 - k[555]*y[IDX_EM] - k[556]*y[IDX_EM] - k[1019]*y[IDX_H2SI] -
        k[2205];
    data[640] = 0.0 + k[1067]*y[IDX_H3II] + k[1092]*y[IDX_H3OII] + k[1150]*y[IDX_HCOII];
    data[641] = 0.0 + k[1176]*y[IDX_H2SI];
    data[642] = 0.0 - k[1019]*y[IDX_HS2II] + k[1176]*y[IDX_HSII] + k[1550]*y[IDX_SII];
    data[643] = 0.0 + k[1550]*y[IDX_H2SI];
    data[644] = 0.0 + k[1092]*y[IDX_S2I];
    data[645] = 0.0 + k[1225]*y[IDX_H2S2I];
    data[646] = 0.0 + k[1067]*y[IDX_S2I];
    data[647] = 0.0 + k[127]*y[IDX_HS2I];
    data[648] = 0.0 + k[1150]*y[IDX_S2I];
    data[649] = 0.0 - k[555]*y[IDX_HS2II] - k[556]*y[IDX_HS2II];
    data[650] = 0.0 + k[2387] + k[2388] + k[2389] + k[2390];
    data[651] = 0.0 - k[437] - k[1663]*y[IDX_CH3I] - k[1691]*y[IDX_CHI] -
        k[1692]*y[IDX_CHI] - k[1719]*y[IDX_COI] - k[1769]*y[IDX_HI] -
        k[1770]*y[IDX_HI] - k[1771]*y[IDX_HI] - k[1786]*y[IDX_HCOI] -
        k[1826]*y[IDX_NI] - k[1909]*y[IDX_OI] - k[1946]*y[IDX_OHI] - k[2063] -
        k[2064] - k[2294];
    data[652] = 0.0 + k[1577]*y[IDX_O2I];
    data[653] = 0.0 + k[967]*y[IDX_O2I];
    data[654] = 0.0 + k[1662]*y[IDX_O2I] - k[1663]*y[IDX_O2HI];
    data[655] = 0.0 + k[1671]*y[IDX_O2I];
    data[656] = 0.0 + k[1785]*y[IDX_O2I] - k[1786]*y[IDX_O2HI];
    data[657] = 0.0 - k[1691]*y[IDX_O2HI] - k[1692]*y[IDX_O2HI];
    data[658] = 0.0 + k[967]*y[IDX_H2COII] + k[1577]*y[IDX_C2H3I] + k[1662]*y[IDX_CH3I]
        + k[1671]*y[IDX_CH4I] + k[1731]*y[IDX_H2I] + k[1785]*y[IDX_HCOI];
    data[659] = 0.0 - k[1946]*y[IDX_O2HI];
    data[660] = 0.0 - k[1826]*y[IDX_O2HI];
    data[661] = 0.0 - k[1909]*y[IDX_O2HI];
    data[662] = 0.0 - k[1719]*y[IDX_O2HI];
    data[663] = 0.0 + k[1731]*y[IDX_O2I];
    data[664] = 0.0 - k[1769]*y[IDX_O2HI] - k[1770]*y[IDX_O2HI] - k[1771]*y[IDX_O2HI];
    data[665] = 0.0 - k[484]*y[IDX_EM] - k[485]*y[IDX_EM] - k[2156];
    data[666] = 0.0 + k[1120]*y[IDX_HCNI];
    data[667] = 0.0 + k[666]*y[IDX_C2H2II] + k[791]*y[IDX_HCO2II] + k[1030]*y[IDX_H3II]
        + k[1083]*y[IDX_H3OII] + k[1130]*y[IDX_HCNHII] + k[1131]*y[IDX_HCNHII] +
        k[1138]*y[IDX_HCOII] + k[1321]*y[IDX_N2HII];
    data[668] = 0.0 + k[791]*y[IDX_CH3CNI];
    data[669] = 0.0 + k[1321]*y[IDX_CH3CNI];
    data[670] = 0.0 + k[1130]*y[IDX_CH3CNI] + k[1131]*y[IDX_CH3CNI];
    data[671] = 0.0 + k[2108]*y[IDX_HCNI];
    data[672] = 0.0 + k[666]*y[IDX_CH3CNI];
    data[673] = 0.0 + k[1120]*y[IDX_CH3OH2II] + k[2108]*y[IDX_CH3II];
    data[674] = 0.0 + k[1083]*y[IDX_CH3CNI];
    data[675] = 0.0 + k[1030]*y[IDX_CH3CNI];
    data[676] = 0.0 + k[1138]*y[IDX_CH3CNI];
    data[677] = 0.0 - k[484]*y[IDX_CH3CNHII] - k[485]*y[IDX_CH3CNHII];
    data[678] = 0.0 + k[2447] + k[2448] + k[2449] + k[2450];
    data[679] = 0.0 - k[431] - k[903]*y[IDX_HII] - k[1059]*y[IDX_H3II] -
        k[1478]*y[IDX_OII] - k[1628]*y[IDX_CH2I] - k[1658]*y[IDX_CH3I] -
        k[1709]*y[IDX_CNI] - k[1717]*y[IDX_COI] - k[1763]*y[IDX_HI] -
        k[1820]*y[IDX_NI] - k[1821]*y[IDX_NI] - k[1822]*y[IDX_NI] -
        k[1846]*y[IDX_NHI] - k[1905]*y[IDX_OI] - k[2056] - k[2270];
    data[680] = 0.0 + k[1864]*y[IDX_O2I];
    data[681] = 0.0 + k[1896]*y[IDX_OI];
    data[682] = 0.0 - k[1478]*y[IDX_NO2I];
    data[683] = 0.0 - k[1628]*y[IDX_NO2I];
    data[684] = 0.0 - k[1658]*y[IDX_NO2I];
    data[685] = 0.0 - k[1846]*y[IDX_NO2I];
    data[686] = 0.0 - k[1709]*y[IDX_NO2I];
    data[687] = 0.0 + k[1859]*y[IDX_O2I] + k[1945]*y[IDX_OHI];
    data[688] = 0.0 + k[1859]*y[IDX_NOI] + k[1864]*y[IDX_OCNI];
    data[689] = 0.0 + k[1945]*y[IDX_NOI];
    data[690] = 0.0 - k[1820]*y[IDX_NO2I] - k[1821]*y[IDX_NO2I] - k[1822]*y[IDX_NO2I];
    data[691] = 0.0 - k[1059]*y[IDX_NO2I];
    data[692] = 0.0 - k[903]*y[IDX_NO2I];
    data[693] = 0.0 + k[1896]*y[IDX_HNOI] - k[1905]*y[IDX_NO2I];
    data[694] = 0.0 - k[1717]*y[IDX_NO2I];
    data[695] = 0.0 - k[1763]*y[IDX_NO2I];
    data[696] = 0.0 + k[1248]*y[IDX_HeII];
    data[697] = 0.0 - k[304]*y[IDX_NOI] - k[583]*y[IDX_EM] - k[793]*y[IDX_CH3OHI] -
        k[2281];
    data[698] = 0.0 + k[139]*y[IDX_HII] + k[2071];
    data[699] = 0.0 - k[793]*y[IDX_S2II];
    data[700] = 0.0 + k[1021]*y[IDX_H2SI] + k[1562]*y[IDX_OCSI];
    data[701] = 0.0 + k[1552]*y[IDX_SII] + k[1562]*y[IDX_SOII];
    data[702] = 0.0 + k[1021]*y[IDX_SOII] + k[1551]*y[IDX_SII];
    data[703] = 0.0 + k[1551]*y[IDX_H2SI] + k[1552]*y[IDX_OCSI];
    data[704] = 0.0 - k[304]*y[IDX_S2II];
    data[705] = 0.0 + k[1248]*y[IDX_HS2I];
    data[706] = 0.0 + k[139]*y[IDX_S2I];
    data[707] = 0.0 - k[583]*y[IDX_S2II];
    data[708] = 0.0 + k[2359] + k[2360] + k[2361] + k[2362];
    data[709] = 0.0 + k[465]*y[IDX_EM] + k[466]*y[IDX_EM];
    data[710] = 0.0 - k[371] - k[1583]*y[IDX_CI] - k[1793]*y[IDX_NI] - k[1794]*y[IDX_NI]
        - k[1874]*y[IDX_OI] - k[1931]*y[IDX_OHI] - k[1968] - k[2301];
    data[711] = 0.0 + k[372] + k[1563]*y[IDX_SiII] + k[1969];
    data[712] = 0.0 + k[672]*y[IDX_C2H2II];
    data[713] = 0.0 + k[1563]*y[IDX_C2H5OHI];
    data[714] = 0.0 + k[672]*y[IDX_SiH4I];
    data[715] = 0.0 + k[1648]*y[IDX_CH3I] + k[1648]*y[IDX_CH3I];
    data[716] = 0.0 - k[1931]*y[IDX_C2H5I];
    data[717] = 0.0 - k[1793]*y[IDX_C2H5I] - k[1794]*y[IDX_C2H5I];
    data[718] = 0.0 - k[1874]*y[IDX_C2H5I];
    data[719] = 0.0 - k[1583]*y[IDX_C2H5I];
    data[720] = 0.0 + k[465]*y[IDX_C2H5OH2II] + k[466]*y[IDX_C2H5OH2II];
    data[721] = 0.0 + k[2439] + k[2440] + k[2441] + k[2442];
    data[722] = 0.0 + k[467]*y[IDX_EM] + k[1420]*y[IDX_NH3I];
    data[723] = 0.0 - k[372] - k[606]*y[IDX_CII] - k[1023]*y[IDX_H3II] -
        k[1024]*y[IDX_H3II] - k[1081]*y[IDX_H3OII] - k[1136]*y[IDX_HCOII] -
        k[1164]*y[IDX_HCSII] - k[1563]*y[IDX_SiII] - k[1969] - k[2150];
    data[724] = 0.0 - k[1164]*y[IDX_C2H5OHI];
    data[725] = 0.0 - k[1563]*y[IDX_C2H5OHI];
    data[726] = 0.0 + k[1420]*y[IDX_C2H5OH2II];
    data[727] = 0.0 - k[1081]*y[IDX_C2H5OHI];
    data[728] = 0.0 - k[606]*y[IDX_C2H5OHI];
    data[729] = 0.0 - k[1023]*y[IDX_C2H5OHI] - k[1024]*y[IDX_C2H5OHI];
    data[730] = 0.0 - k[1136]*y[IDX_C2H5OHI];
    data[731] = 0.0 + k[467]*y[IDX_C2H5OH2II];
    data[732] = 0.0 - k[587]*y[IDX_EM] - k[1342]*y[IDX_NI] - k[1512]*y[IDX_OI] - k[2183];
    data[733] = 0.0 + k[29]*y[IDX_CII] + k[146]*y[IDX_HII] + k[343]*y[IDX_SII];
    data[734] = 0.0 + k[643]*y[IDX_CII];
    data[735] = 0.0 + k[646]*y[IDX_CII];
    data[736] = 0.0 + k[644]*y[IDX_CII];
    data[737] = 0.0 + k[659]*y[IDX_C2I];
    data[738] = 0.0 + k[703]*y[IDX_CI];
    data[739] = 0.0 + k[862]*y[IDX_CHI];
    data[740] = 0.0 + k[659]*y[IDX_SiOII];
    data[741] = 0.0 + k[343]*y[IDX_SiCI];
    data[742] = 0.0 + k[862]*y[IDX_SiII];
    data[743] = 0.0 + k[29]*y[IDX_SiCI] + k[643]*y[IDX_SiH2I] + k[644]*y[IDX_SiHI] +
        k[646]*y[IDX_SiSI];
    data[744] = 0.0 - k[1342]*y[IDX_SiCII];
    data[745] = 0.0 + k[146]*y[IDX_SiCI];
    data[746] = 0.0 - k[1512]*y[IDX_SiCII];
    data[747] = 0.0 + k[703]*y[IDX_SiHII];
    data[748] = 0.0 - k[587]*y[IDX_SiCII];
    data[749] = 0.0 + k[1593]*y[IDX_CI];
    data[750] = 0.0 + k[470]*y[IDX_EM];
    data[751] = 0.0 + k[475]*y[IDX_EM];
    data[752] = 0.0 + k[472]*y[IDX_EM];
    data[753] = 0.0 + k[1798]*y[IDX_NI] + k[1877]*y[IDX_OI];
    data[754] = 0.0 - k[113]*y[IDX_HII] - k[375] - k[376] - k[1026]*y[IDX_H3II] -
        k[1189]*y[IDX_HeII] - k[1584]*y[IDX_CI] - k[1796]*y[IDX_NI] -
        k[1876]*y[IDX_OI] - k[1972] - k[1973] - k[2159];
    data[755] = 0.0 + k[1598]*y[IDX_CI] + k[1818]*y[IDX_NI];
    data[756] = 0.0 + k[1795]*y[IDX_NI];
    data[757] = 0.0 + k[1795]*y[IDX_C2HI] - k[1796]*y[IDX_C2NI] + k[1798]*y[IDX_C3NI] +
        k[1818]*y[IDX_NCCNI];
    data[758] = 0.0 - k[1189]*y[IDX_C2NI];
    data[759] = 0.0 - k[1026]*y[IDX_C2NI];
    data[760] = 0.0 - k[113]*y[IDX_C2NI];
    data[761] = 0.0 - k[1876]*y[IDX_C2NI] + k[1877]*y[IDX_C3NI];
    data[762] = 0.0 - k[1584]*y[IDX_C2NI] + k[1593]*y[IDX_H2CNI] + k[1598]*y[IDX_NCCNI];
    data[763] = 0.0 + k[470]*y[IDX_C2N2II] + k[472]*y[IDX_C2NHII] + k[475]*y[IDX_C4NII];
    data[764] = 0.0 + k[2471] + k[2472] + k[2473] + k[2474];
    data[765] = 0.0 + k[675]*y[IDX_C2H2I] + k[1118]*y[IDX_HCNI];
    data[766] = 0.0 - k[20]*y[IDX_CII] - k[423] - k[1251]*y[IDX_HeII] -
        k[1304]*y[IDX_NII] - k[1305]*y[IDX_NII] - k[1580]*y[IDX_C2HI] -
        k[1598]*y[IDX_CI] - k[1759]*y[IDX_HI] - k[1818]*y[IDX_NI] -
        k[1943]*y[IDX_OHI] - k[2046] - k[2047] - k[2164];
    data[767] = 0.0 + k[675]*y[IDX_C2N2II];
    data[768] = 0.0 + k[1707]*y[IDX_CNI];
    data[769] = 0.0 - k[1304]*y[IDX_NCCNI] - k[1305]*y[IDX_NCCNI];
    data[770] = 0.0 - k[1580]*y[IDX_NCCNI];
    data[771] = 0.0 + k[1705]*y[IDX_HCNI] + k[1707]*y[IDX_HNCI];
    data[772] = 0.0 + k[1118]*y[IDX_C2N2II] + k[1705]*y[IDX_CNI];
    data[773] = 0.0 - k[20]*y[IDX_NCCNI];
    data[774] = 0.0 - k[1943]*y[IDX_NCCNI];
    data[775] = 0.0 - k[1818]*y[IDX_NCCNI];
    data[776] = 0.0 - k[1251]*y[IDX_NCCNI];
    data[777] = 0.0 - k[1598]*y[IDX_NCCNI];
    data[778] = 0.0 - k[1759]*y[IDX_NCCNI];
    data[779] = 0.0 + k[2407] + k[2408] + k[2409] + k[2410];
    data[780] = 0.0 + k[591]*y[IDX_EM];
    data[781] = 0.0 + k[2079];
    data[782] = 0.0 + k[589]*y[IDX_EM];
    data[783] = 0.0 + k[449] + k[1918]*y[IDX_OI];
    data[784] = 0.0 - k[29]*y[IDX_CII] - k[146]*y[IDX_HII] - k[343]*y[IDX_SII] - k[451]
        - k[642]*y[IDX_CII] - k[1276]*y[IDX_HeII] - k[1277]*y[IDX_HeII] -
        k[1832]*y[IDX_NI] - k[1920]*y[IDX_OI] - k[1921]*y[IDX_OI] - k[2081] -
        k[2181];
    data[785] = 0.0 + k[1617]*y[IDX_CI];
    data[786] = 0.0 - k[343]*y[IDX_SiCI];
    data[787] = 0.0 - k[29]*y[IDX_SiCI] - k[642]*y[IDX_SiCI];
    data[788] = 0.0 - k[1832]*y[IDX_SiCI];
    data[789] = 0.0 - k[1276]*y[IDX_SiCI] - k[1277]*y[IDX_SiCI];
    data[790] = 0.0 - k[146]*y[IDX_SiCI];
    data[791] = 0.0 + k[1918]*y[IDX_SiC2I] - k[1920]*y[IDX_SiCI] - k[1921]*y[IDX_SiCI];
    data[792] = 0.0 + k[1617]*y[IDX_SiHI];
    data[793] = 0.0 + k[589]*y[IDX_SiC2II] + k[591]*y[IDX_SiC3II];
    data[794] = 0.0 + k[507]*y[IDX_EM];
    data[795] = 0.0 - k[412] - k[899]*y[IDX_HII] - k[1048]*y[IDX_H3II] -
        k[1239]*y[IDX_HeII] - k[1240]*y[IDX_HeII] - k[1753]*y[IDX_HI] -
        k[1814]*y[IDX_NI] - k[1894]*y[IDX_OI] - k[1895]*y[IDX_OI] - k[2033] -
        k[2258];
    data[796] = 0.0 + k[1560]*y[IDX_SOII];
    data[797] = 0.0 + k[1557]*y[IDX_C2H2I] + k[1560]*y[IDX_C2H4I];
    data[798] = 0.0 + k[1557]*y[IDX_SOII];
    data[799] = 0.0 + k[1645]*y[IDX_SI];
    data[800] = 0.0 + k[1645]*y[IDX_CH2I];
    data[801] = 0.0 - k[1814]*y[IDX_HCSI];
    data[802] = 0.0 - k[1239]*y[IDX_HCSI] - k[1240]*y[IDX_HCSI];
    data[803] = 0.0 - k[1048]*y[IDX_HCSI];
    data[804] = 0.0 - k[899]*y[IDX_HCSI];
    data[805] = 0.0 - k[1894]*y[IDX_HCSI] - k[1895]*y[IDX_HCSI];
    data[806] = 0.0 + k[507]*y[IDX_H2CSII];
    data[807] = 0.0 - k[1753]*y[IDX_HCSI];
    data[808] = 0.0 + k[598]*y[IDX_EM];
    data[809] = 0.0 - k[30]*y[IDX_CII] - k[147]*y[IDX_HII] - k[452] - k[643]*y[IDX_CII]
        - k[905]*y[IDX_HII] - k[1072]*y[IDX_H3II] - k[1094]*y[IDX_H3OII] -
        k[1153]*y[IDX_HCOII] - k[1278]*y[IDX_HeII] - k[1279]*y[IDX_HeII] -
        k[1922]*y[IDX_OI] - k[1923]*y[IDX_OI] - k[2083] - k[2084] - k[2175];
    data[810] = 0.0 + k[596]*y[IDX_EM];
    data[811] = 0.0 + k[453] + k[2085];
    data[812] = 0.0 + k[454] + k[2088];
    data[813] = 0.0 - k[1094]*y[IDX_SiH2I];
    data[814] = 0.0 - k[30]*y[IDX_SiH2I] - k[643]*y[IDX_SiH2I];
    data[815] = 0.0 - k[1278]*y[IDX_SiH2I] - k[1279]*y[IDX_SiH2I];
    data[816] = 0.0 - k[1072]*y[IDX_SiH2I];
    data[817] = 0.0 - k[147]*y[IDX_SiH2I] - k[905]*y[IDX_SiH2I];
    data[818] = 0.0 - k[1922]*y[IDX_SiH2I] - k[1923]*y[IDX_SiH2I];
    data[819] = 0.0 - k[1153]*y[IDX_SiH2I];
    data[820] = 0.0 + k[596]*y[IDX_SiH3II] + k[598]*y[IDX_SiH4II];
    data[821] = 0.0 + k[1072]*y[IDX_H3II] + k[1094]*y[IDX_H3OII] + k[1153]*y[IDX_HCOII];
    data[822] = 0.0 - k[596]*y[IDX_EM] - k[597]*y[IDX_EM] - k[1515]*y[IDX_OI] -
        k[2120]*y[IDX_H2I] - k[2178];
    data[823] = 0.0 + k[31]*y[IDX_CII] + k[148]*y[IDX_HII] + k[2086];
    data[824] = 0.0 + k[2119]*y[IDX_H2I];
    data[825] = 0.0 + k[674]*y[IDX_C2H2II] + k[789]*y[IDX_CH3II] + k[836]*y[IDX_CH5II] +
        k[907]*y[IDX_HII];
    data[826] = 0.0 + k[836]*y[IDX_SiH4I];
    data[827] = 0.0 + k[789]*y[IDX_SiH4I];
    data[828] = 0.0 + k[674]*y[IDX_SiH4I];
    data[829] = 0.0 + k[1094]*y[IDX_SiH2I];
    data[830] = 0.0 + k[31]*y[IDX_SiH3I];
    data[831] = 0.0 + k[1072]*y[IDX_SiH2I];
    data[832] = 0.0 + k[148]*y[IDX_SiH3I] + k[907]*y[IDX_SiH4I];
    data[833] = 0.0 - k[1515]*y[IDX_SiH3II];
    data[834] = 0.0 + k[1153]*y[IDX_SiH2I];
    data[835] = 0.0 + k[2119]*y[IDX_SiHII] - k[2120]*y[IDX_SiH3II];
    data[836] = 0.0 - k[596]*y[IDX_SiH3II] - k[597]*y[IDX_SiH3II];
    data[837] = 0.0 + k[2483] + k[2484] + k[2485] + k[2486];
    data[838] = 0.0 + k[562]*y[IDX_EM] + k[1020]*y[IDX_H2SI] + k[1127]*y[IDX_HCNI] +
        k[1439]*y[IDX_NH3I];
    data[839] = 0.0 - k[32]*y[IDX_CII] - k[152]*y[IDX_HII] - k[344]*y[IDX_SII] - k[457]
        - k[646]*y[IDX_CII] - k[1077]*y[IDX_H3II] - k[1157]*y[IDX_HCOII] -
        k[1287]*y[IDX_HeII] - k[1288]*y[IDX_HeII] - k[2095] - k[2189];
    data[840] = 0.0 + k[1020]*y[IDX_HSiSII];
    data[841] = 0.0 - k[344]*y[IDX_SiSI];
    data[842] = 0.0 + k[1127]*y[IDX_HSiSII];
    data[843] = 0.0 + k[1439]*y[IDX_HSiSII];
    data[844] = 0.0 - k[32]*y[IDX_SiSI] - k[646]*y[IDX_SiSI];
    data[845] = 0.0 - k[1287]*y[IDX_SiSI] - k[1288]*y[IDX_SiSI];
    data[846] = 0.0 - k[1077]*y[IDX_SiSI];
    data[847] = 0.0 - k[152]*y[IDX_SiSI];
    data[848] = 0.0 - k[1157]*y[IDX_SiSI];
    data[849] = 0.0 + k[562]*y[IDX_HSiSII];
    data[850] = 0.0 - k[486]*y[IDX_EM] - k[487]*y[IDX_EM] - k[488]*y[IDX_EM] -
        k[489]*y[IDX_EM] - k[490]*y[IDX_EM] - k[792]*y[IDX_NH3I] -
        k[969]*y[IDX_H2COI] - k[1120]*y[IDX_HCNI] - k[2152];
    data[851] = 0.0 + k[708]*y[IDX_CHII] + k[794]*y[IDX_CH4II] + k[965]*y[IDX_H2COII] +
        k[1032]*y[IDX_H3II] + k[1078]*y[IDX_H3COII] + k[1084]*y[IDX_H3OII] +
        k[1139]*y[IDX_HCOII];
    data[852] = 0.0 + k[794]*y[IDX_CH3OHI];
    data[853] = 0.0 + k[1078]*y[IDX_CH3OHI];
    data[854] = 0.0 + k[965]*y[IDX_CH3OHI];
    data[855] = 0.0 + k[2107]*y[IDX_H2OI];
    data[856] = 0.0 + k[708]*y[IDX_CH3OHI];
    data[857] = 0.0 - k[969]*y[IDX_CH3OH2II];
    data[858] = 0.0 - k[1120]*y[IDX_CH3OH2II];
    data[859] = 0.0 - k[792]*y[IDX_CH3OH2II];
    data[860] = 0.0 + k[1084]*y[IDX_CH3OHI];
    data[861] = 0.0 + k[1032]*y[IDX_CH3OHI];
    data[862] = 0.0 + k[1139]*y[IDX_CH3OHI];
    data[863] = 0.0 + k[2107]*y[IDX_CH3II];
    data[864] = 0.0 - k[486]*y[IDX_CH3OH2II] - k[487]*y[IDX_CH3OH2II] -
        k[488]*y[IDX_CH3OH2II] - k[489]*y[IDX_CH3OH2II] - k[490]*y[IDX_CH3OH2II];
    data[865] = 0.0 + k[1885]*y[IDX_OI];
    data[866] = 0.0 + k[1709]*y[IDX_CNI];
    data[867] = 0.0 + k[1943]*y[IDX_OHI];
    data[868] = 0.0 - k[439] - k[635]*y[IDX_CII] - k[1262]*y[IDX_HeII] -
        k[1263]*y[IDX_HeII] - k[1609]*y[IDX_CI] - k[1772]*y[IDX_HI] -
        k[1773]*y[IDX_HI] - k[1774]*y[IDX_HI] - k[1860]*y[IDX_NOI] -
        k[1863]*y[IDX_O2I] - k[1864]*y[IDX_O2I] - k[1910]*y[IDX_OI] -
        k[1911]*y[IDX_OI] - k[2065] - k[2158];
    data[869] = 0.0 + k[1709]*y[IDX_NO2I] + k[1711]*y[IDX_NOI] + k[1713]*y[IDX_O2I] +
        k[1933]*y[IDX_OHI];
    data[870] = 0.0 + k[1813]*y[IDX_NI];
    data[871] = 0.0 + k[1891]*y[IDX_OI];
    data[872] = 0.0 + k[1686]*y[IDX_CHI] + k[1711]*y[IDX_CNI] - k[1860]*y[IDX_OCNI];
    data[873] = 0.0 + k[1686]*y[IDX_NOI];
    data[874] = 0.0 + k[1713]*y[IDX_CNI] - k[1863]*y[IDX_OCNI] - k[1864]*y[IDX_OCNI];
    data[875] = 0.0 - k[635]*y[IDX_OCNI];
    data[876] = 0.0 + k[1933]*y[IDX_CNI] + k[1943]*y[IDX_NCCNI];
    data[877] = 0.0 + k[1813]*y[IDX_HCOI];
    data[878] = 0.0 - k[1262]*y[IDX_OCNI] - k[1263]*y[IDX_OCNI];
    data[879] = 0.0 + k[1885]*y[IDX_H2CNI] + k[1891]*y[IDX_HCNI] - k[1910]*y[IDX_OCNI] -
        k[1911]*y[IDX_OCNI];
    data[880] = 0.0 - k[1609]*y[IDX_OCNI];
    data[881] = 0.0 - k[1772]*y[IDX_OCNI] - k[1773]*y[IDX_OCNI] - k[1774]*y[IDX_OCNI];
    data[882] = 0.0 + k[556]*y[IDX_EM] + k[1019]*y[IDX_H2SI];
    data[883] = 0.0 + k[304]*y[IDX_NOI];
    data[884] = 0.0 - k[139]*y[IDX_HII] - k[443] - k[1067]*y[IDX_H3II] -
        k[1092]*y[IDX_H3OII] - k[1150]*y[IDX_HCOII] - k[1269]*y[IDX_HeII] -
        k[1613]*y[IDX_CI] - k[1777]*y[IDX_HI] - k[1829]*y[IDX_NI] -
        k[1915]*y[IDX_OI] - k[2071] - k[2072] - k[2201];
    data[885] = 0.0 + k[1955]*y[IDX_SI];
    data[886] = 0.0 + k[1953]*y[IDX_SI];
    data[887] = 0.0 + k[1019]*y[IDX_HS2II];
    data[888] = 0.0 + k[304]*y[IDX_S2II];
    data[889] = 0.0 - k[1092]*y[IDX_S2I];
    data[890] = 0.0 + k[1953]*y[IDX_HSI] + k[1955]*y[IDX_SOI];
    data[891] = 0.0 - k[1829]*y[IDX_S2I];
    data[892] = 0.0 - k[1269]*y[IDX_S2I];
    data[893] = 0.0 - k[1067]*y[IDX_S2I];
    data[894] = 0.0 - k[139]*y[IDX_S2I];
    data[895] = 0.0 - k[1915]*y[IDX_S2I];
    data[896] = 0.0 - k[1613]*y[IDX_S2I];
    data[897] = 0.0 - k[1150]*y[IDX_S2I];
    data[898] = 0.0 + k[556]*y[IDX_HS2II];
    data[899] = 0.0 - k[1777]*y[IDX_S2I];
    data[900] = 0.0 + k[600]*y[IDX_EM];
    data[901] = 0.0 + k[599]*y[IDX_EM] + k[880]*y[IDX_COI] + k[1015]*y[IDX_H2OI];
    data[902] = 0.0 - k[31]*y[IDX_CII] - k[148]*y[IDX_HII] - k[453] - k[906]*y[IDX_HII]
        - k[1073]*y[IDX_H3II] - k[1280]*y[IDX_HeII] - k[1281]*y[IDX_HeII] -
        k[1924]*y[IDX_OI] - k[2085] - k[2086] - k[2087] - k[2177];
    data[903] = 0.0 + k[1715]*y[IDX_CNI] + k[1925]*y[IDX_OI] + k[2089];
    data[904] = 0.0 + k[1715]*y[IDX_SiH4I];
    data[905] = 0.0 - k[31]*y[IDX_SiH3I];
    data[906] = 0.0 - k[1280]*y[IDX_SiH3I] - k[1281]*y[IDX_SiH3I];
    data[907] = 0.0 - k[1073]*y[IDX_SiH3I];
    data[908] = 0.0 - k[148]*y[IDX_SiH3I] - k[906]*y[IDX_SiH3I];
    data[909] = 0.0 - k[1924]*y[IDX_SiH3I] + k[1925]*y[IDX_SiH4I];
    data[910] = 0.0 + k[1015]*y[IDX_SiH4II];
    data[911] = 0.0 + k[880]*y[IDX_SiH4II];
    data[912] = 0.0 + k[599]*y[IDX_SiH4II] + k[600]*y[IDX_SiH5II];
    data[913] = 0.0 + k[2411] + k[2412] + k[2413] + k[2414];
    data[914] = 0.0 + k[484]*y[IDX_EM];
    data[915] = 0.0 - k[387] - k[665]*y[IDX_C2H2II] - k[666]*y[IDX_C2H2II] -
        k[775]*y[IDX_CH3II] - k[791]*y[IDX_HCO2II] - k[886]*y[IDX_HII] -
        k[1030]*y[IDX_H3II] - k[1083]*y[IDX_H3OII] - k[1130]*y[IDX_HCNHII] -
        k[1131]*y[IDX_HCNHII] - k[1138]*y[IDX_HCOII] - k[1198]*y[IDX_HeII] -
        k[1199]*y[IDX_HeII] - k[1321]*y[IDX_N2HII] - k[1989] - k[2299];
    data[916] = 0.0 - k[791]*y[IDX_CH3CNI];
    data[917] = 0.0 - k[1321]*y[IDX_CH3CNI];
    data[918] = 0.0 - k[1130]*y[IDX_CH3CNI] - k[1131]*y[IDX_CH3CNI];
    data[919] = 0.0 - k[775]*y[IDX_CH3CNI];
    data[920] = 0.0 - k[665]*y[IDX_CH3CNI] - k[666]*y[IDX_CH3CNI];
    data[921] = 0.0 + k[2109]*y[IDX_CNI];
    data[922] = 0.0 + k[2109]*y[IDX_CH3I];
    data[923] = 0.0 - k[1083]*y[IDX_CH3CNI];
    data[924] = 0.0 - k[1198]*y[IDX_CH3CNI] - k[1199]*y[IDX_CH3CNI];
    data[925] = 0.0 - k[1030]*y[IDX_CH3CNI];
    data[926] = 0.0 - k[886]*y[IDX_CH3CNI];
    data[927] = 0.0 - k[1138]*y[IDX_CH3CNI];
    data[928] = 0.0 + k[484]*y[IDX_CH3CNHII];
    data[929] = 0.0 + k[2451] + k[2452] + k[2453] + k[2454];
    data[930] = 0.0 + k[550]*y[IDX_EM];
    data[931] = 0.0 + k[1829]*y[IDX_NI];
    data[932] = 0.0 - k[23]*y[IDX_CII] - k[134]*y[IDX_HII] - k[434] - k[632]*y[IDX_CII]
        - k[1061]*y[IDX_H3II] - k[1148]*y[IDX_HCOII] - k[1259]*y[IDX_HeII] -
        k[1260]*y[IDX_HeII] - k[1606]*y[IDX_CI] - k[1607]*y[IDX_CI] -
        k[1766]*y[IDX_HI] - k[1767]*y[IDX_HI] - k[1824]*y[IDX_NI] -
        k[1907]*y[IDX_OI] - k[1908]*y[IDX_OI] - k[2059] - k[2262];
    data[933] = 0.0 + k[1830]*y[IDX_NI];
    data[934] = 0.0 + k[1816]*y[IDX_NI];
    data[935] = 0.0 + k[1857]*y[IDX_SI];
    data[936] = 0.0 + k[1714]*y[IDX_SI];
    data[937] = 0.0 + k[1861]*y[IDX_SI];
    data[938] = 0.0 - k[23]*y[IDX_NSI] - k[632]*y[IDX_NSI];
    data[939] = 0.0 + k[1714]*y[IDX_CNI] + k[1857]*y[IDX_NHI] + k[1861]*y[IDX_NOI];
    data[940] = 0.0 + k[1816]*y[IDX_HSI] - k[1824]*y[IDX_NSI] + k[1829]*y[IDX_S2I] +
        k[1830]*y[IDX_SOI];
    data[941] = 0.0 - k[1259]*y[IDX_NSI] - k[1260]*y[IDX_NSI];
    data[942] = 0.0 - k[1061]*y[IDX_NSI];
    data[943] = 0.0 - k[134]*y[IDX_NSI];
    data[944] = 0.0 - k[1907]*y[IDX_NSI] - k[1908]*y[IDX_NSI];
    data[945] = 0.0 - k[1606]*y[IDX_NSI] - k[1607]*y[IDX_NSI];
    data[946] = 0.0 - k[1148]*y[IDX_NSI];
    data[947] = 0.0 + k[550]*y[IDX_HNSII];
    data[948] = 0.0 - k[1766]*y[IDX_NSI] - k[1767]*y[IDX_NSI];
    data[949] = 0.0 - k[190]*y[IDX_H2SI] - k[291]*y[IDX_NH3I] - k[579]*y[IDX_EM] -
        k[580]*y[IDX_EM] - k[581]*y[IDX_EM] - k[2269];
    data[950] = 0.0 + k[1485]*y[IDX_O2I];
    data[951] = 0.0 + k[1501]*y[IDX_OI];
    data[952] = 0.0 + k[80]*y[IDX_OCSI];
    data[953] = 0.0 + k[24]*y[IDX_CII] + k[80]*y[IDX_CH4II] + k[137]*y[IDX_HII] +
        k[184]*y[IDX_H2OII] + k[250]*y[IDX_NII] + k[257]*y[IDX_N2II] +
        k[318]*y[IDX_OII] + k[440] + k[2066];
    data[954] = 0.0 + k[257]*y[IDX_OCSI];
    data[955] = 0.0 + k[250]*y[IDX_OCSI];
    data[956] = 0.0 + k[318]*y[IDX_OCSI];
    data[957] = 0.0 - k[190]*y[IDX_OCSII];
    data[958] = 0.0 + k[184]*y[IDX_OCSI];
    data[959] = 0.0 - k[291]*y[IDX_OCSII];
    data[960] = 0.0 + k[1485]*y[IDX_CSII];
    data[961] = 0.0 + k[24]*y[IDX_OCSI];
    data[962] = 0.0 + k[137]*y[IDX_OCSI];
    data[963] = 0.0 + k[1501]*y[IDX_HCSII];
    data[964] = 0.0 - k[579]*y[IDX_OCSII] - k[580]*y[IDX_OCSII] - k[581]*y[IDX_OCSII];
    data[965] = 0.0 + k[30]*y[IDX_CII] + k[147]*y[IDX_HII] + k[2083];
    data[966] = 0.0 + k[906]*y[IDX_HII] + k[1281]*y[IDX_HeII];
    data[967] = 0.0 - k[593]*y[IDX_EM] - k[594]*y[IDX_EM] - k[595]*y[IDX_EM] -
        k[1514]*y[IDX_OI] - k[1567]*y[IDX_O2I] - k[1568]*y[IDX_SI] - k[2176];
    data[968] = 0.0 + k[1075]*y[IDX_H3II] + k[1095]*y[IDX_H3OII] + k[1155]*y[IDX_HCOII]
        + k[1536]*y[IDX_OHII];
    data[969] = 0.0 + k[673]*y[IDX_C2H2II];
    data[970] = 0.0 + k[2118]*y[IDX_H2I];
    data[971] = 0.0 + k[1536]*y[IDX_SiHI];
    data[972] = 0.0 + k[673]*y[IDX_SiH4I];
    data[973] = 0.0 + k[1095]*y[IDX_SiHI];
    data[974] = 0.0 - k[1567]*y[IDX_SiH2II];
    data[975] = 0.0 + k[30]*y[IDX_SiH2I];
    data[976] = 0.0 - k[1568]*y[IDX_SiH2II];
    data[977] = 0.0 + k[1281]*y[IDX_SiH3I];
    data[978] = 0.0 + k[1075]*y[IDX_SiHI];
    data[979] = 0.0 + k[147]*y[IDX_SiH2I] + k[906]*y[IDX_SiH3I];
    data[980] = 0.0 - k[1514]*y[IDX_SiH2II];
    data[981] = 0.0 + k[1155]*y[IDX_SiHI];
    data[982] = 0.0 + k[2118]*y[IDX_SiII];
    data[983] = 0.0 - k[593]*y[IDX_SiH2II] - k[594]*y[IDX_SiH2II] - k[595]*y[IDX_SiH2II];
    data[984] = 0.0 + k[2499] + k[2500] + k[2501] + k[2502];
    data[985] = 0.0 + k[558]*y[IDX_EM] + k[1008]*y[IDX_H2OI] + k[1438]*y[IDX_NH3I];
    data[986] = 0.0 + k[325]*y[IDX_O2I];
    data[987] = 0.0 - k[141]*y[IDX_HII] - k[218]*y[IDX_HeII] - k[320]*y[IDX_OII] -
        k[445] - k[638]*y[IDX_CII] - k[873]*y[IDX_COII] - k[989]*y[IDX_H2OII] -
        k[1069]*y[IDX_H3II] - k[1270]*y[IDX_HeII] - k[1271]*y[IDX_HeII] -
        k[1481]*y[IDX_OII] - k[1614]*y[IDX_CI] - k[1916]*y[IDX_OI] -
        k[1954]*y[IDX_SI] - k[2074] - k[2260];
    data[988] = 0.0 + k[1866]*y[IDX_O2I] + k[1949]*y[IDX_OHI] + k[2129]*y[IDX_OI];
    data[989] = 0.0 - k[873]*y[IDX_SO2I];
    data[990] = 0.0 - k[320]*y[IDX_SO2I] - k[1481]*y[IDX_SO2I];
    data[991] = 0.0 - k[989]*y[IDX_SO2I];
    data[992] = 0.0 + k[1438]*y[IDX_HSO2II];
    data[993] = 0.0 + k[325]*y[IDX_SO2II] + k[1866]*y[IDX_SOI];
    data[994] = 0.0 - k[638]*y[IDX_SO2I];
    data[995] = 0.0 - k[1954]*y[IDX_SO2I];
    data[996] = 0.0 + k[1949]*y[IDX_SOI];
    data[997] = 0.0 - k[218]*y[IDX_SO2I] - k[1270]*y[IDX_SO2I] - k[1271]*y[IDX_SO2I];
    data[998] = 0.0 - k[1069]*y[IDX_SO2I];
    data[999] = 0.0 - k[141]*y[IDX_SO2I];
    data[1000] = 0.0 - k[1916]*y[IDX_SO2I] + k[2129]*y[IDX_SOI];
    data[1001] = 0.0 - k[1614]*y[IDX_SO2I];
    data[1002] = 0.0 + k[1008]*y[IDX_HSO2II];
    data[1003] = 0.0 + k[558]*y[IDX_HSO2II];
    data[1004] = 0.0 + k[2371] + k[2372] + k[2373] + k[2374];
    data[1005] = 0.0 + k[510]*y[IDX_EM];
    data[1006] = 0.0 + k[1658]*y[IDX_CH3I] + k[1846]*y[IDX_NHI];
    data[1007] = 0.0 - k[416] - k[901]*y[IDX_HII] - k[1051]*y[IDX_H3II] -
        k[1245]*y[IDX_HeII] - k[1246]*y[IDX_HeII] - k[1626]*y[IDX_CH2I] -
        k[1655]*y[IDX_CH3I] - k[1680]*y[IDX_CHI] - k[1708]*y[IDX_CNI] -
        k[1716]*y[IDX_COI] - k[1755]*y[IDX_HI] - k[1756]*y[IDX_HI] -
        k[1757]*y[IDX_HI] - k[1782]*y[IDX_HCOI] - k[1815]*y[IDX_NI] -
        k[1896]*y[IDX_OI] - k[1897]*y[IDX_OI] - k[1898]*y[IDX_OI] -
        k[1942]*y[IDX_OHI] - k[2038] - k[2271];
    data[1008] = 0.0 + k[300]*y[IDX_NOI];
    data[1009] = 0.0 + k[1902]*y[IDX_OI];
    data[1010] = 0.0 - k[1626]*y[IDX_HNOI];
    data[1011] = 0.0 - k[1655]*y[IDX_HNOI] + k[1658]*y[IDX_NO2I];
    data[1012] = 0.0 + k[1846]*y[IDX_NO2I] + k[1849]*y[IDX_O2I] + k[1854]*y[IDX_OHI];
    data[1013] = 0.0 - k[1708]*y[IDX_HNOI];
    data[1014] = 0.0 - k[1782]*y[IDX_HNOI] + k[1783]*y[IDX_NOI];
    data[1015] = 0.0 + k[300]*y[IDX_HNOII] + k[1783]*y[IDX_HCOI];
    data[1016] = 0.0 - k[1680]*y[IDX_HNOI];
    data[1017] = 0.0 + k[1849]*y[IDX_NHI];
    data[1018] = 0.0 + k[1854]*y[IDX_NHI] - k[1942]*y[IDX_HNOI];
    data[1019] = 0.0 - k[1815]*y[IDX_HNOI];
    data[1020] = 0.0 - k[1245]*y[IDX_HNOI] - k[1246]*y[IDX_HNOI];
    data[1021] = 0.0 - k[1051]*y[IDX_HNOI];
    data[1022] = 0.0 - k[901]*y[IDX_HNOI];
    data[1023] = 0.0 - k[1896]*y[IDX_HNOI] - k[1897]*y[IDX_HNOI] - k[1898]*y[IDX_HNOI] +
        k[1902]*y[IDX_NH2I];
    data[1024] = 0.0 - k[1716]*y[IDX_HNOI];
    data[1025] = 0.0 + k[510]*y[IDX_H2NOII];
    data[1026] = 0.0 - k[1755]*y[IDX_HNOI] - k[1756]*y[IDX_HNOI] - k[1757]*y[IDX_HNOI];
    data[1027] = 0.0 + k[452] + k[2084];
    data[1028] = 0.0 + k[597]*y[IDX_EM];
    data[1029] = 0.0 + k[2087];
    data[1030] = 0.0 + k[595]*y[IDX_EM];
    data[1031] = 0.0 - k[150]*y[IDX_HII] - k[355]*y[IDX_SII] - k[455] - k[644]*y[IDX_CII]
        - k[908]*y[IDX_HII] - k[1075]*y[IDX_H3II] - k[1095]*y[IDX_H3OII] -
        k[1155]*y[IDX_HCOII] - k[1284]*y[IDX_HeII] - k[1536]*y[IDX_OHII] -
        k[1569]*y[IDX_SII] - k[1617]*y[IDX_CI] - k[1926]*y[IDX_OI] - k[2091] -
        k[2172];
    data[1032] = 0.0 + k[2090];
    data[1033] = 0.0 - k[1536]*y[IDX_SiHI];
    data[1034] = 0.0 - k[355]*y[IDX_SiHI] - k[1569]*y[IDX_SiHI];
    data[1035] = 0.0 - k[1095]*y[IDX_SiHI];
    data[1036] = 0.0 - k[644]*y[IDX_SiHI];
    data[1037] = 0.0 - k[1284]*y[IDX_SiHI];
    data[1038] = 0.0 - k[1075]*y[IDX_SiHI];
    data[1039] = 0.0 - k[150]*y[IDX_SiHI] - k[908]*y[IDX_SiHI];
    data[1040] = 0.0 - k[1926]*y[IDX_SiHI];
    data[1041] = 0.0 - k[1617]*y[IDX_SiHI];
    data[1042] = 0.0 - k[1155]*y[IDX_SiHI];
    data[1043] = 0.0 + k[595]*y[IDX_SiH2II] + k[597]*y[IDX_SiH3II];
    data[1044] = 0.0 + k[1488]*y[IDX_O2I];
    data[1045] = 0.0 + k[1512]*y[IDX_OI];
    data[1046] = 0.0 - k[206]*y[IDX_HCOI] - k[232]*y[IDX_MgI] - k[305]*y[IDX_NOI] -
        k[602]*y[IDX_EM] - k[659]*y[IDX_C2I] - k[704]*y[IDX_CI] -
        k[773]*y[IDX_CH2I] - k[864]*y[IDX_CHI] - k[881]*y[IDX_COI] -
        k[964]*y[IDX_H2I] - k[1343]*y[IDX_NI] - k[1344]*y[IDX_NI] -
        k[1516]*y[IDX_OI] - k[1555]*y[IDX_SI] - k[2092] - k[2187];
    data[1047] = 0.0 + k[1513]*y[IDX_OI];
    data[1048] = 0.0 + k[151]*y[IDX_HII] + k[2094];
    data[1049] = 0.0 - k[232]*y[IDX_SiOII];
    data[1050] = 0.0 + k[1549]*y[IDX_OHI] + k[2130]*y[IDX_OI];
    data[1051] = 0.0 - k[659]*y[IDX_SiOII];
    data[1052] = 0.0 - k[773]*y[IDX_SiOII];
    data[1053] = 0.0 - k[206]*y[IDX_SiOII];
    data[1054] = 0.0 - k[305]*y[IDX_SiOII];
    data[1055] = 0.0 - k[864]*y[IDX_SiOII];
    data[1056] = 0.0 + k[1488]*y[IDX_SiSII];
    data[1057] = 0.0 - k[1555]*y[IDX_SiOII];
    data[1058] = 0.0 + k[1549]*y[IDX_SiII];
    data[1059] = 0.0 - k[1343]*y[IDX_SiOII] - k[1344]*y[IDX_SiOII];
    data[1060] = 0.0 + k[151]*y[IDX_SiOI];
    data[1061] = 0.0 + k[1512]*y[IDX_SiCII] + k[1513]*y[IDX_SiHII] - k[1516]*y[IDX_SiOII]
        + k[2130]*y[IDX_SiII];
    data[1062] = 0.0 - k[704]*y[IDX_SiOII];
    data[1063] = 0.0 - k[881]*y[IDX_SiOII];
    data[1064] = 0.0 - k[964]*y[IDX_SiOII];
    data[1065] = 0.0 - k[602]*y[IDX_SiOII];
    data[1066] = 0.0 + k[896]*y[IDX_HII] + k[1228]*y[IDX_HeII];
    data[1067] = 0.0 + k[1009]*y[IDX_H2OI];
    data[1068] = 0.0 + k[1563]*y[IDX_SiII];
    data[1069] = 0.0 + k[1515]*y[IDX_OI];
    data[1070] = 0.0 + k[1514]*y[IDX_OI] + k[1567]*y[IDX_O2I];
    data[1071] = 0.0 + k[964]*y[IDX_H2I];
    data[1072] = 0.0 - k[603]*y[IDX_EM] - k[604]*y[IDX_EM] - k[1443]*y[IDX_NH3I] -
        k[2188];
    data[1073] = 0.0 + k[1076]*y[IDX_H3II] + k[1096]*y[IDX_H3OII] + k[1156]*y[IDX_HCOII]
        + k[1537]*y[IDX_OHII];
    data[1074] = 0.0 + k[1564]*y[IDX_SiII];
    data[1075] = 0.0 + k[1013]*y[IDX_H2OI] + k[1563]*y[IDX_C2H5OHI] +
        k[1564]*y[IDX_CH3OHI];
    data[1076] = 0.0 + k[1537]*y[IDX_SiOI];
    data[1077] = 0.0 - k[1443]*y[IDX_SiOHII];
    data[1078] = 0.0 + k[1096]*y[IDX_SiOI];
    data[1079] = 0.0 + k[1567]*y[IDX_SiH2II];
    data[1080] = 0.0 + k[1228]*y[IDX_H2SiOI];
    data[1081] = 0.0 + k[1076]*y[IDX_SiOI];
    data[1082] = 0.0 + k[896]*y[IDX_H2SiOI];
    data[1083] = 0.0 + k[1514]*y[IDX_SiH2II] + k[1515]*y[IDX_SiH3II];
    data[1084] = 0.0 + k[1156]*y[IDX_SiOI];
    data[1085] = 0.0 + k[1009]*y[IDX_HSiSII] + k[1013]*y[IDX_SiII];
    data[1086] = 0.0 + k[964]*y[IDX_SiOII];
    data[1087] = 0.0 - k[603]*y[IDX_SiOHII] - k[604]*y[IDX_SiOHII];
    data[1088] = 0.0 + k[905]*y[IDX_HII] + k[1279]*y[IDX_HeII];
    data[1089] = 0.0 + k[1280]*y[IDX_HeII];
    data[1090] = 0.0 + k[150]*y[IDX_HII] + k[355]*y[IDX_SII];
    data[1091] = 0.0 - k[592]*y[IDX_EM] - k[703]*y[IDX_CI] - k[863]*y[IDX_CHI] -
        k[1014]*y[IDX_H2OI] - k[1108]*y[IDX_HI] - k[1442]*y[IDX_NH3I] -
        k[1513]*y[IDX_OI] - k[2082] - k[2119]*y[IDX_H2I] - k[2174];
    data[1092] = 0.0 + k[672]*y[IDX_C2H2II] + k[1283]*y[IDX_HeII];
    data[1093] = 0.0 + k[1071]*y[IDX_H3II] + k[1093]*y[IDX_H3OII] + k[1535]*y[IDX_OHII] +
        k[1566]*y[IDX_HCOII];
    data[1094] = 0.0 + k[2126]*y[IDX_HI];
    data[1095] = 0.0 + k[1535]*y[IDX_SiI];
    data[1096] = 0.0 + k[672]*y[IDX_SiH4I];
    data[1097] = 0.0 + k[355]*y[IDX_SiHI];
    data[1098] = 0.0 - k[863]*y[IDX_SiHII];
    data[1099] = 0.0 - k[1442]*y[IDX_SiHII];
    data[1100] = 0.0 + k[1093]*y[IDX_SiI];
    data[1101] = 0.0 + k[1279]*y[IDX_SiH2I] + k[1280]*y[IDX_SiH3I] + k[1283]*y[IDX_SiH4I];
    data[1102] = 0.0 + k[1071]*y[IDX_SiI];
    data[1103] = 0.0 + k[150]*y[IDX_SiHI] + k[905]*y[IDX_SiH2I];
    data[1104] = 0.0 - k[1513]*y[IDX_SiHII];
    data[1105] = 0.0 - k[703]*y[IDX_SiHII];
    data[1106] = 0.0 + k[1566]*y[IDX_SiI];
    data[1107] = 0.0 - k[1014]*y[IDX_SiHII];
    data[1108] = 0.0 - k[2119]*y[IDX_SiHII];
    data[1109] = 0.0 - k[592]*y[IDX_SiHII];
    data[1110] = 0.0 - k[1108]*y[IDX_SiHII] + k[2126]*y[IDX_SiII];
    data[1111] = 0.0 + k[2383] + k[2384] + k[2385] + k[2386];
    data[1112] = 0.0 + k[601]*y[IDX_EM] + k[1016]*y[IDX_H2OI];
    data[1113] = 0.0 - k[149]*y[IDX_HII] - k[454] - k[671]*y[IDX_C2H2II] -
        k[672]*y[IDX_C2H2II] - k[673]*y[IDX_C2H2II] - k[674]*y[IDX_C2H2II] -
        k[789]*y[IDX_CH3II] - k[836]*y[IDX_CH5II] - k[907]*y[IDX_HII] -
        k[1074]*y[IDX_H3II] - k[1154]*y[IDX_HCOII] - k[1282]*y[IDX_HeII] -
        k[1283]*y[IDX_HeII] - k[1715]*y[IDX_CNI] - k[1925]*y[IDX_OI] - k[2088] -
        k[2089] - k[2090] - k[2179];
    data[1114] = 0.0 - k[836]*y[IDX_SiH4I];
    data[1115] = 0.0 - k[789]*y[IDX_SiH4I];
    data[1116] = 0.0 - k[671]*y[IDX_SiH4I] - k[672]*y[IDX_SiH4I] - k[673]*y[IDX_SiH4I] -
        k[674]*y[IDX_SiH4I];
    data[1117] = 0.0 - k[1715]*y[IDX_SiH4I];
    data[1118] = 0.0 - k[1282]*y[IDX_SiH4I] - k[1283]*y[IDX_SiH4I];
    data[1119] = 0.0 - k[1074]*y[IDX_SiH4I];
    data[1120] = 0.0 - k[149]*y[IDX_SiH4I] - k[907]*y[IDX_SiH4I];
    data[1121] = 0.0 - k[1925]*y[IDX_SiH4I];
    data[1122] = 0.0 - k[1154]*y[IDX_SiH4I];
    data[1123] = 0.0 + k[1016]*y[IDX_SiH5II];
    data[1124] = 0.0 + k[601]*y[IDX_SiH5II];
    data[1125] = 0.0 + k[2339] + k[2340] + k[2341] + k[2342];
    data[1126] = 0.0 + k[371] + k[1968];
    data[1127] = 0.0 + k[606]*y[IDX_CII];
    data[1128] = 0.0 + k[674]*y[IDX_C2H2II];
    data[1129] = 0.0 - k[369] - k[882]*y[IDX_HII] - k[1181]*y[IDX_HeII] -
        k[1182]*y[IDX_HeII] - k[1576]*y[IDX_O2I] - k[1577]*y[IDX_O2I] -
        k[1582]*y[IDX_CI] - k[1646]*y[IDX_CH3I] - k[1738]*y[IDX_HI] -
        k[1791]*y[IDX_NI] - k[1869]*y[IDX_OI] - k[1930]*y[IDX_OHI] - k[1966] -
        k[2146];
    data[1130] = 0.0 + k[1702]*y[IDX_CNI] + k[1870]*y[IDX_OI];
    data[1131] = 0.0 + k[674]*y[IDX_SiH4I];
    data[1132] = 0.0 + k[1620]*y[IDX_CH2I] + k[1620]*y[IDX_CH2I];
    data[1133] = 0.0 - k[1646]*y[IDX_C2H3I];
    data[1134] = 0.0 + k[1702]*y[IDX_C2H4I];
    data[1135] = 0.0 - k[1576]*y[IDX_C2H3I] - k[1577]*y[IDX_C2H3I];
    data[1136] = 0.0 + k[606]*y[IDX_C2H5OHI];
    data[1137] = 0.0 - k[1930]*y[IDX_C2H3I];
    data[1138] = 0.0 - k[1791]*y[IDX_C2H3I];
    data[1139] = 0.0 - k[1181]*y[IDX_C2H3I] - k[1182]*y[IDX_C2H3I];
    data[1140] = 0.0 - k[882]*y[IDX_C2H3I];
    data[1141] = 0.0 - k[1869]*y[IDX_C2H3I] + k[1870]*y[IDX_C2H4I];
    data[1142] = 0.0 - k[1582]*y[IDX_C2H3I];
    data[1143] = 0.0 - k[1738]*y[IDX_C2H3I];
    data[1144] = 0.0 + k[623]*y[IDX_CII] + k[1229]*y[IDX_HeII];
    data[1145] = 0.0 + k[113]*y[IDX_HII];
    data[1146] = 0.0 + k[20]*y[IDX_CII] + k[1304]*y[IDX_NII];
    data[1147] = 0.0 - k[468]*y[IDX_EM] - k[469]*y[IDX_EM] - k[992]*y[IDX_H2OI] -
        k[993]*y[IDX_H2OI] - k[1018]*y[IDX_H2SI] - k[1421]*y[IDX_NH3I] - k[2160];
    data[1148] = 0.0 + k[1445]*y[IDX_NHI];
    data[1149] = 0.0 + k[627]*y[IDX_CII];
    data[1150] = 0.0 + k[1346]*y[IDX_C2I];
    data[1151] = 0.0 + k[1304]*y[IDX_NCCNI];
    data[1152] = 0.0 + k[1326]*y[IDX_NI];
    data[1153] = 0.0 - k[1018]*y[IDX_C2NII];
    data[1154] = 0.0 + k[1328]*y[IDX_NI];
    data[1155] = 0.0 + k[1346]*y[IDX_NHII];
    data[1156] = 0.0 + k[713]*y[IDX_CNI] + k[723]*y[IDX_HCNI];
    data[1157] = 0.0 + k[1445]*y[IDX_C2II];
    data[1158] = 0.0 + k[713]*y[IDX_CHII];
    data[1159] = 0.0 + k[723]*y[IDX_CHII];
    data[1160] = 0.0 - k[1421]*y[IDX_C2NII];
    data[1161] = 0.0 + k[20]*y[IDX_NCCNI] + k[623]*y[IDX_HC3NI] + k[627]*y[IDX_HNCI];
    data[1162] = 0.0 + k[1326]*y[IDX_C2HII] + k[1328]*y[IDX_C2H2II];
    data[1163] = 0.0 + k[1229]*y[IDX_HC3NI];
    data[1164] = 0.0 + k[113]*y[IDX_C2NI];
    data[1165] = 0.0 - k[992]*y[IDX_C2NII] - k[993]*y[IDX_C2NII];
    data[1166] = 0.0 - k[468]*y[IDX_C2NII] - k[469]*y[IDX_C2NII];
    data[1167] = 0.0 + k[1238]*y[IDX_HeII];
    data[1168] = 0.0 - k[791]*y[IDX_HCO2II];
    data[1169] = 0.0 - k[543]*y[IDX_EM] - k[544]*y[IDX_EM] - k[545]*y[IDX_EM] -
        k[695]*y[IDX_CI] - k[791]*y[IDX_CH3CNI] - k[812]*y[IDX_CH4I] -
        k[875]*y[IDX_COI] - k[1004]*y[IDX_H2OI] - k[1434]*y[IDX_NH3I] -
        k[1500]*y[IDX_OI] - k[2247];
    data[1170] = 0.0 + k[796]*y[IDX_CO2I];
    data[1171] = 0.0 + k[1489]*y[IDX_CO2I];
    data[1172] = 0.0 + k[1322]*y[IDX_CO2I];
    data[1173] = 0.0 + k[1173]*y[IDX_CO2I];
    data[1174] = 0.0 + k[825]*y[IDX_CO2I];
    data[1175] = 0.0 + k[918]*y[IDX_CO2I];
    data[1176] = 0.0 + k[1110]*y[IDX_CO2I];
    data[1177] = 0.0 + k[796]*y[IDX_CH4II] + k[825]*y[IDX_CH5II] + k[918]*y[IDX_H2II] +
        k[1036]*y[IDX_H3II] + k[1110]*y[IDX_HCNII] + k[1173]*y[IDX_HNOII] +
        k[1322]*y[IDX_N2HII] + k[1350]*y[IDX_NHII] + k[1489]*y[IDX_O2HII] +
        k[1520]*y[IDX_OHII];
    data[1178] = 0.0 + k[1350]*y[IDX_CO2I];
    data[1179] = 0.0 + k[1520]*y[IDX_CO2I];
    data[1180] = 0.0 - k[812]*y[IDX_HCO2II];
    data[1181] = 0.0 - k[1434]*y[IDX_HCO2II];
    data[1182] = 0.0 + k[1543]*y[IDX_HCOII];
    data[1183] = 0.0 + k[1238]*y[IDX_HCOOCH3I];
    data[1184] = 0.0 + k[1036]*y[IDX_CO2I];
    data[1185] = 0.0 - k[1500]*y[IDX_HCO2II];
    data[1186] = 0.0 - k[695]*y[IDX_HCO2II];
    data[1187] = 0.0 + k[1543]*y[IDX_OHI];
    data[1188] = 0.0 - k[1004]*y[IDX_HCO2II];
    data[1189] = 0.0 - k[875]*y[IDX_HCO2II];
    data[1190] = 0.0 - k[543]*y[IDX_HCO2II] - k[544]*y[IDX_HCO2II] - k[545]*y[IDX_HCO2II];
    data[1191] = 0.0 + k[1219]*y[IDX_HeII];
    data[1192] = 0.0 + k[899]*y[IDX_HII] + k[1239]*y[IDX_HeII];
    data[1193] = 0.0 + k[632]*y[IDX_CII];
    data[1194] = 0.0 - k[221]*y[IDX_MgI] - k[348]*y[IDX_SiI] - k[500]*y[IDX_EM] -
        k[808]*y[IDX_CH4I] - k[943]*y[IDX_H2I] - k[1485]*y[IDX_O2I] -
        k[1496]*y[IDX_OI] - k[2005] - k[2266];
    data[1195] = 0.0 + k[639]*y[IDX_CII];
    data[1196] = 0.0 + k[118]*y[IDX_HII] + k[395] + k[2006];
    data[1197] = 0.0 - k[221]*y[IDX_CSII];
    data[1198] = 0.0 + k[650]*y[IDX_SI];
    data[1199] = 0.0 + k[636]*y[IDX_CII] + k[1264]*y[IDX_HeII] + k[1312]*y[IDX_NII];
    data[1200] = 0.0 + k[628]*y[IDX_CII];
    data[1201] = 0.0 - k[348]*y[IDX_CSII];
    data[1202] = 0.0 + k[697]*y[IDX_CI];
    data[1203] = 0.0 + k[1312]*y[IDX_OCSI];
    data[1204] = 0.0 + k[658]*y[IDX_SII];
    data[1205] = 0.0 + k[739]*y[IDX_SI];
    data[1206] = 0.0 + k[658]*y[IDX_C2I] + k[861]*y[IDX_CHI] + k[2105]*y[IDX_CI];
    data[1207] = 0.0 - k[808]*y[IDX_CSII];
    data[1208] = 0.0 + k[861]*y[IDX_SII];
    data[1209] = 0.0 - k[1485]*y[IDX_CSII];
    data[1210] = 0.0 + k[628]*y[IDX_HSI] + k[632]*y[IDX_NSI] + k[636]*y[IDX_OCSI] +
        k[639]*y[IDX_SOI] + k[2099]*y[IDX_SI];
    data[1211] = 0.0 + k[650]*y[IDX_C2II] + k[739]*y[IDX_CHII] + k[2099]*y[IDX_CII];
    data[1212] = 0.0 + k[1219]*y[IDX_H2CSI] + k[1239]*y[IDX_HCSI] + k[1264]*y[IDX_OCSI];
    data[1213] = 0.0 + k[118]*y[IDX_CSI] + k[899]*y[IDX_HCSI];
    data[1214] = 0.0 - k[1496]*y[IDX_CSII];
    data[1215] = 0.0 + k[697]*y[IDX_HSII] + k[2105]*y[IDX_SII];
    data[1216] = 0.0 - k[943]*y[IDX_CSII];
    data[1217] = 0.0 - k[500]*y[IDX_CSII];
    data[1218] = 0.0 - k[1164]*y[IDX_HCSII];
    data[1219] = 0.0 + k[412] + k[2033];
    data[1220] = 0.0 + k[1018]*y[IDX_H2SI];
    data[1221] = 0.0 + k[808]*y[IDX_CH4I] + k[943]*y[IDX_H2I];
    data[1222] = 0.0 - k[546]*y[IDX_EM] - k[547]*y[IDX_EM] - k[1164]*y[IDX_C2H5OHI] -
        k[1435]*y[IDX_NH3I] - k[1501]*y[IDX_OI] - k[1502]*y[IDX_OI] - k[2268];
    data[1223] = 0.0 + k[676]*y[IDX_SII];
    data[1224] = 0.0 + k[1039]*y[IDX_H3II] + k[1085]*y[IDX_H3OII] + k[1140]*y[IDX_HCOII];
    data[1225] = 0.0 + k[1558]*y[IDX_C2H2I];
    data[1226] = 0.0 + k[736]*y[IDX_CHII] + k[752]*y[IDX_CH2II];
    data[1227] = 0.0 + k[691]*y[IDX_CI];
    data[1228] = 0.0 + k[1558]*y[IDX_SOII];
    data[1229] = 0.0 + k[746]*y[IDX_H2SI] + k[752]*y[IDX_OCSI] + k[753]*y[IDX_SI];
    data[1230] = 0.0 + k[622]*y[IDX_CII] + k[722]*y[IDX_CHII] + k[746]*y[IDX_CH2II] +
        k[1018]*y[IDX_C2NII];
    data[1231] = 0.0 + k[787]*y[IDX_SI];
    data[1232] = 0.0 + k[772]*y[IDX_SII];
    data[1233] = 0.0 + k[722]*y[IDX_H2SI] + k[736]*y[IDX_OCSI];
    data[1234] = 0.0 + k[676]*y[IDX_C2H4I] + k[772]*y[IDX_CH2I] + k[822]*y[IDX_CH4I];
    data[1235] = 0.0 + k[808]*y[IDX_CSII] + k[822]*y[IDX_SII];
    data[1236] = 0.0 - k[1435]*y[IDX_HCSII];
    data[1237] = 0.0 + k[1085]*y[IDX_CSI];
    data[1238] = 0.0 + k[622]*y[IDX_H2SI];
    data[1239] = 0.0 + k[753]*y[IDX_CH2II] + k[787]*y[IDX_CH3II];
    data[1240] = 0.0 + k[1039]*y[IDX_CSI];
    data[1241] = 0.0 - k[1501]*y[IDX_HCSII] - k[1502]*y[IDX_HCSII];
    data[1242] = 0.0 + k[691]*y[IDX_H2SII];
    data[1243] = 0.0 + k[1140]*y[IDX_CSI];
    data[1244] = 0.0 + k[943]*y[IDX_CSII];
    data[1245] = 0.0 - k[546]*y[IDX_HCSII] - k[547]*y[IDX_HCSII];
    data[1246] = 0.0 + k[2427] + k[2428] + k[2429] + k[2430];
    data[1247] = 0.0 + k[405] + k[2023] + k[2024];
    data[1248] = 0.0 + k[1487]*y[IDX_O2I];
    data[1249] = 0.0 + k[1921]*y[IDX_OI];
    data[1250] = 0.0 + k[1922]*y[IDX_OI] + k[1923]*y[IDX_OI];
    data[1251] = 0.0 + k[1926]*y[IDX_OI];
    data[1252] = 0.0 + k[206]*y[IDX_HCOI] + k[232]*y[IDX_MgI] + k[305]*y[IDX_NOI];
    data[1253] = 0.0 + k[604]*y[IDX_EM] + k[1443]*y[IDX_NH3I];
    data[1254] = 0.0 - k[151]*y[IDX_HII] - k[456] - k[645]*y[IDX_CII] -
        k[1076]*y[IDX_H3II] - k[1096]*y[IDX_H3OII] - k[1156]*y[IDX_HCOII] -
        k[1285]*y[IDX_HeII] - k[1286]*y[IDX_HeII] - k[1537]*y[IDX_OHII] -
        k[2093] - k[2094] - k[2171];
    data[1255] = 0.0 + k[232]*y[IDX_SiOII];
    data[1256] = 0.0 + k[1950]*y[IDX_OHI] + k[1956]*y[IDX_CO2I] + k[1957]*y[IDX_COI] +
        k[1958]*y[IDX_NOI] + k[1959]*y[IDX_O2I] + k[2131]*y[IDX_OI];
    data[1257] = 0.0 + k[1956]*y[IDX_SiI];
    data[1258] = 0.0 - k[1537]*y[IDX_SiOI];
    data[1259] = 0.0 + k[206]*y[IDX_SiOII];
    data[1260] = 0.0 + k[305]*y[IDX_SiOII] + k[1958]*y[IDX_SiI];
    data[1261] = 0.0 + k[1443]*y[IDX_SiOHII];
    data[1262] = 0.0 - k[1096]*y[IDX_SiOI];
    data[1263] = 0.0 + k[1487]*y[IDX_SiSII] + k[1959]*y[IDX_SiI];
    data[1264] = 0.0 - k[645]*y[IDX_SiOI];
    data[1265] = 0.0 + k[1950]*y[IDX_SiI];
    data[1266] = 0.0 - k[1285]*y[IDX_SiOI] - k[1286]*y[IDX_SiOI];
    data[1267] = 0.0 - k[1076]*y[IDX_SiOI];
    data[1268] = 0.0 - k[151]*y[IDX_SiOI];
    data[1269] = 0.0 + k[1921]*y[IDX_SiCI] + k[1922]*y[IDX_SiH2I] + k[1923]*y[IDX_SiH2I]
        + k[1926]*y[IDX_SiHI] + k[2131]*y[IDX_SiI];
    data[1270] = 0.0 - k[1156]*y[IDX_SiOI];
    data[1271] = 0.0 + k[1957]*y[IDX_SiI];
    data[1272] = 0.0 + k[604]*y[IDX_SiOHII];
    data[1273] = 0.0 + k[2379] + k[2380] + k[2381] + k[2382];
    data[1274] = 0.0 + k[536]*y[IDX_EM];
    data[1275] = 0.0 - k[793]*y[IDX_CH3OHI];
    data[1276] = 0.0 + k[489]*y[IDX_EM] + k[792]*y[IDX_NH3I];
    data[1277] = 0.0 - k[388] - k[389] - k[612]*y[IDX_CII] - k[613]*y[IDX_CII] -
        k[708]*y[IDX_CHII] - k[709]*y[IDX_CHII] - k[710]*y[IDX_CHII] -
        k[776]*y[IDX_CH3II] - k[793]*y[IDX_S2II] - k[794]*y[IDX_CH4II] -
        k[887]*y[IDX_HII] - k[888]*y[IDX_HII] - k[889]*y[IDX_HII] -
        k[965]*y[IDX_H2COII] - k[1031]*y[IDX_H3II] - k[1032]*y[IDX_H3II] -
        k[1078]*y[IDX_H3COII] - k[1084]*y[IDX_H3OII] - k[1139]*y[IDX_HCOII] -
        k[1200]*y[IDX_HeII] - k[1201]*y[IDX_HeII] - k[1289]*y[IDX_NII] -
        k[1290]*y[IDX_NII] - k[1291]*y[IDX_NII] - k[1292]*y[IDX_NII] -
        k[1466]*y[IDX_OII] - k[1467]*y[IDX_OII] - k[1483]*y[IDX_O2II] -
        k[1564]*y[IDX_SiII] - k[1990] - k[1991] - k[1992] - k[2154];
    data[1278] = 0.0 - k[794]*y[IDX_CH3OHI];
    data[1279] = 0.0 - k[1078]*y[IDX_CH3OHI];
    data[1280] = 0.0 - k[1564]*y[IDX_CH3OHI];
    data[1281] = 0.0 - k[1483]*y[IDX_CH3OHI];
    data[1282] = 0.0 - k[1289]*y[IDX_CH3OHI] - k[1290]*y[IDX_CH3OHI] -
        k[1291]*y[IDX_CH3OHI] - k[1292]*y[IDX_CH3OHI];
    data[1283] = 0.0 - k[1466]*y[IDX_CH3OHI] - k[1467]*y[IDX_CH3OHI];
    data[1284] = 0.0 - k[965]*y[IDX_CH3OHI];
    data[1285] = 0.0 - k[776]*y[IDX_CH3OHI];
    data[1286] = 0.0 - k[708]*y[IDX_CH3OHI] - k[709]*y[IDX_CH3OHI] - k[710]*y[IDX_CH3OHI];
    data[1287] = 0.0 + k[792]*y[IDX_CH3OH2II];
    data[1288] = 0.0 - k[1084]*y[IDX_CH3OHI];
    data[1289] = 0.0 - k[612]*y[IDX_CH3OHI] - k[613]*y[IDX_CH3OHI];
    data[1290] = 0.0 - k[1200]*y[IDX_CH3OHI] - k[1201]*y[IDX_CH3OHI];
    data[1291] = 0.0 - k[1031]*y[IDX_CH3OHI] - k[1032]*y[IDX_CH3OHI];
    data[1292] = 0.0 - k[887]*y[IDX_CH3OHI] - k[888]*y[IDX_CH3OHI] - k[889]*y[IDX_CH3OHI];
    data[1293] = 0.0 - k[1139]*y[IDX_CH3OHI];
    data[1294] = 0.0 + k[489]*y[IDX_CH3OH2II] + k[536]*y[IDX_H5C2O2II];
    data[1295] = 0.0 + k[1020]*y[IDX_H2SI];
    data[1296] = 0.0 + k[1019]*y[IDX_H2SI];
    data[1297] = 0.0 - k[532]*y[IDX_EM] - k[533]*y[IDX_EM] - k[534]*y[IDX_EM] -
        k[535]*y[IDX_EM] - k[970]*y[IDX_H2COI] - k[1104]*y[IDX_HI] -
        k[1123]*y[IDX_HCNI] - k[1167]*y[IDX_HNCI] - k[1429]*y[IDX_NH3I] -
        k[1553]*y[IDX_SI] - k[2278];
    data[1298] = 0.0 + k[800]*y[IDX_H2SI];
    data[1299] = 0.0 + k[946]*y[IDX_H2I] + k[1017]*y[IDX_H2SI];
    data[1300] = 0.0 + k[829]*y[IDX_H2SI];
    data[1301] = 0.0 + k[1134]*y[IDX_H2SI] + k[1135]*y[IDX_H2SI];
    data[1302] = 0.0 + k[1079]*y[IDX_H2SI];
    data[1303] = 0.0 - k[1167]*y[IDX_H3SII];
    data[1304] = 0.0 + k[1175]*y[IDX_H2SI] + k[2116]*y[IDX_H2I];
    data[1305] = 0.0 + k[1381]*y[IDX_H2SI];
    data[1306] = 0.0 + k[745]*y[IDX_H2SI];
    data[1307] = 0.0 + k[667]*y[IDX_C2H2II] + k[721]*y[IDX_CHII] + k[745]*y[IDX_CH2II] +
        k[800]*y[IDX_CH4II] + k[829]*y[IDX_CH5II] + k[981]*y[IDX_H2OII] +
        k[1017]*y[IDX_H2SII] + k[1019]*y[IDX_HS2II] + k[1020]*y[IDX_HSiSII] +
        k[1044]*y[IDX_H3II] + k[1079]*y[IDX_H3COII] + k[1087]*y[IDX_H3OII] +
        k[1134]*y[IDX_HCNHII] + k[1135]*y[IDX_HCNHII] + k[1143]*y[IDX_HCOII] +
        k[1175]*y[IDX_HSII] + k[1381]*y[IDX_NH2II] + k[1524]*y[IDX_OHII];
    data[1308] = 0.0 + k[981]*y[IDX_H2SI];
    data[1309] = 0.0 + k[1524]*y[IDX_H2SI];
    data[1310] = 0.0 + k[667]*y[IDX_H2SI];
    data[1311] = 0.0 + k[721]*y[IDX_H2SI];
    data[1312] = 0.0 - k[970]*y[IDX_H3SII];
    data[1313] = 0.0 - k[1123]*y[IDX_H3SII];
    data[1314] = 0.0 - k[1429]*y[IDX_H3SII];
    data[1315] = 0.0 + k[1087]*y[IDX_H2SI];
    data[1316] = 0.0 - k[1553]*y[IDX_H3SII];
    data[1317] = 0.0 + k[1044]*y[IDX_H2SI];
    data[1318] = 0.0 + k[1143]*y[IDX_H2SI];
    data[1319] = 0.0 + k[946]*y[IDX_H2SII] + k[2116]*y[IDX_HSII];
    data[1320] = 0.0 - k[532]*y[IDX_H3SII] - k[533]*y[IDX_H3SII] - k[534]*y[IDX_H3SII] -
        k[535]*y[IDX_H3SII];
    data[1321] = 0.0 - k[1104]*y[IDX_H3SII];
    data[1322] = 0.0 + k[2459] + k[2460] + k[2461] + k[2462];
    data[1323] = 0.0 + k[557]*y[IDX_EM];
    data[1324] = 0.0 + k[559]*y[IDX_EM] + k[560]*y[IDX_EM];
    data[1325] = 0.0 + k[586]*y[IDX_EM];
    data[1326] = 0.0 + k[1488]*y[IDX_O2I];
    data[1327] = 0.0 + k[1915]*y[IDX_OI];
    data[1328] = 0.0 + k[1908]*y[IDX_OI];
    data[1329] = 0.0 + k[579]*y[IDX_EM];
    data[1330] = 0.0 + k[445] + k[1614]*y[IDX_CI] + k[1916]*y[IDX_OI] + k[1954]*y[IDX_SI]
        + k[1954]*y[IDX_SI] + k[2074];
    data[1331] = 0.0 + k[1555]*y[IDX_SI];
    data[1332] = 0.0 - k[25]*y[IDX_CII] - k[142]*y[IDX_HII] - k[446] - k[447] -
        k[639]*y[IDX_CII] - k[640]*y[IDX_CII] - k[641]*y[IDX_CII] -
        k[788]*y[IDX_CH3II] - k[1070]*y[IDX_H3II] - k[1152]*y[IDX_HCOII] -
        k[1272]*y[IDX_HeII] - k[1273]*y[IDX_HeII] - k[1615]*y[IDX_CI] -
        k[1616]*y[IDX_CI] - k[1699]*y[IDX_CHI] - k[1700]*y[IDX_CHI] -
        k[1778]*y[IDX_HI] - k[1779]*y[IDX_HI] - k[1830]*y[IDX_NI] -
        k[1831]*y[IDX_NI] - k[1866]*y[IDX_O2I] - k[1917]*y[IDX_OI] -
        k[1949]*y[IDX_OHI] - k[1955]*y[IDX_SI] - k[2075] - k[2076] -
        k[2129]*y[IDX_OI] - k[2256];
    data[1333] = 0.0 + k[1884]*y[IDX_OI];
    data[1334] = 0.0 + k[230]*y[IDX_SOII];
    data[1335] = 0.0 + k[230]*y[IDX_MgI] + k[293]*y[IDX_NH3I];
    data[1336] = 0.0 + k[1913]*y[IDX_OI];
    data[1337] = 0.0 + k[1900]*y[IDX_OI];
    data[1338] = 0.0 - k[788]*y[IDX_SOI];
    data[1339] = 0.0 + k[1862]*y[IDX_SI];
    data[1340] = 0.0 - k[1699]*y[IDX_SOI] - k[1700]*y[IDX_SOI];
    data[1341] = 0.0 + k[293]*y[IDX_SOII];
    data[1342] = 0.0 + k[1488]*y[IDX_SiSII] + k[1865]*y[IDX_SI] - k[1866]*y[IDX_SOI];
    data[1343] = 0.0 - k[25]*y[IDX_SOI] - k[639]*y[IDX_SOI] - k[640]*y[IDX_SOI] -
        k[641]*y[IDX_SOI];
    data[1344] = 0.0 + k[1555]*y[IDX_SiOII] + k[1862]*y[IDX_NOI] + k[1865]*y[IDX_O2I] +
        k[1948]*y[IDX_OHI] + k[1954]*y[IDX_SO2I] + k[1954]*y[IDX_SO2I] -
        k[1955]*y[IDX_SOI];
    data[1345] = 0.0 + k[1948]*y[IDX_SI] - k[1949]*y[IDX_SOI];
    data[1346] = 0.0 - k[1830]*y[IDX_SOI] - k[1831]*y[IDX_SOI];
    data[1347] = 0.0 - k[1272]*y[IDX_SOI] - k[1273]*y[IDX_SOI];
    data[1348] = 0.0 - k[1070]*y[IDX_SOI];
    data[1349] = 0.0 - k[142]*y[IDX_SOI];
    data[1350] = 0.0 + k[1884]*y[IDX_CSI] + k[1900]*y[IDX_HSI] + k[1908]*y[IDX_NSI] +
        k[1913]*y[IDX_OCSI] + k[1915]*y[IDX_S2I] + k[1916]*y[IDX_SO2I] -
        k[1917]*y[IDX_SOI] - k[2129]*y[IDX_SOI];
    data[1351] = 0.0 + k[1614]*y[IDX_SO2I] - k[1615]*y[IDX_SOI] - k[1616]*y[IDX_SOI];
    data[1352] = 0.0 - k[1152]*y[IDX_SOI];
    data[1353] = 0.0 + k[557]*y[IDX_HSOII] + k[559]*y[IDX_HSO2II] + k[560]*y[IDX_HSO2II]
        + k[579]*y[IDX_OCSII] + k[586]*y[IDX_SO2II];
    data[1354] = 0.0 - k[1778]*y[IDX_SOI] - k[1779]*y[IDX_SOI];
    data[1355] = 0.0 + k[2351] + k[2352] + k[2353] + k[2354];
    data[1356] = 0.0 + k[464]*y[IDX_EM];
    data[1357] = 0.0 + k[1793]*y[IDX_NI] + k[1931]*y[IDX_OHI];
    data[1358] = 0.0 + k[775]*y[IDX_CH3II];
    data[1359] = 0.0 + k[671]*y[IDX_C2H2II] + k[673]*y[IDX_C2H2II];
    data[1360] = 0.0 - k[370] - k[676]*y[IDX_SII] - k[774]*y[IDX_CH3II] -
        k[883]*y[IDX_HII] - k[910]*y[IDX_H2II] - k[1183]*y[IDX_HeII] -
        k[1184]*y[IDX_HeII] - k[1185]*y[IDX_HeII] - k[1464]*y[IDX_OII] -
        k[1559]*y[IDX_SOII] - k[1560]*y[IDX_SOII] - k[1561]*y[IDX_SOII] -
        k[1675]*y[IDX_CHI] - k[1702]*y[IDX_CNI] - k[1792]*y[IDX_NI] -
        k[1870]*y[IDX_OI] - k[1871]*y[IDX_OI] - k[1872]*y[IDX_OI] -
        k[1873]*y[IDX_OI] - k[1967] - k[2121]*y[IDX_H3OII] - k[2300];
    data[1361] = 0.0 - k[1559]*y[IDX_C2H4I] - k[1560]*y[IDX_C2H4I] - k[1561]*y[IDX_C2H4I];
    data[1362] = 0.0 - k[910]*y[IDX_C2H4I];
    data[1363] = 0.0 - k[1464]*y[IDX_C2H4I];
    data[1364] = 0.0 - k[774]*y[IDX_C2H4I] + k[775]*y[IDX_CH3CNI];
    data[1365] = 0.0 + k[671]*y[IDX_SiH4I] + k[673]*y[IDX_SiH4I];
    data[1366] = 0.0 + k[1647]*y[IDX_CH3I] + k[1647]*y[IDX_CH3I];
    data[1367] = 0.0 - k[676]*y[IDX_C2H4I];
    data[1368] = 0.0 - k[1702]*y[IDX_C2H4I];
    data[1369] = 0.0 + k[1676]*y[IDX_CHI];
    data[1370] = 0.0 - k[1675]*y[IDX_C2H4I] + k[1676]*y[IDX_CH4I];
    data[1371] = 0.0 - k[2121]*y[IDX_C2H4I];
    data[1372] = 0.0 + k[1931]*y[IDX_C2H5I];
    data[1373] = 0.0 - k[1792]*y[IDX_C2H4I] + k[1793]*y[IDX_C2H5I];
    data[1374] = 0.0 - k[1183]*y[IDX_C2H4I] - k[1184]*y[IDX_C2H4I] - k[1185]*y[IDX_C2H4I];
    data[1375] = 0.0 - k[883]*y[IDX_C2H4I];
    data[1376] = 0.0 - k[1870]*y[IDX_C2H4I] - k[1871]*y[IDX_C2H4I] - k[1872]*y[IDX_C2H4I]
        - k[1873]*y[IDX_C2H4I];
    data[1377] = 0.0 + k[464]*y[IDX_C2H5OH2II];
    data[1378] = 0.0 - k[794]*y[IDX_CH4II];
    data[1379] = 0.0 - k[75]*y[IDX_C2H2I] - k[76]*y[IDX_H2COI] - k[77]*y[IDX_H2SI] -
        k[78]*y[IDX_NH3I] - k[79]*y[IDX_O2I] - k[80]*y[IDX_OCSI] -
        k[491]*y[IDX_EM] - k[492]*y[IDX_EM] - k[794]*y[IDX_CH3OHI] -
        k[795]*y[IDX_CH4I] - k[796]*y[IDX_CO2I] - k[797]*y[IDX_COI] -
        k[798]*y[IDX_H2COI] - k[799]*y[IDX_H2OI] - k[800]*y[IDX_H2SI] -
        k[801]*y[IDX_NH3I] - k[802]*y[IDX_OCSI] - k[939]*y[IDX_H2I] -
        k[1101]*y[IDX_HI] - k[1493]*y[IDX_OI] - k[1993] - k[1994] - k[2249];
    data[1380] = 0.0 - k[80]*y[IDX_CH4II] - k[802]*y[IDX_CH4II];
    data[1381] = 0.0 + k[1102]*y[IDX_HI];
    data[1382] = 0.0 + k[157]*y[IDX_CH4I];
    data[1383] = 0.0 - k[75]*y[IDX_CH4II];
    data[1384] = 0.0 + k[81]*y[IDX_CH4I];
    data[1385] = 0.0 - k[796]*y[IDX_CH4II];
    data[1386] = 0.0 + k[236]*y[IDX_CH4I];
    data[1387] = 0.0 + k[309]*y[IDX_CH4I];
    data[1388] = 0.0 - k[77]*y[IDX_CH4II] - k[800]*y[IDX_CH4II];
    data[1389] = 0.0 + k[779]*y[IDX_HCOI];
    data[1390] = 0.0 + k[1029]*y[IDX_H3II];
    data[1391] = 0.0 - k[76]*y[IDX_CH4II] - k[798]*y[IDX_CH4II];
    data[1392] = 0.0 + k[81]*y[IDX_COII] + k[116]*y[IDX_HII] + k[157]*y[IDX_H2II] +
        k[210]*y[IDX_HeII] + k[236]*y[IDX_NII] + k[309]*y[IDX_OII] -
        k[795]*y[IDX_CH4II] + k[1997];
    data[1393] = 0.0 + k[779]*y[IDX_CH3II];
    data[1394] = 0.0 - k[78]*y[IDX_CH4II] - k[801]*y[IDX_CH4II];
    data[1395] = 0.0 - k[79]*y[IDX_CH4II];
    data[1396] = 0.0 + k[210]*y[IDX_CH4I];
    data[1397] = 0.0 + k[1029]*y[IDX_CH3I];
    data[1398] = 0.0 + k[116]*y[IDX_CH4I];
    data[1399] = 0.0 - k[1493]*y[IDX_CH4II];
    data[1400] = 0.0 - k[799]*y[IDX_CH4II];
    data[1401] = 0.0 - k[797]*y[IDX_CH4II];
    data[1402] = 0.0 - k[939]*y[IDX_CH4II];
    data[1403] = 0.0 - k[491]*y[IDX_CH4II] - k[492]*y[IDX_CH4II];
    data[1404] = 0.0 - k[1101]*y[IDX_CH4II] + k[1102]*y[IDX_CH5II];
    data[1405] = 0.0 + k[2431] + k[2432] + k[2433] + k[2434];
    data[1406] = 0.0 + k[552]*y[IDX_EM];
    data[1407] = 0.0 + k[526]*y[IDX_EM];
    data[1408] = 0.0 + k[400] + k[619]*y[IDX_CII] + k[2015];
    data[1409] = 0.0 + k[506]*y[IDX_EM];
    data[1410] = 0.0 + k[1164]*y[IDX_HCSII];
    data[1411] = 0.0 + k[1240]*y[IDX_HeII] + k[1753]*y[IDX_HI];
    data[1412] = 0.0 + k[1613]*y[IDX_CI];
    data[1413] = 0.0 + k[1606]*y[IDX_CI];
    data[1414] = 0.0 + k[580]*y[IDX_EM];
    data[1415] = 0.0 + k[221]*y[IDX_MgI] + k[348]*y[IDX_SiI];
    data[1416] = 0.0 + k[547]*y[IDX_EM] + k[1164]*y[IDX_C2H5OHI] + k[1435]*y[IDX_NH3I];
    data[1417] = 0.0 + k[1615]*y[IDX_CI];
    data[1418] = 0.0 - k[118]*y[IDX_HII] - k[395] - k[396] - k[1039]*y[IDX_H3II] -
        k[1085]*y[IDX_H3OII] - k[1140]*y[IDX_HCOII] - k[1214]*y[IDX_HeII] -
        k[1215]*y[IDX_HeII] - k[1592]*y[IDX_CI] - k[1809]*y[IDX_NI] -
        k[1883]*y[IDX_OI] - k[1884]*y[IDX_OI] - k[1935]*y[IDX_OHI] -
        k[1936]*y[IDX_OHI] - k[2006] - k[2007] - k[2255];
    data[1419] = 0.0 + k[221]*y[IDX_CSII];
    data[1420] = 0.0 + k[1265]*y[IDX_HeII] + k[1610]*y[IDX_CI] + k[1695]*y[IDX_CHI];
    data[1421] = 0.0 + k[1595]*y[IDX_CI];
    data[1422] = 0.0 + k[348]*y[IDX_CSII];
    data[1423] = 0.0 + k[1573]*y[IDX_SI];
    data[1424] = 0.0 + k[1644]*y[IDX_SI];
    data[1425] = 0.0 + k[1695]*y[IDX_OCSI] + k[1697]*y[IDX_SI];
    data[1426] = 0.0 + k[1435]*y[IDX_HCSII];
    data[1427] = 0.0 - k[1085]*y[IDX_CSI];
    data[1428] = 0.0 + k[619]*y[IDX_H2CSI];
    data[1429] = 0.0 + k[1573]*y[IDX_C2I] + k[1644]*y[IDX_CH2I] + k[1697]*y[IDX_CHI] +
        k[2106]*y[IDX_CI];
    data[1430] = 0.0 - k[1935]*y[IDX_CSI] - k[1936]*y[IDX_CSI];
    data[1431] = 0.0 - k[1809]*y[IDX_CSI];
    data[1432] = 0.0 - k[1214]*y[IDX_CSI] - k[1215]*y[IDX_CSI] + k[1240]*y[IDX_HCSI] +
        k[1265]*y[IDX_OCSI];
    data[1433] = 0.0 - k[1039]*y[IDX_CSI];
    data[1434] = 0.0 - k[118]*y[IDX_CSI];
    data[1435] = 0.0 - k[1883]*y[IDX_CSI] - k[1884]*y[IDX_CSI];
    data[1436] = 0.0 - k[1592]*y[IDX_CSI] + k[1595]*y[IDX_HSI] + k[1606]*y[IDX_NSI] +
        k[1610]*y[IDX_OCSI] + k[1613]*y[IDX_S2I] + k[1615]*y[IDX_SOI] +
        k[2106]*y[IDX_SI];
    data[1437] = 0.0 - k[1140]*y[IDX_CSI];
    data[1438] = 0.0 + k[506]*y[IDX_H2CSII] + k[526]*y[IDX_H3CSII] + k[547]*y[IDX_HCSII]
        + k[552]*y[IDX_HOCSII] + k[580]*y[IDX_OCSII];
    data[1439] = 0.0 + k[1753]*y[IDX_HCSI];
    data[1440] = 0.0 + k[2319] + k[2320] + k[2321] + k[2322];
    data[1441] = 0.0 - k[232]*y[IDX_MgI];
    data[1442] = 0.0 - k[221]*y[IDX_MgI];
    data[1443] = 0.0 - k[19]*y[IDX_CII] - k[56]*y[IDX_CHII] - k[73]*y[IDX_CH3II] -
        k[129]*y[IDX_HII] - k[181]*y[IDX_H2OII] - k[220]*y[IDX_C2H2II] -
        k[221]*y[IDX_CSII] - k[222]*y[IDX_H2COII] - k[223]*y[IDX_H2SII] -
        k[224]*y[IDX_HCOII] - k[225]*y[IDX_HSII] - k[226]*y[IDX_N2II] -
        k[227]*y[IDX_NOII] - k[228]*y[IDX_O2II] - k[229]*y[IDX_SII] -
        k[230]*y[IDX_SOII] - k[231]*y[IDX_SiII] - k[232]*y[IDX_SiOII] -
        k[244]*y[IDX_NII] - k[279]*y[IDX_NH3II] - k[420] - k[834]*y[IDX_CH5II] -
        k[1054]*y[IDX_H3II] - k[2044] - k[2287];
    data[1444] = 0.0 + k[2140]*y[IDX_EM];
    data[1445] = 0.0 - k[230]*y[IDX_MgI];
    data[1446] = 0.0 - k[223]*y[IDX_MgI];
    data[1447] = 0.0 - k[834]*y[IDX_MgI];
    data[1448] = 0.0 - k[226]*y[IDX_MgI];
    data[1449] = 0.0 - k[231]*y[IDX_MgI];
    data[1450] = 0.0 - k[225]*y[IDX_MgI];
    data[1451] = 0.0 - k[228]*y[IDX_MgI];
    data[1452] = 0.0 - k[244]*y[IDX_MgI];
    data[1453] = 0.0 - k[227]*y[IDX_MgI];
    data[1454] = 0.0 - k[222]*y[IDX_MgI];
    data[1455] = 0.0 - k[279]*y[IDX_MgI];
    data[1456] = 0.0 - k[181]*y[IDX_MgI];
    data[1457] = 0.0 - k[73]*y[IDX_MgI];
    data[1458] = 0.0 - k[220]*y[IDX_MgI];
    data[1459] = 0.0 - k[56]*y[IDX_MgI];
    data[1460] = 0.0 - k[229]*y[IDX_MgI];
    data[1461] = 0.0 - k[19]*y[IDX_MgI];
    data[1462] = 0.0 - k[1054]*y[IDX_MgI];
    data[1463] = 0.0 - k[129]*y[IDX_MgI];
    data[1464] = 0.0 - k[224]*y[IDX_MgI];
    data[1465] = 0.0 + k[2140]*y[IDX_MgII];
    data[1466] = 0.0 + k[232]*y[IDX_MgI];
    data[1467] = 0.0 + k[221]*y[IDX_MgI];
    data[1468] = 0.0 + k[19]*y[IDX_CII] + k[56]*y[IDX_CHII] + k[73]*y[IDX_CH3II] +
        k[129]*y[IDX_HII] + k[181]*y[IDX_H2OII] + k[220]*y[IDX_C2H2II] +
        k[221]*y[IDX_CSII] + k[222]*y[IDX_H2COII] + k[223]*y[IDX_H2SII] +
        k[224]*y[IDX_HCOII] + k[225]*y[IDX_HSII] + k[226]*y[IDX_N2II] +
        k[227]*y[IDX_NOII] + k[228]*y[IDX_O2II] + k[229]*y[IDX_SII] +
        k[230]*y[IDX_SOII] + k[231]*y[IDX_SiII] + k[232]*y[IDX_SiOII] +
        k[244]*y[IDX_NII] + k[279]*y[IDX_NH3II] + k[420] + k[834]*y[IDX_CH5II] +
        k[1054]*y[IDX_H3II] + k[2044];
    data[1469] = 0.0 - k[2140]*y[IDX_EM] - k[2286];
    data[1470] = 0.0 + k[230]*y[IDX_MgI];
    data[1471] = 0.0 + k[223]*y[IDX_MgI];
    data[1472] = 0.0 + k[834]*y[IDX_MgI];
    data[1473] = 0.0 + k[226]*y[IDX_MgI];
    data[1474] = 0.0 + k[231]*y[IDX_MgI];
    data[1475] = 0.0 + k[225]*y[IDX_MgI];
    data[1476] = 0.0 + k[228]*y[IDX_MgI];
    data[1477] = 0.0 + k[244]*y[IDX_MgI];
    data[1478] = 0.0 + k[227]*y[IDX_MgI];
    data[1479] = 0.0 + k[222]*y[IDX_MgI];
    data[1480] = 0.0 + k[279]*y[IDX_MgI];
    data[1481] = 0.0 + k[181]*y[IDX_MgI];
    data[1482] = 0.0 + k[73]*y[IDX_MgI];
    data[1483] = 0.0 + k[220]*y[IDX_MgI];
    data[1484] = 0.0 + k[56]*y[IDX_MgI];
    data[1485] = 0.0 + k[229]*y[IDX_MgI];
    data[1486] = 0.0 + k[19]*y[IDX_MgI];
    data[1487] = 0.0 + k[1054]*y[IDX_MgI];
    data[1488] = 0.0 + k[129]*y[IDX_MgI];
    data[1489] = 0.0 + k[224]*y[IDX_MgI];
    data[1490] = 0.0 - k[2140]*y[IDX_MgII];
    data[1491] = 0.0 + k[879]*y[IDX_COI] + k[1107]*y[IDX_HI];
    data[1492] = 0.0 + k[1487]*y[IDX_O2I];
    data[1493] = 0.0 + k[638]*y[IDX_CII] + k[873]*y[IDX_COII] + k[1271]*y[IDX_HeII] +
        k[1481]*y[IDX_OII];
    data[1494] = 0.0 + k[25]*y[IDX_CII] + k[142]*y[IDX_HII] + k[447] + k[2076];
    data[1495] = 0.0 - k[1559]*y[IDX_SOII] - k[1560]*y[IDX_SOII] - k[1561]*y[IDX_SOII];
    data[1496] = 0.0 - k[230]*y[IDX_SOII];
    data[1497] = 0.0 - k[230]*y[IDX_MgI] - k[293]*y[IDX_NH3I] - k[584]*y[IDX_EM] -
        k[1021]*y[IDX_H2SI] - k[1341]*y[IDX_NI] - k[1556]*y[IDX_C2H2I] -
        k[1557]*y[IDX_C2H2I] - k[1558]*y[IDX_C2H2I] - k[1559]*y[IDX_C2H4I] -
        k[1560]*y[IDX_C2H4I] - k[1561]*y[IDX_C2H4I] - k[1562]*y[IDX_OCSI] -
        k[2267];
    data[1498] = 0.0 - k[1562]*y[IDX_SOII];
    data[1499] = 0.0 + k[1499]*y[IDX_OI];
    data[1500] = 0.0 - k[1556]*y[IDX_SOII] - k[1557]*y[IDX_SOII] - k[1558]*y[IDX_SOII];
    data[1501] = 0.0 + k[873]*y[IDX_SO2I];
    data[1502] = 0.0 + k[1504]*y[IDX_OI];
    data[1503] = 0.0 + k[1484]*y[IDX_SI];
    data[1504] = 0.0 + k[1481]*y[IDX_SO2I];
    data[1505] = 0.0 - k[1021]*y[IDX_SOII];
    data[1506] = 0.0 + k[1534]*y[IDX_SI];
    data[1507] = 0.0 + k[1486]*y[IDX_O2I] + k[1548]*y[IDX_OHI];
    data[1508] = 0.0 - k[293]*y[IDX_SOII];
    data[1509] = 0.0 + k[1486]*y[IDX_SII] + k[1487]*y[IDX_SiSII];
    data[1510] = 0.0 + k[25]*y[IDX_SOI] + k[638]*y[IDX_SO2I];
    data[1511] = 0.0 + k[1484]*y[IDX_O2II] + k[1534]*y[IDX_OHII];
    data[1512] = 0.0 + k[1548]*y[IDX_SII];
    data[1513] = 0.0 - k[1341]*y[IDX_SOII];
    data[1514] = 0.0 + k[1271]*y[IDX_SO2I];
    data[1515] = 0.0 + k[142]*y[IDX_SOI];
    data[1516] = 0.0 + k[1499]*y[IDX_H2SII] + k[1504]*y[IDX_HSII];
    data[1517] = 0.0 + k[879]*y[IDX_SO2II];
    data[1518] = 0.0 - k[584]*y[IDX_SOII];
    data[1519] = 0.0 + k[1107]*y[IDX_SO2II];
    data[1520] = 0.0 + k[1190]*y[IDX_HeII];
    data[1521] = 0.0 - k[33]*y[IDX_HCOI] - k[34]*y[IDX_NOI] - k[35]*y[IDX_SI] -
        k[50]*y[IDX_CI] - k[62]*y[IDX_CH2I] - k[82]*y[IDX_CHI] -
        k[271]*y[IDX_NH2I] - k[339]*y[IDX_OHI] - k[458]*y[IDX_EM] -
        k[647]*y[IDX_C2I] - k[648]*y[IDX_HCOI] - k[649]*y[IDX_O2I] -
        k[650]*y[IDX_SI] - k[803]*y[IDX_CH4I] - k[804]*y[IDX_CH4I] -
        k[837]*y[IDX_CHI] - k[935]*y[IDX_H2I] - k[990]*y[IDX_H2OI] -
        k[1325]*y[IDX_NI] - k[1394]*y[IDX_NH2I] - k[1444]*y[IDX_NHI] -
        k[1445]*y[IDX_NHI] - k[1490]*y[IDX_OI] - k[1960] - k[2228];
    data[1522] = 0.0 + k[36]*y[IDX_C2I];
    data[1523] = 0.0 + k[38]*y[IDX_C2I];
    data[1524] = 0.0 + k[153]*y[IDX_C2I];
    data[1525] = 0.0 + k[1178]*y[IDX_HeII];
    data[1526] = 0.0 + k[37]*y[IDX_C2I];
    data[1527] = 0.0 - k[271]*y[IDX_C2II] - k[1394]*y[IDX_C2II];
    data[1528] = 0.0 + k[39]*y[IDX_C2I];
    data[1529] = 0.0 + k[233]*y[IDX_C2I];
    data[1530] = 0.0 + k[1963];
    data[1531] = 0.0 + k[884]*y[IDX_HII] + k[1186]*y[IDX_HeII];
    data[1532] = 0.0 + k[306]*y[IDX_C2I];
    data[1533] = 0.0 + k[175]*y[IDX_C2I];
    data[1534] = 0.0 + k[329]*y[IDX_C2I];
    data[1535] = 0.0 + k[36]*y[IDX_CNII] + k[37]*y[IDX_COII] + k[38]*y[IDX_N2II] +
        k[39]*y[IDX_O2II] + k[110]*y[IDX_HII] + k[153]*y[IDX_H2II] +
        k[175]*y[IDX_H2OII] + k[207]*y[IDX_HeII] + k[233]*y[IDX_NII] +
        k[306]*y[IDX_OII] + k[329]*y[IDX_OHII] - k[647]*y[IDX_C2II] + k[1961];
    data[1536] = 0.0 - k[62]*y[IDX_C2II];
    data[1537] = 0.0 + k[686]*y[IDX_CI] + k[712]*y[IDX_CHI];
    data[1538] = 0.0 - k[1444]*y[IDX_C2II] - k[1445]*y[IDX_C2II];
    data[1539] = 0.0 - k[803]*y[IDX_C2II] - k[804]*y[IDX_C2II];
    data[1540] = 0.0 - k[33]*y[IDX_C2II] - k[648]*y[IDX_C2II];
    data[1541] = 0.0 - k[34]*y[IDX_C2II];
    data[1542] = 0.0 - k[82]*y[IDX_C2II] + k[615]*y[IDX_CII] + k[712]*y[IDX_CHII] -
        k[837]*y[IDX_C2II];
    data[1543] = 0.0 - k[649]*y[IDX_C2II];
    data[1544] = 0.0 + k[615]*y[IDX_CHI] + k[2096]*y[IDX_CI];
    data[1545] = 0.0 - k[35]*y[IDX_C2II] - k[650]*y[IDX_C2II];
    data[1546] = 0.0 - k[339]*y[IDX_C2II];
    data[1547] = 0.0 - k[1325]*y[IDX_C2II];
    data[1548] = 0.0 + k[207]*y[IDX_C2I] + k[1178]*y[IDX_C2H2I] + k[1186]*y[IDX_C2HI] +
        k[1190]*y[IDX_C3NI];
    data[1549] = 0.0 + k[110]*y[IDX_C2I] + k[884]*y[IDX_C2HI];
    data[1550] = 0.0 - k[1490]*y[IDX_C2II];
    data[1551] = 0.0 - k[50]*y[IDX_C2II] + k[686]*y[IDX_CHII] + k[2096]*y[IDX_CII];
    data[1552] = 0.0 - k[990]*y[IDX_C2II];
    data[1553] = 0.0 - k[935]*y[IDX_C2II];
    data[1554] = 0.0 - k[458]*y[IDX_C2II];
    data[1555] = 0.0 - k[578]*y[IDX_EM] - k[657]*y[IDX_C2I] - k[683]*y[IDX_C2HI] -
        k[701]*y[IDX_CI] - k[770]*y[IDX_CH2I] - k[859]*y[IDX_CHI] -
        k[870]*y[IDX_CNI] - k[878]*y[IDX_COI] - k[959]*y[IDX_H2I] -
        k[973]*y[IDX_H2COI] - k[1012]*y[IDX_H2OI] - k[1129]*y[IDX_HCNI] -
        k[1162]*y[IDX_HCOI] - k[1172]*y[IDX_HNCI] - k[1320]*y[IDX_N2I] -
        k[1410]*y[IDX_NH2I] - k[1441]*y[IDX_NH3I] - k[1459]*y[IDX_NHI] -
        k[1462]*y[IDX_NOI] - k[1489]*y[IDX_CO2I] - k[1510]*y[IDX_OI] -
        k[1547]*y[IDX_OHI] - k[1554]*y[IDX_SI] - k[2274];
    data[1556] = 0.0 + k[931]*y[IDX_O2I];
    data[1557] = 0.0 - k[1172]*y[IDX_O2HII];
    data[1558] = 0.0 - k[1489]*y[IDX_O2HII];
    data[1559] = 0.0 + k[1369]*y[IDX_O2I];
    data[1560] = 0.0 - k[1410]*y[IDX_O2HII];
    data[1561] = 0.0 + k[1161]*y[IDX_HCOI];
    data[1562] = 0.0 - k[683]*y[IDX_O2HII];
    data[1563] = 0.0 - k[1320]*y[IDX_O2HII];
    data[1564] = 0.0 - k[657]*y[IDX_O2HII];
    data[1565] = 0.0 - k[770]*y[IDX_O2HII];
    data[1566] = 0.0 - k[1459]*y[IDX_O2HII];
    data[1567] = 0.0 - k[870]*y[IDX_O2HII];
    data[1568] = 0.0 - k[973]*y[IDX_O2HII];
    data[1569] = 0.0 + k[1161]*y[IDX_O2II] - k[1162]*y[IDX_O2HII];
    data[1570] = 0.0 - k[1129]*y[IDX_O2HII];
    data[1571] = 0.0 - k[1462]*y[IDX_O2HII];
    data[1572] = 0.0 - k[859]*y[IDX_O2HII];
    data[1573] = 0.0 - k[1441]*y[IDX_O2HII];
    data[1574] = 0.0 + k[931]*y[IDX_H2II] + k[1062]*y[IDX_H3II] + k[1369]*y[IDX_NHII];
    data[1575] = 0.0 - k[1554]*y[IDX_O2HII];
    data[1576] = 0.0 - k[1547]*y[IDX_O2HII];
    data[1577] = 0.0 + k[1062]*y[IDX_O2I];
    data[1578] = 0.0 - k[1510]*y[IDX_O2HII];
    data[1579] = 0.0 - k[701]*y[IDX_O2HII];
    data[1580] = 0.0 - k[1012]*y[IDX_O2HII];
    data[1581] = 0.0 - k[878]*y[IDX_O2HII];
    data[1582] = 0.0 - k[959]*y[IDX_O2HII];
    data[1583] = 0.0 - k[578]*y[IDX_O2HII];
    data[1584] = 0.0 + k[1251]*y[IDX_HeII];
    data[1585] = 0.0 + k[1262]*y[IDX_HeII];
    data[1586] = 0.0 + k[1198]*y[IDX_HeII];
    data[1587] = 0.0 - k[36]*y[IDX_C2I] - k[47]*y[IDX_C2HI] - k[51]*y[IDX_CI] -
        k[63]*y[IDX_CH2I] - k[83]*y[IDX_CHI] - k[93]*y[IDX_COI] -
        k[94]*y[IDX_H2COI] - k[95]*y[IDX_HCNI] - k[96]*y[IDX_HCOI] -
        k[97]*y[IDX_NOI] - k[98]*y[IDX_O2I] - k[99]*y[IDX_SI] - k[191]*y[IDX_HI]
        - k[272]*y[IDX_NH2I] - k[294]*y[IDX_NHI] - k[326]*y[IDX_OI] -
        k[340]*y[IDX_OHI] - k[498]*y[IDX_EM] - k[865]*y[IDX_H2COI] -
        k[866]*y[IDX_HCNI] - k[867]*y[IDX_HCOI] - k[868]*y[IDX_O2I] -
        k[940]*y[IDX_H2I] - k[995]*y[IDX_H2OI] - k[996]*y[IDX_H2OI] -
        k[1332]*y[IDX_NI] - k[2235];
    data[1588] = 0.0 + k[100]*y[IDX_CNI];
    data[1589] = 0.0 + k[159]*y[IDX_CNI];
    data[1590] = 0.0 + k[1242]*y[IDX_HeII];
    data[1591] = 0.0 - k[272]*y[IDX_CNII];
    data[1592] = 0.0 + k[237]*y[IDX_CNI] + k[852]*y[IDX_CHI];
    data[1593] = 0.0 - k[47]*y[IDX_CNII];
    data[1594] = 0.0 - k[36]*y[IDX_CNII];
    data[1595] = 0.0 - k[63]*y[IDX_CNII];
    data[1596] = 0.0 + k[728]*y[IDX_NI] + k[731]*y[IDX_NHI];
    data[1597] = 0.0 - k[294]*y[IDX_CNII] + k[631]*y[IDX_CII] + k[731]*y[IDX_CHII];
    data[1598] = 0.0 + k[100]*y[IDX_N2II] + k[159]*y[IDX_H2II] + k[237]*y[IDX_NII];
    data[1599] = 0.0 - k[94]*y[IDX_CNII] - k[865]*y[IDX_CNII];
    data[1600] = 0.0 - k[96]*y[IDX_CNII] - k[867]*y[IDX_CNII];
    data[1601] = 0.0 - k[95]*y[IDX_CNII] - k[866]*y[IDX_CNII] + k[1231]*y[IDX_HeII];
    data[1602] = 0.0 - k[97]*y[IDX_CNII];
    data[1603] = 0.0 - k[83]*y[IDX_CNII] + k[852]*y[IDX_NII];
    data[1604] = 0.0 - k[98]*y[IDX_CNII] - k[868]*y[IDX_CNII];
    data[1605] = 0.0 + k[631]*y[IDX_NHI] + k[2097]*y[IDX_NI];
    data[1606] = 0.0 - k[99]*y[IDX_CNII];
    data[1607] = 0.0 - k[340]*y[IDX_CNII];
    data[1608] = 0.0 + k[728]*y[IDX_CHII] - k[1332]*y[IDX_CNII] + k[2097]*y[IDX_CII];
    data[1609] = 0.0 + k[1198]*y[IDX_CH3CNI] + k[1231]*y[IDX_HCNI] + k[1242]*y[IDX_HNCI]
        + k[1251]*y[IDX_NCCNI] + k[1262]*y[IDX_OCNI];
    data[1610] = 0.0 - k[326]*y[IDX_CNII];
    data[1611] = 0.0 - k[51]*y[IDX_CNII];
    data[1612] = 0.0 - k[995]*y[IDX_CNII] - k[996]*y[IDX_CNII];
    data[1613] = 0.0 - k[93]*y[IDX_CNII];
    data[1614] = 0.0 - k[940]*y[IDX_CNII];
    data[1615] = 0.0 - k[498]*y[IDX_CNII];
    data[1616] = 0.0 - k[191]*y[IDX_CNII];
    data[1617] = 0.0 - k[1321]*y[IDX_N2HII];
    data[1618] = 0.0 + k[1320]*y[IDX_N2I];
    data[1619] = 0.0 - k[565]*y[IDX_EM] - k[566]*y[IDX_EM] - k[655]*y[IDX_C2I] -
        k[682]*y[IDX_C2HI] - k[698]*y[IDX_CI] - k[765]*y[IDX_CH2I] -
        k[817]*y[IDX_CH4I] - k[853]*y[IDX_CHI] - k[877]*y[IDX_COI] -
        k[1011]*y[IDX_H2OI] - k[1128]*y[IDX_HCNI] - k[1160]*y[IDX_HCOI] -
        k[1171]*y[IDX_HNCI] - k[1321]*y[IDX_CH3CNI] - k[1322]*y[IDX_CO2I] -
        k[1323]*y[IDX_H2COI] - k[1324]*y[IDX_SI] - k[1408]*y[IDX_NH2I] -
        k[1440]*y[IDX_NH3I] - k[1454]*y[IDX_NHI] - k[1506]*y[IDX_OI] -
        k[1545]*y[IDX_OHI] - k[2290];
    data[1620] = 0.0 + k[1319]*y[IDX_N2I];
    data[1621] = 0.0 + k[953]*y[IDX_H2I] + k[1010]*y[IDX_H2OI] + k[1317]*y[IDX_HCOI];
    data[1622] = 0.0 + k[927]*y[IDX_N2I];
    data[1623] = 0.0 - k[1171]*y[IDX_N2HII];
    data[1624] = 0.0 - k[1322]*y[IDX_N2HII];
    data[1625] = 0.0 + k[1363]*y[IDX_N2I] + k[1367]*y[IDX_NOI];
    data[1626] = 0.0 - k[1408]*y[IDX_N2HII];
    data[1627] = 0.0 + k[1306]*y[IDX_NH3I];
    data[1628] = 0.0 + k[1338]*y[IDX_NI];
    data[1629] = 0.0 - k[682]*y[IDX_N2HII];
    data[1630] = 0.0 + k[1529]*y[IDX_N2I];
    data[1631] = 0.0 + k[927]*y[IDX_H2II] + k[1055]*y[IDX_H3II] + k[1319]*y[IDX_HNOII] +
        k[1320]*y[IDX_O2HII] + k[1363]*y[IDX_NHII] + k[1529]*y[IDX_OHII];
    data[1632] = 0.0 - k[655]*y[IDX_N2HII];
    data[1633] = 0.0 - k[765]*y[IDX_N2HII];
    data[1634] = 0.0 - k[1454]*y[IDX_N2HII];
    data[1635] = 0.0 - k[1323]*y[IDX_N2HII];
    data[1636] = 0.0 - k[817]*y[IDX_N2HII];
    data[1637] = 0.0 - k[1160]*y[IDX_N2HII] + k[1317]*y[IDX_N2II];
    data[1638] = 0.0 - k[1128]*y[IDX_N2HII];
    data[1639] = 0.0 + k[1367]*y[IDX_NHII];
    data[1640] = 0.0 - k[853]*y[IDX_N2HII];
    data[1641] = 0.0 + k[1306]*y[IDX_NII] - k[1440]*y[IDX_N2HII];
    data[1642] = 0.0 - k[1324]*y[IDX_N2HII];
    data[1643] = 0.0 - k[1545]*y[IDX_N2HII];
    data[1644] = 0.0 + k[1338]*y[IDX_NH2II];
    data[1645] = 0.0 + k[1055]*y[IDX_N2I];
    data[1646] = 0.0 - k[1506]*y[IDX_N2HII];
    data[1647] = 0.0 - k[698]*y[IDX_N2HII];
    data[1648] = 0.0 + k[1010]*y[IDX_N2II] - k[1011]*y[IDX_N2HII];
    data[1649] = 0.0 - k[877]*y[IDX_N2HII];
    data[1650] = 0.0 + k[953]*y[IDX_N2II];
    data[1651] = 0.0 - k[565]*y[IDX_N2HII] - k[566]*y[IDX_N2HII];
    data[1652] = 0.0 + k[2487] + k[2488] + k[2489] + k[2490];
    data[1653] = 0.0 + k[553]*y[IDX_EM] + k[1006]*y[IDX_H2OI];
    data[1654] = 0.0 + k[1895]*y[IDX_OI];
    data[1655] = 0.0 + k[190]*y[IDX_H2SI] + k[291]*y[IDX_NH3I];
    data[1656] = 0.0 + k[1700]*y[IDX_CHI];
    data[1657] = 0.0 - k[80]*y[IDX_OCSI] - k[802]*y[IDX_OCSI];
    data[1658] = 0.0 + k[1936]*y[IDX_OHI];
    data[1659] = 0.0 - k[1562]*y[IDX_OCSI];
    data[1660] = 0.0 - k[24]*y[IDX_CII] - k[80]*y[IDX_CH4II] - k[137]*y[IDX_HII] -
        k[184]*y[IDX_H2OII] - k[250]*y[IDX_NII] - k[257]*y[IDX_N2II] -
        k[318]*y[IDX_OII] - k[440] - k[441] - k[636]*y[IDX_CII] -
        k[736]*y[IDX_CHII] - k[737]*y[IDX_CHII] - k[751]*y[IDX_CH2II] -
        k[752]*y[IDX_CH2II] - k[785]*y[IDX_CH3II] - k[802]*y[IDX_CH4II] -
        k[904]*y[IDX_HII] - k[1065]*y[IDX_H3II] - k[1149]*y[IDX_HCOII] -
        k[1264]*y[IDX_HeII] - k[1265]*y[IDX_HeII] - k[1266]*y[IDX_HeII] -
        k[1267]*y[IDX_HeII] - k[1312]*y[IDX_NII] - k[1313]*y[IDX_NII] -
        k[1318]*y[IDX_N2II] - k[1479]*y[IDX_OII] - k[1552]*y[IDX_SII] -
        k[1562]*y[IDX_SOII] - k[1565]*y[IDX_SiII] - k[1610]*y[IDX_CI] -
        k[1695]*y[IDX_CHI] - k[1775]*y[IDX_HI] - k[1912]*y[IDX_OI] -
        k[1913]*y[IDX_OI] - k[2066] - k[2067] - k[2259];
    data[1661] = 0.0 - k[257]*y[IDX_OCSI] - k[1318]*y[IDX_OCSI];
    data[1662] = 0.0 - k[1565]*y[IDX_OCSI];
    data[1663] = 0.0 - k[250]*y[IDX_OCSI] - k[1312]*y[IDX_OCSI] - k[1313]*y[IDX_OCSI];
    data[1664] = 0.0 - k[318]*y[IDX_OCSI] - k[1479]*y[IDX_OCSI];
    data[1665] = 0.0 - k[751]*y[IDX_OCSI] - k[752]*y[IDX_OCSI];
    data[1666] = 0.0 + k[190]*y[IDX_OCSII];
    data[1667] = 0.0 - k[184]*y[IDX_OCSI];
    data[1668] = 0.0 - k[785]*y[IDX_OCSI];
    data[1669] = 0.0 - k[736]*y[IDX_OCSI] - k[737]*y[IDX_OCSI];
    data[1670] = 0.0 - k[1552]*y[IDX_OCSI];
    data[1671] = 0.0 + k[1951]*y[IDX_SI];
    data[1672] = 0.0 - k[1695]*y[IDX_OCSI] + k[1700]*y[IDX_SOI];
    data[1673] = 0.0 + k[291]*y[IDX_OCSII];
    data[1674] = 0.0 - k[24]*y[IDX_OCSI] - k[636]*y[IDX_OCSI];
    data[1675] = 0.0 + k[1951]*y[IDX_HCOI];
    data[1676] = 0.0 + k[1936]*y[IDX_CSI];
    data[1677] = 0.0 - k[1264]*y[IDX_OCSI] - k[1265]*y[IDX_OCSI] - k[1266]*y[IDX_OCSI] -
        k[1267]*y[IDX_OCSI];
    data[1678] = 0.0 - k[1065]*y[IDX_OCSI];
    data[1679] = 0.0 - k[137]*y[IDX_OCSI] - k[904]*y[IDX_OCSI];
    data[1680] = 0.0 + k[1895]*y[IDX_HCSI] - k[1912]*y[IDX_OCSI] - k[1913]*y[IDX_OCSI];
    data[1681] = 0.0 - k[1610]*y[IDX_OCSI];
    data[1682] = 0.0 - k[1149]*y[IDX_OCSI];
    data[1683] = 0.0 + k[1006]*y[IDX_HOCSII];
    data[1684] = 0.0 + k[553]*y[IDX_HOCSII];
    data[1685] = 0.0 - k[1775]*y[IDX_OCSI];
    data[1686] = 0.0 + k[190]*y[IDX_H2SI];
    data[1687] = 0.0 + k[1104]*y[IDX_HI];
    data[1688] = 0.0 + k[77]*y[IDX_H2SI];
    data[1689] = 0.0 - k[223]*y[IDX_H2SII];
    data[1690] = 0.0 - k[203]*y[IDX_HCOI] - k[223]*y[IDX_MgI] - k[286]*y[IDX_NH3I] -
        k[299]*y[IDX_NOI] - k[346]*y[IDX_SI] - k[350]*y[IDX_SiI] -
        k[515]*y[IDX_EM] - k[516]*y[IDX_EM] - k[691]*y[IDX_CI] -
        k[946]*y[IDX_H2I] - k[1000]*y[IDX_H2OI] - k[1017]*y[IDX_H2SI] -
        k[1103]*y[IDX_HI] - k[1335]*y[IDX_NI] - k[1426]*y[IDX_NH3I] -
        k[1498]*y[IDX_OI] - k[1499]*y[IDX_OI] - k[2138]*y[IDX_EM] - k[2293];
    data[1691] = 0.0 + k[253]*y[IDX_H2SI];
    data[1692] = 0.0 + k[1053]*y[IDX_H3II] + k[1147]*y[IDX_HCOII];
    data[1693] = 0.0 + k[163]*y[IDX_H2SI];
    data[1694] = 0.0 - k[350]*y[IDX_H2SII];
    data[1695] = 0.0 + k[102]*y[IDX_H2SI];
    data[1696] = 0.0 + k[949]*y[IDX_H2I];
    data[1697] = 0.0 + k[322]*y[IDX_H2SI];
    data[1698] = 0.0 + k[241]*y[IDX_H2SI];
    data[1699] = 0.0 + k[266]*y[IDX_H2SI];
    data[1700] = 0.0 + k[313]*y[IDX_H2SI];
    data[1701] = 0.0 + k[17]*y[IDX_CII] + k[43]*y[IDX_C2H2II] + k[77]*y[IDX_CH4II] +
        k[102]*y[IDX_COII] + k[123]*y[IDX_HII] + k[163]*y[IDX_H2II] +
        k[179]*y[IDX_H2OII] + k[190]*y[IDX_OCSII] + k[214]*y[IDX_HeII] +
        k[241]*y[IDX_NII] + k[253]*y[IDX_N2II] + k[266]*y[IDX_NH2II] +
        k[313]*y[IDX_OII] + k[322]*y[IDX_O2II] + k[333]*y[IDX_OHII] + k[403] -
        k[1017]*y[IDX_H2SII] + k[2020];
    data[1702] = 0.0 + k[179]*y[IDX_H2SI];
    data[1703] = 0.0 + k[333]*y[IDX_H2SI];
    data[1704] = 0.0 + k[43]*y[IDX_H2SI];
    data[1705] = 0.0 + k[974]*y[IDX_H2COI] + k[2117]*y[IDX_H2I];
    data[1706] = 0.0 + k[974]*y[IDX_SII];
    data[1707] = 0.0 - k[203]*y[IDX_H2SII];
    data[1708] = 0.0 - k[299]*y[IDX_H2SII];
    data[1709] = 0.0 - k[286]*y[IDX_H2SII] - k[1426]*y[IDX_H2SII];
    data[1710] = 0.0 + k[17]*y[IDX_H2SI];
    data[1711] = 0.0 - k[346]*y[IDX_H2SII];
    data[1712] = 0.0 - k[1335]*y[IDX_H2SII];
    data[1713] = 0.0 + k[214]*y[IDX_H2SI];
    data[1714] = 0.0 + k[1053]*y[IDX_HSI];
    data[1715] = 0.0 + k[123]*y[IDX_H2SI];
    data[1716] = 0.0 - k[1498]*y[IDX_H2SII] - k[1499]*y[IDX_H2SII];
    data[1717] = 0.0 - k[691]*y[IDX_H2SII];
    data[1718] = 0.0 + k[1147]*y[IDX_HSI];
    data[1719] = 0.0 - k[1000]*y[IDX_H2SII];
    data[1720] = 0.0 - k[946]*y[IDX_H2SII] + k[949]*y[IDX_HSII] + k[2117]*y[IDX_SII];
    data[1721] = 0.0 - k[515]*y[IDX_H2SII] - k[516]*y[IDX_H2SII] - k[2138]*y[IDX_H2SII];
    data[1722] = 0.0 - k[1103]*y[IDX_H2SII] + k[1104]*y[IDX_H3SII];
    data[1723] = 0.0 + k[1462]*y[IDX_NOI];
    data[1724] = 0.0 - k[300]*y[IDX_NOI] - k[549]*y[IDX_EM] - k[654]*y[IDX_C2I] -
        k[681]*y[IDX_C2HI] - k[696]*y[IDX_CI] - k[764]*y[IDX_CH2I] -
        k[813]*y[IDX_CH4I] - k[850]*y[IDX_CHI] - k[869]*y[IDX_CNI] -
        k[876]*y[IDX_COI] - k[971]*y[IDX_H2COI] - k[1005]*y[IDX_H2OI] -
        k[1125]*y[IDX_HCNI] - k[1159]*y[IDX_HCOI] - k[1169]*y[IDX_HNCI] -
        k[1173]*y[IDX_CO2I] - k[1174]*y[IDX_SI] - k[1319]*y[IDX_N2I] -
        k[1407]*y[IDX_NH2I] - k[1436]*y[IDX_NH3I] - k[1453]*y[IDX_NHI] -
        k[1544]*y[IDX_OHI] - k[2272];
    data[1725] = 0.0 + k[930]*y[IDX_NOI];
    data[1726] = 0.0 - k[1169]*y[IDX_HNOII];
    data[1727] = 0.0 - k[1173]*y[IDX_HNOII] + k[1351]*y[IDX_NHII];
    data[1728] = 0.0 + k[1351]*y[IDX_CO2I] + k[1357]*y[IDX_H2OI];
    data[1729] = 0.0 - k[1407]*y[IDX_HNOII];
    data[1730] = 0.0 + k[1458]*y[IDX_NHI];
    data[1731] = 0.0 + k[1391]*y[IDX_O2I] + k[1507]*y[IDX_OI];
    data[1732] = 0.0 - k[681]*y[IDX_HNOII];
    data[1733] = 0.0 + k[1508]*y[IDX_OI];
    data[1734] = 0.0 + k[1333]*y[IDX_NI];
    data[1735] = 0.0 + k[1531]*y[IDX_NOI];
    data[1736] = 0.0 - k[1319]*y[IDX_HNOII];
    data[1737] = 0.0 - k[654]*y[IDX_HNOII];
    data[1738] = 0.0 - k[764]*y[IDX_HNOII];
    data[1739] = 0.0 - k[1453]*y[IDX_HNOII] + k[1458]*y[IDX_O2II];
    data[1740] = 0.0 - k[869]*y[IDX_HNOII];
    data[1741] = 0.0 - k[971]*y[IDX_HNOII];
    data[1742] = 0.0 - k[813]*y[IDX_HNOII];
    data[1743] = 0.0 - k[1159]*y[IDX_HNOII];
    data[1744] = 0.0 - k[1125]*y[IDX_HNOII];
    data[1745] = 0.0 - k[300]*y[IDX_HNOII] + k[930]*y[IDX_H2II] + k[1060]*y[IDX_H3II] +
        k[1462]*y[IDX_O2HII] + k[1531]*y[IDX_OHII];
    data[1746] = 0.0 - k[850]*y[IDX_HNOII];
    data[1747] = 0.0 - k[1436]*y[IDX_HNOII];
    data[1748] = 0.0 + k[1391]*y[IDX_NH2II];
    data[1749] = 0.0 - k[1174]*y[IDX_HNOII];
    data[1750] = 0.0 - k[1544]*y[IDX_HNOII];
    data[1751] = 0.0 + k[1333]*y[IDX_H2OII];
    data[1752] = 0.0 + k[1060]*y[IDX_NOI];
    data[1753] = 0.0 + k[1507]*y[IDX_NH2II] + k[1508]*y[IDX_NH3II];
    data[1754] = 0.0 - k[696]*y[IDX_HNOII];
    data[1755] = 0.0 - k[1005]*y[IDX_HNOII] + k[1357]*y[IDX_NHII];
    data[1756] = 0.0 - k[876]*y[IDX_HNOII];
    data[1757] = 0.0 - k[549]*y[IDX_HNOII];
    data[1758] = 0.0 - k[832]*y[IDX_CH5II];
    data[1759] = 0.0 - k[836]*y[IDX_CH5II];
    data[1760] = 0.0 + k[812]*y[IDX_CH4I];
    data[1761] = 0.0 + k[795]*y[IDX_CH4I] + k[939]*y[IDX_H2I];
    data[1762] = 0.0 - k[834]*y[IDX_CH5II];
    data[1763] = 0.0 + k[817]*y[IDX_CH4I];
    data[1764] = 0.0 + k[813]*y[IDX_CH4I];
    data[1765] = 0.0 - k[493]*y[IDX_EM] - k[494]*y[IDX_EM] - k[495]*y[IDX_EM] -
        k[496]*y[IDX_EM] - k[497]*y[IDX_EM] - k[689]*y[IDX_CI] -
        k[755]*y[IDX_CH2I] - k[823]*y[IDX_C2I] - k[824]*y[IDX_C2HI] -
        k[825]*y[IDX_CO2I] - k[826]*y[IDX_COI] - k[827]*y[IDX_H2COI] -
        k[828]*y[IDX_H2OI] - k[829]*y[IDX_H2SI] - k[830]*y[IDX_HCNI] -
        k[831]*y[IDX_HCOI] - k[832]*y[IDX_HClI] - k[833]*y[IDX_HNCI] -
        k[834]*y[IDX_MgI] - k[835]*y[IDX_SI] - k[836]*y[IDX_SiH4I] -
        k[840]*y[IDX_CHI] - k[1102]*y[IDX_HI] - k[1397]*y[IDX_NH2I] -
        k[1422]*y[IDX_NH3I] - k[1447]*y[IDX_NHI] - k[1494]*y[IDX_OI] -
        k[1495]*y[IDX_OI] - k[1538]*y[IDX_OHI] - k[2248];
    data[1766] = 0.0 + k[915]*y[IDX_CH4I];
    data[1767] = 0.0 - k[833]*y[IDX_CH5II];
    data[1768] = 0.0 - k[825]*y[IDX_CH5II];
    data[1769] = 0.0 - k[1397]*y[IDX_CH5II];
    data[1770] = 0.0 - k[824]*y[IDX_CH5II];
    data[1771] = 0.0 - k[829]*y[IDX_CH5II];
    data[1772] = 0.0 + k[819]*y[IDX_CH4I];
    data[1773] = 0.0 + k[2114]*y[IDX_H2I];
    data[1774] = 0.0 - k[823]*y[IDX_CH5II];
    data[1775] = 0.0 - k[755]*y[IDX_CH5II];
    data[1776] = 0.0 - k[1447]*y[IDX_CH5II];
    data[1777] = 0.0 - k[827]*y[IDX_CH5II];
    data[1778] = 0.0 + k[795]*y[IDX_CH4II] + k[812]*y[IDX_HCO2II] + k[813]*y[IDX_HNOII] +
        k[817]*y[IDX_N2HII] + k[819]*y[IDX_OHII] + k[915]*y[IDX_H2II] +
        k[1033]*y[IDX_H3II];
    data[1779] = 0.0 - k[831]*y[IDX_CH5II];
    data[1780] = 0.0 - k[830]*y[IDX_CH5II];
    data[1781] = 0.0 - k[840]*y[IDX_CH5II];
    data[1782] = 0.0 - k[1422]*y[IDX_CH5II];
    data[1783] = 0.0 - k[835]*y[IDX_CH5II];
    data[1784] = 0.0 - k[1538]*y[IDX_CH5II];
    data[1785] = 0.0 + k[1033]*y[IDX_CH4I];
    data[1786] = 0.0 - k[1494]*y[IDX_CH5II] - k[1495]*y[IDX_CH5II];
    data[1787] = 0.0 - k[689]*y[IDX_CH5II];
    data[1788] = 0.0 - k[828]*y[IDX_CH5II];
    data[1789] = 0.0 - k[826]*y[IDX_CH5II];
    data[1790] = 0.0 + k[939]*y[IDX_CH4II] + k[2114]*y[IDX_CH3II];
    data[1791] = 0.0 - k[493]*y[IDX_CH5II] - k[494]*y[IDX_CH5II] - k[495]*y[IDX_CH5II] -
        k[496]*y[IDX_CH5II] - k[497]*y[IDX_CH5II];
    data[1792] = 0.0 - k[1102]*y[IDX_CH5II];
    data[1793] = 0.0 - k[226]*y[IDX_N2II];
    data[1794] = 0.0 + k[1332]*y[IDX_NI];
    data[1795] = 0.0 - k[257]*y[IDX_N2II] - k[1318]*y[IDX_N2II];
    data[1796] = 0.0 - k[38]*y[IDX_C2I] - k[49]*y[IDX_C2HI] - k[53]*y[IDX_CI] -
        k[67]*y[IDX_CH2I] - k[88]*y[IDX_CHI] - k[100]*y[IDX_CNI] -
        k[107]*y[IDX_COI] - k[189]*y[IDX_H2OI] - k[201]*y[IDX_HCNI] -
        k[226]*y[IDX_MgI] - k[252]*y[IDX_H2COI] - k[253]*y[IDX_H2SI] -
        k[254]*y[IDX_HCOI] - k[255]*y[IDX_NOI] - k[256]*y[IDX_O2I] -
        k[257]*y[IDX_OCSI] - k[258]*y[IDX_SI] - k[259]*y[IDX_NI] -
        k[275]*y[IDX_NH2I] - k[289]*y[IDX_NH3I] - k[296]*y[IDX_NHI] -
        k[328]*y[IDX_OI] - k[342]*y[IDX_OHI] - k[564]*y[IDX_EM] -
        k[815]*y[IDX_CH4I] - k[816]*y[IDX_CH4I] - k[953]*y[IDX_H2I] -
        k[1010]*y[IDX_H2OI] - k[1314]*y[IDX_H2COI] - k[1315]*y[IDX_H2SI] -
        k[1316]*y[IDX_H2SI] - k[1317]*y[IDX_HCOI] - k[1318]*y[IDX_OCSI] -
        k[1505]*y[IDX_OI] - k[2230];
    data[1797] = 0.0 + k[1337]*y[IDX_NI];
    data[1798] = 0.0 - k[275]*y[IDX_N2II];
    data[1799] = 0.0 + k[1308]*y[IDX_NHI] + k[1309]*y[IDX_NOI] + k[2127]*y[IDX_NI];
    data[1800] = 0.0 - k[49]*y[IDX_N2II];
    data[1801] = 0.0 - k[253]*y[IDX_N2II] - k[1315]*y[IDX_N2II] - k[1316]*y[IDX_N2II];
    data[1802] = 0.0 + k[215]*y[IDX_HeII];
    data[1803] = 0.0 - k[38]*y[IDX_N2II];
    data[1804] = 0.0 - k[67]*y[IDX_N2II];
    data[1805] = 0.0 - k[296]*y[IDX_N2II] + k[1308]*y[IDX_NII];
    data[1806] = 0.0 - k[100]*y[IDX_N2II];
    data[1807] = 0.0 - k[252]*y[IDX_N2II] - k[1314]*y[IDX_N2II];
    data[1808] = 0.0 - k[815]*y[IDX_N2II] - k[816]*y[IDX_N2II];
    data[1809] = 0.0 - k[254]*y[IDX_N2II] - k[1317]*y[IDX_N2II];
    data[1810] = 0.0 - k[201]*y[IDX_N2II];
    data[1811] = 0.0 - k[255]*y[IDX_N2II] + k[1309]*y[IDX_NII];
    data[1812] = 0.0 - k[88]*y[IDX_N2II];
    data[1813] = 0.0 - k[289]*y[IDX_N2II];
    data[1814] = 0.0 - k[256]*y[IDX_N2II];
    data[1815] = 0.0 - k[258]*y[IDX_N2II];
    data[1816] = 0.0 - k[342]*y[IDX_N2II];
    data[1817] = 0.0 - k[259]*y[IDX_N2II] + k[1332]*y[IDX_CNII] + k[1337]*y[IDX_NHII] +
        k[2127]*y[IDX_NII];
    data[1818] = 0.0 + k[215]*y[IDX_N2I];
    data[1819] = 0.0 - k[328]*y[IDX_N2II] - k[1505]*y[IDX_N2II];
    data[1820] = 0.0 - k[53]*y[IDX_N2II];
    data[1821] = 0.0 - k[189]*y[IDX_N2II] - k[1010]*y[IDX_N2II];
    data[1822] = 0.0 - k[107]*y[IDX_N2II];
    data[1823] = 0.0 - k[953]*y[IDX_N2II];
    data[1824] = 0.0 - k[564]*y[IDX_N2II];
    data[1825] = 0.0 + k[402] + k[402] + k[1224]*y[IDX_HeII] + k[2019] + k[2019];
    data[1826] = 0.0 + k[518]*y[IDX_EM] + k[518]*y[IDX_EM];
    data[1827] = 0.0 + k[417] + k[1247]*y[IDX_HeII] + k[2042];
    data[1828] = 0.0 + k[561]*y[IDX_EM];
    data[1829] = 0.0 + k[1109]*y[IDX_HI];
    data[1830] = 0.0 + k[555]*y[IDX_EM];
    data[1831] = 0.0 + k[1894]*y[IDX_OI];
    data[1832] = 0.0 + k[1777]*y[IDX_HI];
    data[1833] = 0.0 + k[1766]*y[IDX_HI];
    data[1834] = 0.0 + k[533]*y[IDX_EM] + k[534]*y[IDX_EM];
    data[1835] = 0.0 + k[1699]*y[IDX_CHI] + k[1778]*y[IDX_HI];
    data[1836] = 0.0 + k[1935]*y[IDX_OHI];
    data[1837] = 0.0 + k[225]*y[IDX_HSII];
    data[1838] = 0.0 + k[1775]*y[IDX_HI];
    data[1839] = 0.0 + k[515]*y[IDX_EM] + k[1000]*y[IDX_H2OI] + k[1017]*y[IDX_H2SI] +
        k[1426]*y[IDX_NH3I];
    data[1840] = 0.0 - k[128]*y[IDX_HII] - k[418] - k[628]*y[IDX_CII] -
        k[780]*y[IDX_CH3II] - k[902]*y[IDX_HII] - k[1053]*y[IDX_H3II] -
        k[1147]*y[IDX_HCOII] - k[1249]*y[IDX_HeII] - k[1595]*y[IDX_CI] -
        k[1596]*y[IDX_CI] - k[1727]*y[IDX_H2I] - k[1758]*y[IDX_HI] -
        k[1789]*y[IDX_HSI] - k[1789]*y[IDX_HSI] - k[1789]*y[IDX_HSI] -
        k[1789]*y[IDX_HSI] - k[1816]*y[IDX_NI] - k[1817]*y[IDX_NI] -
        k[1899]*y[IDX_OI] - k[1900]*y[IDX_OI] - k[1953]*y[IDX_SI] - k[2043] -
        k[2254];
    data[1841] = 0.0 + k[351]*y[IDX_HSII];
    data[1842] = 0.0 + k[872]*y[IDX_H2SI];
    data[1843] = 0.0 + k[225]*y[IDX_MgI] + k[288]*y[IDX_NH3I] + k[301]*y[IDX_NOI] +
        k[347]*y[IDX_SI] + k[351]*y[IDX_SiI];
    data[1844] = 0.0 + k[1301]*y[IDX_H2SI];
    data[1845] = 0.0 + k[1383]*y[IDX_H2SI];
    data[1846] = 0.0 + k[1415]*y[IDX_H2SI];
    data[1847] = 0.0 + k[872]*y[IDX_COII] + k[982]*y[IDX_H2OII] + k[1017]*y[IDX_H2SII] +
        k[1301]*y[IDX_NII] + k[1383]*y[IDX_NH2II] + k[1415]*y[IDX_NH3II] +
        k[1653]*y[IDX_CH3I] + k[1749]*y[IDX_HI] + k[1888]*y[IDX_OI] +
        k[1938]*y[IDX_OHI] + k[2021];
    data[1848] = 0.0 + k[982]*y[IDX_H2SI];
    data[1849] = 0.0 - k[780]*y[IDX_HSI];
    data[1850] = 0.0 + k[1653]*y[IDX_H2SI];
    data[1851] = 0.0 + k[1856]*y[IDX_SI];
    data[1852] = 0.0 + k[975]*y[IDX_H2COI];
    data[1853] = 0.0 + k[975]*y[IDX_SII];
    data[1854] = 0.0 + k[1673]*y[IDX_SI];
    data[1855] = 0.0 + k[1952]*y[IDX_SI];
    data[1856] = 0.0 + k[301]*y[IDX_HSII];
    data[1857] = 0.0 + k[1698]*y[IDX_SI] + k[1699]*y[IDX_SOI];
    data[1858] = 0.0 + k[288]*y[IDX_HSII] + k[1426]*y[IDX_H2SII];
    data[1859] = 0.0 - k[628]*y[IDX_HSI];
    data[1860] = 0.0 + k[347]*y[IDX_HSII] + k[1673]*y[IDX_CH4I] + k[1698]*y[IDX_CHI] +
        k[1735]*y[IDX_H2I] + k[1856]*y[IDX_NHI] + k[1952]*y[IDX_HCOI] -
        k[1953]*y[IDX_HSI];
    data[1861] = 0.0 + k[1935]*y[IDX_CSI] + k[1938]*y[IDX_H2SI];
    data[1862] = 0.0 - k[1816]*y[IDX_HSI] - k[1817]*y[IDX_HSI];
    data[1863] = 0.0 + k[1224]*y[IDX_H2S2I] + k[1247]*y[IDX_HS2I] - k[1249]*y[IDX_HSI];
    data[1864] = 0.0 - k[1053]*y[IDX_HSI];
    data[1865] = 0.0 - k[128]*y[IDX_HSI] - k[902]*y[IDX_HSI];
    data[1866] = 0.0 + k[1888]*y[IDX_H2SI] + k[1894]*y[IDX_HCSI] - k[1899]*y[IDX_HSI] -
        k[1900]*y[IDX_HSI];
    data[1867] = 0.0 - k[1595]*y[IDX_HSI] - k[1596]*y[IDX_HSI];
    data[1868] = 0.0 - k[1147]*y[IDX_HSI];
    data[1869] = 0.0 + k[1000]*y[IDX_H2SII];
    data[1870] = 0.0 - k[1727]*y[IDX_HSI] + k[1735]*y[IDX_SI];
    data[1871] = 0.0 + k[515]*y[IDX_H2SII] + k[518]*y[IDX_H2S2II] + k[518]*y[IDX_H2S2II]
        + k[533]*y[IDX_H3SII] + k[534]*y[IDX_H3SII] + k[555]*y[IDX_HS2II] +
        k[561]*y[IDX_HSiSII];
    data[1872] = 0.0 + k[1109]*y[IDX_SiSII] + k[1749]*y[IDX_H2SI] - k[1758]*y[IDX_HSI] +
        k[1766]*y[IDX_NSI] + k[1775]*y[IDX_OCSI] + k[1777]*y[IDX_S2I] +
        k[1778]*y[IDX_SOI];
    data[1873] = 0.0 + k[1127]*y[IDX_HCNI];
    data[1874] = 0.0 + k[775]*y[IDX_CH3II] - k[1130]*y[IDX_HCNHII] -
        k[1131]*y[IDX_HCNHII];
    data[1875] = 0.0 + k[1421]*y[IDX_NH3I];
    data[1876] = 0.0 + k[1123]*y[IDX_HCNI] + k[1167]*y[IDX_HNCI];
    data[1877] = 0.0 + k[1129]*y[IDX_HCNI] + k[1172]*y[IDX_HNCI];
    data[1878] = 0.0 + k[1128]*y[IDX_HCNI] + k[1171]*y[IDX_HNCI];
    data[1879] = 0.0 + k[1125]*y[IDX_HCNI] + k[1169]*y[IDX_HNCI];
    data[1880] = 0.0 + k[830]*y[IDX_HCNI] + k[833]*y[IDX_HNCI];
    data[1881] = 0.0 - k[539]*y[IDX_EM] - k[540]*y[IDX_EM] - k[541]*y[IDX_EM] -
        k[761]*y[IDX_CH2I] - k[762]*y[IDX_CH2I] - k[847]*y[IDX_CHI] -
        k[848]*y[IDX_CHI] - k[1130]*y[IDX_CH3CNI] - k[1131]*y[IDX_CH3CNI] -
        k[1132]*y[IDX_H2COI] - k[1133]*y[IDX_H2COI] - k[1134]*y[IDX_H2SI] -
        k[1135]*y[IDX_H2SI] - k[1404]*y[IDX_NH2I] - k[1405]*y[IDX_NH2I] -
        k[1431]*y[IDX_NH3I] - k[1432]*y[IDX_NH3I] - k[2289];
    data[1882] = 0.0 + k[1122]*y[IDX_HCNI] + k[1166]*y[IDX_HNCI];
    data[1883] = 0.0 + k[811]*y[IDX_CH4I] + k[947]*y[IDX_H2I] + k[1113]*y[IDX_HCNI] +
        k[1115]*y[IDX_HCOI] + k[1116]*y[IDX_HNCI] + k[1430]*y[IDX_NH3I];
    data[1884] = 0.0 + k[664]*y[IDX_C2HII] + k[669]*y[IDX_C2H2II] + k[727]*y[IDX_CHII] +
        k[833]*y[IDX_CH5II] + k[986]*y[IDX_H2OII] + k[1050]*y[IDX_H3II] +
        k[1090]*y[IDX_H3OII] + k[1116]*y[IDX_HCNII] + k[1165]*y[IDX_H2COII] +
        k[1166]*y[IDX_H3COII] + k[1167]*y[IDX_H3SII] + k[1168]*y[IDX_HCOII] +
        k[1169]*y[IDX_HNOII] + k[1170]*y[IDX_HSII] + k[1171]*y[IDX_N2HII] +
        k[1172]*y[IDX_O2HII] + k[1362]*y[IDX_NHII] + k[1387]*y[IDX_NH2II] +
        k[1528]*y[IDX_OHII];
    data[1885] = 0.0 + k[1126]*y[IDX_HCNI] + k[1170]*y[IDX_HNCI];
    data[1886] = 0.0 + k[1360]*y[IDX_HCNI] + k[1362]*y[IDX_HNCI];
    data[1887] = 0.0 - k[1404]*y[IDX_HCNHII] - k[1405]*y[IDX_HCNHII];
    data[1888] = 0.0 + k[1295]*y[IDX_CH4I];
    data[1889] = 0.0 + k[1385]*y[IDX_HCNI] + k[1387]*y[IDX_HNCI];
    data[1890] = 0.0 + k[662]*y[IDX_HCNI] + k[664]*y[IDX_HNCI];
    data[1891] = 0.0 + k[1121]*y[IDX_HCNI] + k[1165]*y[IDX_HNCI];
    data[1892] = 0.0 - k[1134]*y[IDX_HCNHII] - k[1135]*y[IDX_HCNHII];
    data[1893] = 0.0 + k[983]*y[IDX_HCNI] + k[986]*y[IDX_HNCI];
    data[1894] = 0.0 + k[1525]*y[IDX_HCNI] + k[1528]*y[IDX_HNCI];
    data[1895] = 0.0 + k[775]*y[IDX_CH3CNI] + k[1446]*y[IDX_NHI];
    data[1896] = 0.0 + k[668]*y[IDX_HCNI] + k[669]*y[IDX_HNCI];
    data[1897] = 0.0 - k[761]*y[IDX_HCNHII] - k[762]*y[IDX_HCNHII];
    data[1898] = 0.0 + k[725]*y[IDX_HCNI] + k[727]*y[IDX_HNCI];
    data[1899] = 0.0 + k[1446]*y[IDX_CH3II];
    data[1900] = 0.0 - k[1132]*y[IDX_HCNHII] - k[1133]*y[IDX_HCNHII];
    data[1901] = 0.0 + k[811]*y[IDX_HCNII] + k[1295]*y[IDX_NII];
    data[1902] = 0.0 + k[1115]*y[IDX_HCNII];
    data[1903] = 0.0 + k[662]*y[IDX_C2HII] + k[668]*y[IDX_C2H2II] + k[725]*y[IDX_CHII] +
        k[830]*y[IDX_CH5II] + k[983]*y[IDX_H2OII] + k[1045]*y[IDX_H3II] +
        k[1088]*y[IDX_H3OII] + k[1113]*y[IDX_HCNII] + k[1121]*y[IDX_H2COII] +
        k[1122]*y[IDX_H3COII] + k[1123]*y[IDX_H3SII] + k[1124]*y[IDX_HCOII] +
        k[1125]*y[IDX_HNOII] + k[1126]*y[IDX_HSII] + k[1127]*y[IDX_HSiSII] +
        k[1128]*y[IDX_N2HII] + k[1129]*y[IDX_O2HII] + k[1360]*y[IDX_NHII] +
        k[1385]*y[IDX_NH2II] + k[1525]*y[IDX_OHII];
    data[1904] = 0.0 - k[847]*y[IDX_HCNHII] - k[848]*y[IDX_HCNHII];
    data[1905] = 0.0 + k[1421]*y[IDX_C2NII] + k[1430]*y[IDX_HCNII] -
        k[1431]*y[IDX_HCNHII] - k[1432]*y[IDX_HCNHII];
    data[1906] = 0.0 + k[1088]*y[IDX_HCNI] + k[1090]*y[IDX_HNCI];
    data[1907] = 0.0 + k[1045]*y[IDX_HCNI] + k[1050]*y[IDX_HNCI];
    data[1908] = 0.0 + k[1124]*y[IDX_HCNI] + k[1168]*y[IDX_HNCI];
    data[1909] = 0.0 + k[947]*y[IDX_HCNII];
    data[1910] = 0.0 - k[539]*y[IDX_HCNHII] - k[540]*y[IDX_HCNHII] - k[541]*y[IDX_HCNHII];
    data[1911] = 0.0 + k[1106]*y[IDX_HI];
    data[1912] = 0.0 - k[910]*y[IDX_H2II];
    data[1913] = 0.0 - k[153]*y[IDX_C2I] - k[154]*y[IDX_C2H2I] - k[155]*y[IDX_C2HI] -
        k[156]*y[IDX_CH2I] - k[157]*y[IDX_CH4I] - k[158]*y[IDX_CHI] -
        k[159]*y[IDX_CNI] - k[160]*y[IDX_COI] - k[161]*y[IDX_H2COI] -
        k[162]*y[IDX_H2OI] - k[163]*y[IDX_H2SI] - k[164]*y[IDX_HCNI] -
        k[165]*y[IDX_HCOI] - k[166]*y[IDX_NH2I] - k[167]*y[IDX_NH3I] -
        k[168]*y[IDX_NHI] - k[169]*y[IDX_NOI] - k[170]*y[IDX_O2I] -
        k[171]*y[IDX_OHI] - k[193]*y[IDX_HI] - k[501]*y[IDX_EM] -
        k[909]*y[IDX_C2I] - k[910]*y[IDX_C2H4I] - k[911]*y[IDX_C2HI] -
        k[912]*y[IDX_CI] - k[913]*y[IDX_CH2I] - k[914]*y[IDX_CH4I] -
        k[915]*y[IDX_CH4I] - k[916]*y[IDX_CHI] - k[917]*y[IDX_CNI] -
        k[918]*y[IDX_CO2I] - k[919]*y[IDX_COI] - k[920]*y[IDX_H2I] -
        k[921]*y[IDX_H2COI] - k[922]*y[IDX_H2OI] - k[923]*y[IDX_H2SI] -
        k[924]*y[IDX_H2SI] - k[925]*y[IDX_HCOI] - k[926]*y[IDX_HeI] -
        k[927]*y[IDX_N2I] - k[928]*y[IDX_NI] - k[929]*y[IDX_NHI] -
        k[930]*y[IDX_NOI] - k[931]*y[IDX_O2I] - k[932]*y[IDX_OI] -
        k[933]*y[IDX_OHI] - k[2009];
    data[1914] = 0.0 - k[154]*y[IDX_H2II];
    data[1915] = 0.0 - k[918]*y[IDX_H2II];
    data[1916] = 0.0 - k[166]*y[IDX_H2II];
    data[1917] = 0.0 - k[155]*y[IDX_H2II] - k[911]*y[IDX_H2II];
    data[1918] = 0.0 - k[163]*y[IDX_H2II] - k[923]*y[IDX_H2II] - k[924]*y[IDX_H2II];
    data[1919] = 0.0 - k[927]*y[IDX_H2II];
    data[1920] = 0.0 - k[153]*y[IDX_H2II] - k[909]*y[IDX_H2II];
    data[1921] = 0.0 - k[156]*y[IDX_H2II] - k[913]*y[IDX_H2II];
    data[1922] = 0.0 - k[168]*y[IDX_H2II] - k[929]*y[IDX_H2II];
    data[1923] = 0.0 - k[159]*y[IDX_H2II] - k[917]*y[IDX_H2II];
    data[1924] = 0.0 - k[161]*y[IDX_H2II] - k[921]*y[IDX_H2II];
    data[1925] = 0.0 - k[157]*y[IDX_H2II] - k[914]*y[IDX_H2II] - k[915]*y[IDX_H2II];
    data[1926] = 0.0 - k[165]*y[IDX_H2II] + k[898]*y[IDX_HII] - k[925]*y[IDX_H2II];
    data[1927] = 0.0 - k[164]*y[IDX_H2II];
    data[1928] = 0.0 - k[169]*y[IDX_H2II] - k[930]*y[IDX_H2II];
    data[1929] = 0.0 - k[158]*y[IDX_H2II] - k[916]*y[IDX_H2II];
    data[1930] = 0.0 - k[167]*y[IDX_H2II];
    data[1931] = 0.0 - k[170]*y[IDX_H2II] - k[931]*y[IDX_H2II];
    data[1932] = 0.0 - k[171]*y[IDX_H2II] - k[933]*y[IDX_H2II];
    data[1933] = 0.0 - k[928]*y[IDX_H2II];
    data[1934] = 0.0 + k[172]*y[IDX_H2I];
    data[1935] = 0.0 - k[926]*y[IDX_H2II];
    data[1936] = 0.0 + k[2025];
    data[1937] = 0.0 + k[898]*y[IDX_HCOI] + k[2110]*y[IDX_HI];
    data[1938] = 0.0 - k[932]*y[IDX_H2II];
    data[1939] = 0.0 - k[912]*y[IDX_H2II];
    data[1940] = 0.0 - k[162]*y[IDX_H2II] - k[922]*y[IDX_H2II];
    data[1941] = 0.0 - k[160]*y[IDX_H2II] - k[919]*y[IDX_H2II];
    data[1942] = 0.0 + k[172]*y[IDX_HeII] + k[360] - k[920]*y[IDX_H2II];
    data[1943] = 0.0 - k[501]*y[IDX_H2II];
    data[1944] = 0.0 - k[193]*y[IDX_H2II] + k[1106]*y[IDX_HeHII] + k[2110]*y[IDX_HII];
    data[1945] = 0.0 + k[606]*y[IDX_CII] + k[1024]*y[IDX_H3II];
    data[1946] = 0.0 + k[612]*y[IDX_CII] + k[710]*y[IDX_CHII] + k[776]*y[IDX_CH3II] +
        k[888]*y[IDX_HII] - k[1078]*y[IDX_H3COII] + k[1290]*y[IDX_NII] +
        k[1467]*y[IDX_OII] + k[1483]*y[IDX_O2II] + k[1991];
    data[1947] = 0.0 + k[970]*y[IDX_H2COI];
    data[1948] = 0.0 + k[1560]*y[IDX_SOII];
    data[1949] = 0.0 + k[798]*y[IDX_H2COI];
    data[1950] = 0.0 + k[1560]*y[IDX_C2H4I];
    data[1951] = 0.0 + k[973]*y[IDX_H2COI];
    data[1952] = 0.0 + k[1323]*y[IDX_H2COI];
    data[1953] = 0.0 + k[971]*y[IDX_H2COI];
    data[1954] = 0.0 + k[827]*y[IDX_H2COI] + k[1494]*y[IDX_OI];
    data[1955] = 0.0 + k[1132]*y[IDX_H2COI] + k[1133]*y[IDX_H2COI];
    data[1956] = 0.0 - k[521]*y[IDX_EM] - k[522]*y[IDX_EM] - k[523]*y[IDX_EM] -
        k[524]*y[IDX_EM] - k[525]*y[IDX_EM] - k[844]*y[IDX_CHI] -
        k[1001]*y[IDX_H2OI] - k[1078]*y[IDX_CH3OHI] - k[1079]*y[IDX_H2SI] -
        k[1122]*y[IDX_HCNI] - k[1166]*y[IDX_HNCI] - k[1401]*y[IDX_NH2I] -
        k[1427]*y[IDX_NH3I] - k[2202];
    data[1957] = 0.0 + k[1112]*y[IDX_H2COI];
    data[1958] = 0.0 - k[1166]*y[IDX_H3COII];
    data[1959] = 0.0 + k[1354]*y[IDX_H2COI];
    data[1960] = 0.0 - k[1401]*y[IDX_H3COII];
    data[1961] = 0.0 + k[1483]*y[IDX_CH3OHI];
    data[1962] = 0.0 + k[1290]*y[IDX_CH3OHI];
    data[1963] = 0.0 + k[1376]*y[IDX_H2COI];
    data[1964] = 0.0 + k[660]*y[IDX_H2COI];
    data[1965] = 0.0 + k[1467]*y[IDX_CH3OHI];
    data[1966] = 0.0 + k[809]*y[IDX_CH4I] + k[966]*y[IDX_H2COI] + k[1158]*y[IDX_HCOI] +
        k[1449]*y[IDX_NHI];
    data[1967] = 0.0 + k[743]*y[IDX_H2OI];
    data[1968] = 0.0 - k[1079]*y[IDX_H3COII];
    data[1969] = 0.0 + k[979]*y[IDX_H2COI];
    data[1970] = 0.0 + k[1522]*y[IDX_H2COI];
    data[1971] = 0.0 + k[776]*y[IDX_CH3OHI] + k[782]*y[IDX_O2I];
    data[1972] = 0.0 + k[710]*y[IDX_CH3OHI] + k[716]*y[IDX_H2COI];
    data[1973] = 0.0 + k[1449]*y[IDX_H2COII];
    data[1974] = 0.0 + k[660]*y[IDX_C2HII] + k[716]*y[IDX_CHII] + k[798]*y[IDX_CH4II] +
        k[827]*y[IDX_CH5II] + k[966]*y[IDX_H2COII] + k[970]*y[IDX_H3SII] +
        k[971]*y[IDX_HNOII] + k[973]*y[IDX_O2HII] + k[979]*y[IDX_H2OII] +
        k[1041]*y[IDX_H3II] + k[1086]*y[IDX_H3OII] + k[1112]*y[IDX_HCNII] +
        k[1132]*y[IDX_HCNHII] + k[1133]*y[IDX_HCNHII] + k[1141]*y[IDX_HCOII] +
        k[1323]*y[IDX_N2HII] + k[1354]*y[IDX_NHII] + k[1376]*y[IDX_NH2II] +
        k[1522]*y[IDX_OHII];
    data[1975] = 0.0 + k[809]*y[IDX_H2COII];
    data[1976] = 0.0 + k[1158]*y[IDX_H2COII];
    data[1977] = 0.0 - k[1122]*y[IDX_H3COII];
    data[1978] = 0.0 - k[844]*y[IDX_H3COII];
    data[1979] = 0.0 - k[1427]*y[IDX_H3COII];
    data[1980] = 0.0 + k[1086]*y[IDX_H2COI];
    data[1981] = 0.0 + k[782]*y[IDX_CH3II];
    data[1982] = 0.0 + k[606]*y[IDX_C2H5OHI] + k[612]*y[IDX_CH3OHI];
    data[1983] = 0.0 + k[1024]*y[IDX_C2H5OHI] + k[1041]*y[IDX_H2COI];
    data[1984] = 0.0 + k[888]*y[IDX_CH3OHI];
    data[1985] = 0.0 + k[1494]*y[IDX_CH5II];
    data[1986] = 0.0 + k[1141]*y[IDX_H2COI];
    data[1987] = 0.0 + k[743]*y[IDX_CH2II] - k[1001]*y[IDX_H3COII];
    data[1988] = 0.0 - k[521]*y[IDX_H3COII] - k[522]*y[IDX_H3COII] - k[523]*y[IDX_H3COII]
        - k[524]*y[IDX_H3COII] - k[525]*y[IDX_H3COII];
    data[1989] = 0.0 + k[588]*y[IDX_EM];
    data[1990] = 0.0 + k[2078];
    data[1991] = 0.0 + k[561]*y[IDX_EM];
    data[1992] = 0.0 + k[605]*y[IDX_EM];
    data[1993] = 0.0 + k[587]*y[IDX_EM];
    data[1994] = 0.0 + k[451] + k[1277]*y[IDX_HeII] + k[1832]*y[IDX_NI] +
        k[1920]*y[IDX_OI] + k[2081];
    data[1995] = 0.0 + k[457] + k[1287]*y[IDX_HeII] + k[2095];
    data[1996] = 0.0 + k[593]*y[IDX_EM] + k[594]*y[IDX_EM];
    data[1997] = 0.0 + k[455] + k[2091];
    data[1998] = 0.0 + k[602]*y[IDX_EM] + k[864]*y[IDX_CHI] + k[1343]*y[IDX_NI];
    data[1999] = 0.0 + k[603]*y[IDX_EM];
    data[2000] = 0.0 + k[592]*y[IDX_EM] + k[863]*y[IDX_CHI] + k[1014]*y[IDX_H2OI] +
        k[1442]*y[IDX_NH3I];
    data[2001] = 0.0 - k[348]*y[IDX_SiI];
    data[2002] = 0.0 + k[456] + k[1286]*y[IDX_HeII] + k[2093];
    data[2003] = 0.0 + k[231]*y[IDX_SiII];
    data[2004] = 0.0 - k[350]*y[IDX_SiI];
    data[2005] = 0.0 - k[26]*y[IDX_CII] - k[60]*y[IDX_CHII] - k[143]*y[IDX_HII] -
        k[186]*y[IDX_H2OII] - k[219]*y[IDX_HeII] - k[281]*y[IDX_NH3II] -
        k[348]*y[IDX_CSII] - k[349]*y[IDX_H2COII] - k[350]*y[IDX_H2SII] -
        k[351]*y[IDX_HSII] - k[352]*y[IDX_NOII] - k[353]*y[IDX_O2II] -
        k[354]*y[IDX_SII] - k[448] - k[670]*y[IDX_C2H2II] - k[1071]*y[IDX_H3II]
        - k[1093]*y[IDX_H3OII] - k[1535]*y[IDX_OHII] - k[1566]*y[IDX_HCOII] -
        k[1575]*y[IDX_C2H2I] - k[1950]*y[IDX_OHI] - k[1956]*y[IDX_CO2I] -
        k[1957]*y[IDX_COI] - k[1958]*y[IDX_NOI] - k[1959]*y[IDX_O2I] - k[2077] -
        k[2131]*y[IDX_OI] - k[2170];
    data[2006] = 0.0 - k[1575]*y[IDX_SiI];
    data[2007] = 0.0 + k[231]*y[IDX_MgI] + k[2144]*y[IDX_EM];
    data[2008] = 0.0 - k[351]*y[IDX_SiI];
    data[2009] = 0.0 - k[1956]*y[IDX_SiI];
    data[2010] = 0.0 - k[353]*y[IDX_SiI];
    data[2011] = 0.0 - k[352]*y[IDX_SiI];
    data[2012] = 0.0 - k[349]*y[IDX_SiI];
    data[2013] = 0.0 - k[281]*y[IDX_SiI];
    data[2014] = 0.0 - k[186]*y[IDX_SiI];
    data[2015] = 0.0 - k[1535]*y[IDX_SiI];
    data[2016] = 0.0 - k[670]*y[IDX_SiI];
    data[2017] = 0.0 - k[60]*y[IDX_SiI];
    data[2018] = 0.0 - k[354]*y[IDX_SiI];
    data[2019] = 0.0 - k[1958]*y[IDX_SiI];
    data[2020] = 0.0 + k[863]*y[IDX_SiHII] + k[864]*y[IDX_SiOII];
    data[2021] = 0.0 + k[1442]*y[IDX_SiHII];
    data[2022] = 0.0 - k[1093]*y[IDX_SiI];
    data[2023] = 0.0 - k[1959]*y[IDX_SiI];
    data[2024] = 0.0 - k[26]*y[IDX_SiI];
    data[2025] = 0.0 - k[1950]*y[IDX_SiI];
    data[2026] = 0.0 + k[1343]*y[IDX_SiOII] + k[1832]*y[IDX_SiCI];
    data[2027] = 0.0 - k[219]*y[IDX_SiI] + k[1277]*y[IDX_SiCI] + k[1286]*y[IDX_SiOI] +
        k[1287]*y[IDX_SiSI];
    data[2028] = 0.0 - k[1071]*y[IDX_SiI];
    data[2029] = 0.0 - k[143]*y[IDX_SiI];
    data[2030] = 0.0 + k[1920]*y[IDX_SiCI] - k[2131]*y[IDX_SiI];
    data[2031] = 0.0 - k[1566]*y[IDX_SiI];
    data[2032] = 0.0 + k[1014]*y[IDX_SiHII];
    data[2033] = 0.0 - k[1957]*y[IDX_SiI];
    data[2034] = 0.0 + k[561]*y[IDX_HSiSII] + k[587]*y[IDX_SiCII] + k[588]*y[IDX_SiC2II]
        + k[592]*y[IDX_SiHII] + k[593]*y[IDX_SiH2II] + k[594]*y[IDX_SiH2II] +
        k[602]*y[IDX_SiOII] + k[603]*y[IDX_SiOHII] + k[605]*y[IDX_SiSII] +
        k[2144]*y[IDX_SiII];
    data[2035] = 0.0 + k[2327] + k[2328] + k[2329] + k[2330];
    data[2036] = 0.0 - k[675]*y[IDX_C2H2I];
    data[2037] = 0.0 + k[611]*y[IDX_CII];
    data[2038] = 0.0 + k[369] + k[1577]*y[IDX_O2I] + k[1646]*y[IDX_CH3I] +
        k[1738]*y[IDX_HI] + k[1791]*y[IDX_NI] + k[1930]*y[IDX_OHI] + k[1966];
    data[2039] = 0.0 + k[370] + k[1967];
    data[2040] = 0.0 - k[75]*y[IDX_C2H2I];
    data[2041] = 0.0 + k[220]*y[IDX_C2H2II];
    data[2042] = 0.0 - k[1556]*y[IDX_C2H2I] - k[1557]*y[IDX_C2H2I] - k[1558]*y[IDX_C2H2I];
    data[2043] = 0.0 - k[154]*y[IDX_C2H2I];
    data[2044] = 0.0 - k[1575]*y[IDX_C2H2I];
    data[2045] = 0.0 - k[46]*y[IDX_HCNII] - k[75]*y[IDX_CH4II] - k[111]*y[IDX_HII] -
        k[154]*y[IDX_H2II] - k[176]*y[IDX_H2OII] - k[208]*y[IDX_HeII] -
        k[307]*y[IDX_OII] - k[321]*y[IDX_O2II] - k[367] - k[368] -
        k[675]*y[IDX_C2N2II] - k[1178]*y[IDX_HeII] - k[1179]*y[IDX_HeII] -
        k[1180]*y[IDX_HeII] - k[1482]*y[IDX_O2II] - k[1556]*y[IDX_SOII] -
        k[1557]*y[IDX_SOII] - k[1558]*y[IDX_SOII] - k[1570]*y[IDX_C2I] -
        k[1574]*y[IDX_NOI] - k[1575]*y[IDX_SiI] - k[1674]*y[IDX_CHI] -
        k[1701]*y[IDX_CNI] - k[1737]*y[IDX_HI] - k[1868]*y[IDX_OI] -
        k[1927]*y[IDX_OHI] - k[1928]*y[IDX_OHI] - k[1929]*y[IDX_OHI] - k[1964] -
        k[1965] - k[2298];
    data[2046] = 0.0 - k[46]*y[IDX_C2H2I];
    data[2047] = 0.0 - k[321]*y[IDX_C2H2I] - k[1482]*y[IDX_C2H2I];
    data[2048] = 0.0 + k[1721]*y[IDX_H2I];
    data[2049] = 0.0 - k[307]*y[IDX_C2H2I];
    data[2050] = 0.0 + k[43]*y[IDX_C2H2II];
    data[2051] = 0.0 - k[176]*y[IDX_C2H2I];
    data[2052] = 0.0 + k[42]*y[IDX_H2COI] + k[43]*y[IDX_H2SI] + k[44]*y[IDX_HCOI] +
        k[45]*y[IDX_NOI] + k[220]*y[IDX_MgI] + k[282]*y[IDX_NH3I];
    data[2053] = 0.0 - k[1570]*y[IDX_C2H2I];
    data[2054] = 0.0 + k[1618]*y[IDX_CH2I] + k[1618]*y[IDX_CH2I] + k[1619]*y[IDX_CH2I] +
        k[1619]*y[IDX_CH2I];
    data[2055] = 0.0 + k[1588]*y[IDX_CI] + k[1646]*y[IDX_C2H3I];
    data[2056] = 0.0 - k[1701]*y[IDX_C2H2I];
    data[2057] = 0.0 + k[42]*y[IDX_C2H2II];
    data[2058] = 0.0 + k[44]*y[IDX_C2H2II];
    data[2059] = 0.0 + k[45]*y[IDX_C2H2II] - k[1574]*y[IDX_C2H2I];
    data[2060] = 0.0 - k[1674]*y[IDX_C2H2I];
    data[2061] = 0.0 + k[282]*y[IDX_C2H2II];
    data[2062] = 0.0 + k[1577]*y[IDX_C2H3I];
    data[2063] = 0.0 + k[611]*y[IDX_CH3CCHI];
    data[2064] = 0.0 - k[1927]*y[IDX_C2H2I] - k[1928]*y[IDX_C2H2I] - k[1929]*y[IDX_C2H2I]
        + k[1930]*y[IDX_C2H3I];
    data[2065] = 0.0 + k[1791]*y[IDX_C2H3I];
    data[2066] = 0.0 - k[208]*y[IDX_C2H2I] - k[1178]*y[IDX_C2H2I] - k[1179]*y[IDX_C2H2I]
        - k[1180]*y[IDX_C2H2I];
    data[2067] = 0.0 - k[111]*y[IDX_C2H2I];
    data[2068] = 0.0 - k[1868]*y[IDX_C2H2I];
    data[2069] = 0.0 + k[1588]*y[IDX_CH3I];
    data[2070] = 0.0 + k[1721]*y[IDX_C2HI];
    data[2071] = 0.0 - k[1737]*y[IDX_C2H2I] + k[1738]*y[IDX_C2H3I];
    data[2072] = 0.0 + k[1097]*y[IDX_HI] + k[1118]*y[IDX_HCNI];
    data[2073] = 0.0 + k[870]*y[IDX_CNI];
    data[2074] = 0.0 + k[95]*y[IDX_HCNI] + k[867]*y[IDX_HCOI] + k[940]*y[IDX_H2I] +
        k[995]*y[IDX_H2OI];
    data[2075] = 0.0 + k[869]*y[IDX_CNI];
    data[2076] = 0.0 + k[201]*y[IDX_HCNI];
    data[2077] = 0.0 + k[164]*y[IDX_HCNI] + k[917]*y[IDX_CNI];
    data[2078] = 0.0 - k[46]*y[IDX_HCNII];
    data[2079] = 0.0 - k[46]*y[IDX_C2H2I] - k[188]*y[IDX_H2OI] - k[194]*y[IDX_HI] -
        k[197]*y[IDX_NOI] - k[198]*y[IDX_O2I] - k[199]*y[IDX_SI] -
        k[287]*y[IDX_NH3I] - k[538]*y[IDX_EM] - k[652]*y[IDX_C2I] -
        k[679]*y[IDX_C2HI] - k[693]*y[IDX_CI] - k[760]*y[IDX_CH2I] -
        k[811]*y[IDX_CH4I] - k[846]*y[IDX_CHI] - k[947]*y[IDX_H2I] -
        k[1002]*y[IDX_H2OI] - k[1110]*y[IDX_CO2I] - k[1111]*y[IDX_COI] -
        k[1112]*y[IDX_H2COI] - k[1113]*y[IDX_HCNI] - k[1114]*y[IDX_HCOI] -
        k[1115]*y[IDX_HCOI] - k[1116]*y[IDX_HNCI] - k[1117]*y[IDX_SI] -
        k[1403]*y[IDX_NH2I] - k[1430]*y[IDX_NH3I] - k[1451]*y[IDX_NHI] -
        k[1541]*y[IDX_OHI] - k[2241];
    data[2080] = 0.0 - k[1116]*y[IDX_HCNII];
    data[2081] = 0.0 + k[200]*y[IDX_HCNI];
    data[2082] = 0.0 - k[1110]*y[IDX_HCNII];
    data[2083] = 0.0 + k[1347]*y[IDX_C2I] + k[1349]*y[IDX_CNI];
    data[2084] = 0.0 + k[629]*y[IDX_CII] + k[729]*y[IDX_CHII] - k[1403]*y[IDX_HCNII];
    data[2085] = 0.0 + k[242]*y[IDX_HCNI] + k[1294]*y[IDX_CH4I];
    data[2086] = 0.0 - k[679]*y[IDX_HCNII];
    data[2087] = 0.0 + k[1331]*y[IDX_NI];
    data[2088] = 0.0 + k[1519]*y[IDX_CNI];
    data[2089] = 0.0 - k[652]*y[IDX_HCNII] + k[1347]*y[IDX_NHII];
    data[2090] = 0.0 - k[760]*y[IDX_HCNII];
    data[2091] = 0.0 + k[729]*y[IDX_NH2I];
    data[2092] = 0.0 - k[1451]*y[IDX_HCNII];
    data[2093] = 0.0 + k[869]*y[IDX_HNOII] + k[870]*y[IDX_O2HII] + k[917]*y[IDX_H2II] +
        k[1035]*y[IDX_H3II] + k[1349]*y[IDX_NHII] + k[1519]*y[IDX_OHII];
    data[2094] = 0.0 - k[1112]*y[IDX_HCNII];
    data[2095] = 0.0 - k[811]*y[IDX_HCNII] + k[1294]*y[IDX_NII];
    data[2096] = 0.0 + k[867]*y[IDX_CNII] - k[1114]*y[IDX_HCNII] - k[1115]*y[IDX_HCNII];
    data[2097] = 0.0 + k[95]*y[IDX_CNII] + k[124]*y[IDX_HII] + k[164]*y[IDX_H2II] +
        k[200]*y[IDX_COII] + k[201]*y[IDX_N2II] + k[242]*y[IDX_NII] -
        k[1113]*y[IDX_HCNII] + k[1118]*y[IDX_C2N2II];
    data[2098] = 0.0 - k[197]*y[IDX_HCNII];
    data[2099] = 0.0 - k[846]*y[IDX_HCNII];
    data[2100] = 0.0 - k[287]*y[IDX_HCNII] + k[630]*y[IDX_CII] - k[1430]*y[IDX_HCNII];
    data[2101] = 0.0 - k[198]*y[IDX_HCNII];
    data[2102] = 0.0 + k[629]*y[IDX_NH2I] + k[630]*y[IDX_NH3I];
    data[2103] = 0.0 - k[199]*y[IDX_HCNII] - k[1117]*y[IDX_HCNII];
    data[2104] = 0.0 - k[1541]*y[IDX_HCNII];
    data[2105] = 0.0 + k[1331]*y[IDX_CH2II];
    data[2106] = 0.0 + k[1035]*y[IDX_CNI];
    data[2107] = 0.0 + k[124]*y[IDX_HCNI];
    data[2108] = 0.0 - k[693]*y[IDX_HCNII];
    data[2109] = 0.0 - k[188]*y[IDX_HCNII] + k[995]*y[IDX_CNII] - k[1002]*y[IDX_HCNII];
    data[2110] = 0.0 - k[1111]*y[IDX_HCNII];
    data[2111] = 0.0 + k[940]*y[IDX_CNII] - k[947]*y[IDX_HCNII];
    data[2112] = 0.0 - k[538]*y[IDX_HCNII];
    data[2113] = 0.0 - k[194]*y[IDX_HCNII] + k[1097]*y[IDX_C2N2II];
    data[2114] = 0.0 + k[2335] + k[2336] + k[2337] + k[2338];
    data[2115] = 0.0 + k[1788]*y[IDX_CI];
    data[2116] = 0.0 + k[485]*y[IDX_EM];
    data[2117] = 0.0 + k[1131]*y[IDX_HCNHII];
    data[2118] = 0.0 - k[1167]*y[IDX_HNCI];
    data[2119] = 0.0 - k[1172]*y[IDX_HNCI];
    data[2120] = 0.0 - k[1171]*y[IDX_HNCI];
    data[2121] = 0.0 - k[1169]*y[IDX_HNCI];
    data[2122] = 0.0 - k[833]*y[IDX_HNCI];
    data[2123] = 0.0 + k[541]*y[IDX_EM] + k[762]*y[IDX_CH2I] + k[848]*y[IDX_CHI] +
        k[1131]*y[IDX_CH3CNI] + k[1133]*y[IDX_H2COI] + k[1135]*y[IDX_H2SI] +
        k[1405]*y[IDX_NH2I] + k[1432]*y[IDX_NH3I];
    data[2124] = 0.0 - k[1166]*y[IDX_HNCI];
    data[2125] = 0.0 - k[1116]*y[IDX_HNCI];
    data[2126] = 0.0 - k[1]*y[IDX_HII] - k[414] - k[627]*y[IDX_CII] - k[664]*y[IDX_C2HII]
        - k[669]*y[IDX_C2H2II] - k[727]*y[IDX_CHII] - k[833]*y[IDX_CH5II] -
        k[986]*y[IDX_H2OII] - k[1050]*y[IDX_H3II] - k[1090]*y[IDX_H3OII] -
        k[1116]*y[IDX_HCNII] - k[1165]*y[IDX_H2COII] - k[1166]*y[IDX_H3COII] -
        k[1167]*y[IDX_H3SII] - k[1168]*y[IDX_HCOII] - k[1169]*y[IDX_HNOII] -
        k[1170]*y[IDX_HSII] - k[1171]*y[IDX_N2HII] - k[1172]*y[IDX_O2HII] -
        k[1242]*y[IDX_HeII] - k[1243]*y[IDX_HeII] - k[1244]*y[IDX_HeII] -
        k[1362]*y[IDX_NHII] - k[1387]*y[IDX_NH2II] - k[1528]*y[IDX_OHII] -
        k[1579]*y[IDX_C2HI] - k[1707]*y[IDX_CNI] - k[1754]*y[IDX_HI] - k[2036] -
        k[2297];
    data[2127] = 0.0 - k[1170]*y[IDX_HNCI];
    data[2128] = 0.0 - k[1362]*y[IDX_HNCI];
    data[2129] = 0.0 + k[1405]*y[IDX_HCNHII] + k[1600]*y[IDX_CI];
    data[2130] = 0.0 - k[1387]*y[IDX_HNCI];
    data[2131] = 0.0 - k[664]*y[IDX_HNCI];
    data[2132] = 0.0 - k[1579]*y[IDX_HNCI];
    data[2133] = 0.0 - k[1165]*y[IDX_HNCI];
    data[2134] = 0.0 + k[1135]*y[IDX_HCNHII];
    data[2135] = 0.0 - k[986]*y[IDX_HNCI];
    data[2136] = 0.0 - k[1528]*y[IDX_HNCI];
    data[2137] = 0.0 - k[669]*y[IDX_HNCI];
    data[2138] = 0.0 + k[762]*y[IDX_HCNHII] + k[1802]*y[IDX_NI];
    data[2139] = 0.0 - k[727]*y[IDX_HNCI];
    data[2140] = 0.0 - k[1707]*y[IDX_HNCI];
    data[2141] = 0.0 + k[1133]*y[IDX_HCNHII];
    data[2142] = 0.0 + k[848]*y[IDX_HCNHII];
    data[2143] = 0.0 + k[1432]*y[IDX_HCNHII];
    data[2144] = 0.0 - k[1090]*y[IDX_HNCI];
    data[2145] = 0.0 - k[627]*y[IDX_HNCI];
    data[2146] = 0.0 + k[1802]*y[IDX_CH2I];
    data[2147] = 0.0 - k[1242]*y[IDX_HNCI] - k[1243]*y[IDX_HNCI] - k[1244]*y[IDX_HNCI];
    data[2148] = 0.0 - k[1050]*y[IDX_HNCI];
    data[2149] = 0.0 - k[1]*y[IDX_HNCI];
    data[2150] = 0.0 + k[1600]*y[IDX_NH2I] + k[1788]*y[IDX_HNCOI];
    data[2151] = 0.0 - k[1168]*y[IDX_HNCI];
    data[2152] = 0.0 + k[485]*y[IDX_CH3CNHII] + k[541]*y[IDX_HCNHII];
    data[2153] = 0.0 - k[1754]*y[IDX_HNCI];
    data[2154] = 0.0 + k[1274]*y[IDX_HeII];
    data[2155] = 0.0 + k[1109]*y[IDX_HI];
    data[2156] = 0.0 - k[1563]*y[IDX_SiII];
    data[2157] = 0.0 + k[1342]*y[IDX_NI];
    data[2158] = 0.0 + k[642]*y[IDX_CII] + k[1276]*y[IDX_HeII];
    data[2159] = 0.0 + k[1278]*y[IDX_HeII];
    data[2160] = 0.0 + k[1288]*y[IDX_HeII];
    data[2161] = 0.0 + k[908]*y[IDX_HII] + k[1284]*y[IDX_HeII];
    data[2162] = 0.0 + k[704]*y[IDX_CI] + k[773]*y[IDX_CH2I] + k[881]*y[IDX_COI] +
        k[1344]*y[IDX_NI] + k[1516]*y[IDX_OI] + k[1555]*y[IDX_SI] + k[2092];
    data[2163] = 0.0 + k[1108]*y[IDX_HI] + k[2082];
    data[2164] = 0.0 + k[671]*y[IDX_C2H2II] + k[1282]*y[IDX_HeII];
    data[2165] = 0.0 + k[348]*y[IDX_SiI];
    data[2166] = 0.0 + k[645]*y[IDX_CII] + k[1285]*y[IDX_HeII];
    data[2167] = 0.0 - k[1564]*y[IDX_SiII];
    data[2168] = 0.0 - k[231]*y[IDX_SiII];
    data[2169] = 0.0 - k[1565]*y[IDX_SiII];
    data[2170] = 0.0 + k[350]*y[IDX_SiI];
    data[2171] = 0.0 + k[26]*y[IDX_CII] + k[60]*y[IDX_CHII] + k[143]*y[IDX_HII] +
        k[186]*y[IDX_H2OII] + k[219]*y[IDX_HeII] + k[281]*y[IDX_NH3II] +
        k[348]*y[IDX_CSII] + k[349]*y[IDX_H2COII] + k[350]*y[IDX_H2SII] +
        k[351]*y[IDX_HSII] + k[352]*y[IDX_NOII] + k[353]*y[IDX_O2II] +
        k[354]*y[IDX_SII] + k[448] + k[2077];
    data[2172] = 0.0 - k[231]*y[IDX_MgI] - k[684]*y[IDX_C2HI] - k[862]*y[IDX_CHI] -
        k[1013]*y[IDX_H2OI] - k[1549]*y[IDX_OHI] - k[1563]*y[IDX_C2H5OHI] -
        k[1564]*y[IDX_CH3OHI] - k[1565]*y[IDX_OCSI] - k[2118]*y[IDX_H2I] -
        k[2126]*y[IDX_HI] - k[2130]*y[IDX_OI] - k[2144]*y[IDX_EM] - k[2173];
    data[2173] = 0.0 + k[351]*y[IDX_SiI];
    data[2174] = 0.0 + k[353]*y[IDX_SiI];
    data[2175] = 0.0 - k[684]*y[IDX_SiII];
    data[2176] = 0.0 + k[352]*y[IDX_SiI];
    data[2177] = 0.0 + k[349]*y[IDX_SiI];
    data[2178] = 0.0 + k[281]*y[IDX_SiI];
    data[2179] = 0.0 + k[186]*y[IDX_SiI];
    data[2180] = 0.0 + k[671]*y[IDX_SiH4I];
    data[2181] = 0.0 + k[773]*y[IDX_SiOII];
    data[2182] = 0.0 + k[60]*y[IDX_SiI];
    data[2183] = 0.0 + k[354]*y[IDX_SiI];
    data[2184] = 0.0 - k[862]*y[IDX_SiII];
    data[2185] = 0.0 + k[26]*y[IDX_SiI] + k[642]*y[IDX_SiCI] + k[645]*y[IDX_SiOI];
    data[2186] = 0.0 + k[1555]*y[IDX_SiOII];
    data[2187] = 0.0 - k[1549]*y[IDX_SiII];
    data[2188] = 0.0 + k[1342]*y[IDX_SiCII] + k[1344]*y[IDX_SiOII];
    data[2189] = 0.0 + k[219]*y[IDX_SiI] + k[1274]*y[IDX_SiC2I] + k[1276]*y[IDX_SiCI] +
        k[1278]*y[IDX_SiH2I] + k[1282]*y[IDX_SiH4I] + k[1284]*y[IDX_SiHI] +
        k[1285]*y[IDX_SiOI] + k[1288]*y[IDX_SiSI];
    data[2190] = 0.0 + k[143]*y[IDX_SiI] + k[908]*y[IDX_SiHI];
    data[2191] = 0.0 + k[1516]*y[IDX_SiOII] - k[2130]*y[IDX_SiII];
    data[2192] = 0.0 + k[704]*y[IDX_SiOII];
    data[2193] = 0.0 - k[1013]*y[IDX_SiII];
    data[2194] = 0.0 + k[881]*y[IDX_SiOII];
    data[2195] = 0.0 - k[2118]*y[IDX_SiII];
    data[2196] = 0.0 - k[2144]*y[IDX_SiII];
    data[2197] = 0.0 + k[1108]*y[IDX_SiHII] + k[1109]*y[IDX_SiSII] - k[2126]*y[IDX_SiII];
    data[2198] = 0.0 + k[1194]*y[IDX_HeII];
    data[2199] = 0.0 + k[635]*y[IDX_CII];
    data[2200] = 0.0 - k[873]*y[IDX_COII];
    data[2201] = 0.0 + k[1496]*y[IDX_OI];
    data[2202] = 0.0 + k[641]*y[IDX_CII];
    data[2203] = 0.0 + k[649]*y[IDX_O2I] + k[1490]*y[IDX_OI];
    data[2204] = 0.0 + k[93]*y[IDX_COI];
    data[2205] = 0.0 + k[1267]*y[IDX_HeII];
    data[2206] = 0.0 + k[107]*y[IDX_COI];
    data[2207] = 0.0 + k[160]*y[IDX_COI];
    data[2208] = 0.0 - k[37]*y[IDX_C2I] - k[48]*y[IDX_C2HI] - k[52]*y[IDX_CI] -
        k[64]*y[IDX_CH2I] - k[81]*y[IDX_CH4I] - k[84]*y[IDX_CHI] -
        k[101]*y[IDX_H2COI] - k[102]*y[IDX_H2SI] - k[103]*y[IDX_HCOI] -
        k[104]*y[IDX_NOI] - k[105]*y[IDX_O2I] - k[106]*y[IDX_SI] -
        k[187]*y[IDX_H2OI] - k[192]*y[IDX_HI] - k[200]*y[IDX_HCNI] -
        k[273]*y[IDX_NH2I] - k[283]*y[IDX_NH3I] - k[295]*y[IDX_NHI] -
        k[327]*y[IDX_OI] - k[341]*y[IDX_OHI] - k[499]*y[IDX_EM] -
        k[677]*y[IDX_C2HI] - k[756]*y[IDX_CH2I] - k[807]*y[IDX_CH4I] -
        k[841]*y[IDX_CHI] - k[871]*y[IDX_H2COI] - k[872]*y[IDX_H2SI] -
        k[873]*y[IDX_SO2I] - k[941]*y[IDX_H2I] - k[942]*y[IDX_H2I] -
        k[997]*y[IDX_H2OI] - k[1398]*y[IDX_NH2I] - k[1423]*y[IDX_NH3I] -
        k[1448]*y[IDX_NHI] - k[1539]*y[IDX_OHI] - k[2002] - k[2234];
    data[2209] = 0.0 + k[616]*y[IDX_CII] + k[1209]*y[IDX_HeII] + k[1296]*y[IDX_NII];
    data[2210] = 0.0 - k[273]*y[IDX_COII] - k[1398]*y[IDX_COII];
    data[2211] = 0.0 + k[656]*y[IDX_C2I] + k[700]*y[IDX_CI];
    data[2212] = 0.0 + k[238]*y[IDX_COI] + k[1296]*y[IDX_CO2I];
    data[2213] = 0.0 - k[48]*y[IDX_COII] - k[677]*y[IDX_COII] + k[1465]*y[IDX_OII];
    data[2214] = 0.0 + k[310]*y[IDX_COI] + k[857]*y[IDX_CHI] + k[1463]*y[IDX_C2I] +
        k[1465]*y[IDX_C2HI] + k[2103]*y[IDX_CI];
    data[2215] = 0.0 - k[102]*y[IDX_COII] - k[872]*y[IDX_COII];
    data[2216] = 0.0 - k[37]*y[IDX_COII] + k[656]*y[IDX_O2II] + k[1463]*y[IDX_OII];
    data[2217] = 0.0 - k[64]*y[IDX_COII] - k[756]*y[IDX_COII];
    data[2218] = 0.0 + k[732]*y[IDX_O2I] + k[735]*y[IDX_OI] + k[738]*y[IDX_OHI];
    data[2219] = 0.0 - k[295]*y[IDX_COII] - k[1448]*y[IDX_COII];
    data[2220] = 0.0 - k[101]*y[IDX_COII] - k[871]*y[IDX_COII] + k[892]*y[IDX_HII] +
        k[1216]*y[IDX_HeII];
    data[2221] = 0.0 - k[81]*y[IDX_COII] - k[807]*y[IDX_COII];
    data[2222] = 0.0 - k[103]*y[IDX_COII] + k[897]*y[IDX_HII] + k[1235]*y[IDX_HeII];
    data[2223] = 0.0 - k[200]*y[IDX_COII];
    data[2224] = 0.0 - k[104]*y[IDX_COII];
    data[2225] = 0.0 - k[84]*y[IDX_COII] - k[841]*y[IDX_COII] + k[857]*y[IDX_OII];
    data[2226] = 0.0 - k[283]*y[IDX_COII] - k[1423]*y[IDX_COII];
    data[2227] = 0.0 - k[105]*y[IDX_COII] + k[633]*y[IDX_CII] + k[649]*y[IDX_C2II] +
        k[732]*y[IDX_CHII];
    data[2228] = 0.0 + k[616]*y[IDX_CO2I] + k[633]*y[IDX_O2I] + k[635]*y[IDX_OCNI] +
        k[637]*y[IDX_OHI] + k[641]*y[IDX_SOI] + k[2098]*y[IDX_OI];
    data[2229] = 0.0 - k[106]*y[IDX_COII];
    data[2230] = 0.0 - k[341]*y[IDX_COII] + k[637]*y[IDX_CII] + k[738]*y[IDX_CHII] -
        k[1539]*y[IDX_COII];
    data[2231] = 0.0 + k[1194]*y[IDX_CH2COI] + k[1209]*y[IDX_CO2I] + k[1216]*y[IDX_H2COI]
        + k[1235]*y[IDX_HCOI] + k[1267]*y[IDX_OCSI];
    data[2232] = 0.0 + k[892]*y[IDX_H2COI] + k[897]*y[IDX_HCOI];
    data[2233] = 0.0 - k[327]*y[IDX_COII] + k[735]*y[IDX_CHII] + k[1490]*y[IDX_C2II] +
        k[1496]*y[IDX_CSII] + k[2098]*y[IDX_CII];
    data[2234] = 0.0 - k[52]*y[IDX_COII] + k[700]*y[IDX_O2II] + k[2103]*y[IDX_OII];
    data[2235] = 0.0 + k[2029];
    data[2236] = 0.0 - k[187]*y[IDX_COII] - k[997]*y[IDX_COII];
    data[2237] = 0.0 + k[93]*y[IDX_CNII] + k[107]*y[IDX_N2II] + k[160]*y[IDX_H2II] +
        k[238]*y[IDX_NII] + k[310]*y[IDX_OII] + k[357];
    data[2238] = 0.0 - k[941]*y[IDX_COII] - k[942]*y[IDX_COII];
    data[2239] = 0.0 - k[499]*y[IDX_COII];
    data[2240] = 0.0 - k[192]*y[IDX_COII];
    data[2241] = 0.0 + k[1224]*y[IDX_HeII];
    data[2242] = 0.0 - k[225]*y[IDX_HSII];
    data[2243] = 0.0 + k[1554]*y[IDX_SI];
    data[2244] = 0.0 + k[1324]*y[IDX_SI];
    data[2245] = 0.0 + k[904]*y[IDX_HII];
    data[2246] = 0.0 + k[1103]*y[IDX_HI] + k[1498]*y[IDX_OI];
    data[2247] = 0.0 + k[1174]*y[IDX_SI];
    data[2248] = 0.0 + k[835]*y[IDX_SI];
    data[2249] = 0.0 + k[1315]*y[IDX_H2SI];
    data[2250] = 0.0 + k[128]*y[IDX_HII];
    data[2251] = 0.0 + k[923]*y[IDX_H2SI];
    data[2252] = 0.0 - k[351]*y[IDX_HSII];
    data[2253] = 0.0 + k[1117]*y[IDX_SI];
    data[2254] = 0.0 - k[1170]*y[IDX_HSII];
    data[2255] = 0.0 - k[225]*y[IDX_MgI] - k[288]*y[IDX_NH3I] - k[301]*y[IDX_NOI] -
        k[347]*y[IDX_SI] - k[351]*y[IDX_SiI] - k[554]*y[IDX_EM] -
        k[697]*y[IDX_CI] - k[814]*y[IDX_CH4I] - k[851]*y[IDX_CHI] -
        k[949]*y[IDX_H2I] - k[1007]*y[IDX_H2OI] - k[1105]*y[IDX_HI] -
        k[1126]*y[IDX_HCNI] - k[1170]*y[IDX_HNCI] - k[1175]*y[IDX_H2SI] -
        k[1176]*y[IDX_H2SI] - k[1336]*y[IDX_NI] - k[1437]*y[IDX_NH3I] -
        k[1503]*y[IDX_OI] - k[1504]*y[IDX_OI] - k[2039] - k[2040] -
        k[2116]*y[IDX_H2I] - k[2265];
    data[2256] = 0.0 + k[1372]*y[IDX_SI];
    data[2257] = 0.0 + k[1300]*y[IDX_H2SI];
    data[2258] = 0.0 + k[1382]*y[IDX_H2SI] + k[1393]*y[IDX_SI];
    data[2259] = 0.0 + k[1472]*y[IDX_H2SI];
    data[2260] = 0.0 + k[968]*y[IDX_SI];
    data[2261] = 0.0 + k[894]*y[IDX_HII] + k[923]*y[IDX_H2II] - k[1175]*y[IDX_HSII] -
        k[1176]*y[IDX_HSII] + k[1226]*y[IDX_HeII] + k[1300]*y[IDX_NII] +
        k[1315]*y[IDX_N2II] + k[1382]*y[IDX_NH2II] + k[1472]*y[IDX_OII];
    data[2262] = 0.0 + k[987]*y[IDX_SI];
    data[2263] = 0.0 + k[1533]*y[IDX_SI];
    data[2264] = 0.0 + k[740]*y[IDX_SI];
    data[2265] = 0.0 + k[961]*y[IDX_H2I] + k[1163]*y[IDX_HCOI];
    data[2266] = 0.0 - k[814]*y[IDX_HSII];
    data[2267] = 0.0 + k[1163]*y[IDX_SII];
    data[2268] = 0.0 - k[1126]*y[IDX_HSII];
    data[2269] = 0.0 - k[301]*y[IDX_HSII];
    data[2270] = 0.0 - k[851]*y[IDX_HSII];
    data[2271] = 0.0 - k[288]*y[IDX_HSII] - k[1437]*y[IDX_HSII];
    data[2272] = 0.0 - k[347]*y[IDX_HSII] + k[740]*y[IDX_CHII] + k[835]*y[IDX_CH5II] +
        k[968]*y[IDX_H2COII] + k[987]*y[IDX_H2OII] + k[1068]*y[IDX_H3II] +
        k[1117]*y[IDX_HCNII] + k[1151]*y[IDX_HCOII] + k[1174]*y[IDX_HNOII] +
        k[1324]*y[IDX_N2HII] + k[1372]*y[IDX_NHII] + k[1393]*y[IDX_NH2II] +
        k[1533]*y[IDX_OHII] + k[1554]*y[IDX_O2HII];
    data[2273] = 0.0 - k[1336]*y[IDX_HSII];
    data[2274] = 0.0 + k[1224]*y[IDX_H2S2I] + k[1226]*y[IDX_H2SI];
    data[2275] = 0.0 + k[1068]*y[IDX_SI];
    data[2276] = 0.0 + k[128]*y[IDX_HSI] + k[894]*y[IDX_H2SI] + k[904]*y[IDX_OCSI];
    data[2277] = 0.0 + k[1498]*y[IDX_H2SII] - k[1503]*y[IDX_HSII] - k[1504]*y[IDX_HSII];
    data[2278] = 0.0 - k[697]*y[IDX_HSII];
    data[2279] = 0.0 + k[1151]*y[IDX_SI];
    data[2280] = 0.0 - k[1007]*y[IDX_HSII];
    data[2281] = 0.0 - k[949]*y[IDX_HSII] + k[961]*y[IDX_SII] - k[2116]*y[IDX_HSII];
    data[2282] = 0.0 - k[554]*y[IDX_HSII];
    data[2283] = 0.0 + k[1103]*y[IDX_H2SII] - k[1105]*y[IDX_HSII];
    data[2284] = 0.0 + k[2435] + k[2436] + k[2437] + k[2438];
    data[2285] = 0.0 + k[411];
    data[2286] = 0.0 + k[879]*y[IDX_COI];
    data[2287] = 0.0 + k[1719]*y[IDX_COI];
    data[2288] = 0.0 + k[1717]*y[IDX_COI];
    data[2289] = 0.0 + k[1860]*y[IDX_NOI] + k[1863]*y[IDX_O2I];
    data[2290] = 0.0 + k[791]*y[IDX_HCO2II];
    data[2291] = 0.0 + k[873]*y[IDX_COII];
    data[2292] = 0.0 + k[1716]*y[IDX_COI];
    data[2293] = 0.0 + k[881]*y[IDX_COI];
    data[2294] = 0.0 + k[543]*y[IDX_EM] + k[695]*y[IDX_CI] + k[791]*y[IDX_CH3CNI] +
        k[812]*y[IDX_CH4I] + k[875]*y[IDX_COI] + k[1004]*y[IDX_H2OI] +
        k[1434]*y[IDX_NH3I];
    data[2295] = 0.0 - k[796]*y[IDX_CO2I];
    data[2296] = 0.0 + k[1562]*y[IDX_OCSI];
    data[2297] = 0.0 - k[1489]*y[IDX_CO2I];
    data[2298] = 0.0 - k[1322]*y[IDX_CO2I];
    data[2299] = 0.0 + k[1479]*y[IDX_OII] + k[1562]*y[IDX_SOII] + k[1912]*y[IDX_OI];
    data[2300] = 0.0 - k[1173]*y[IDX_CO2I];
    data[2301] = 0.0 - k[825]*y[IDX_CO2I];
    data[2302] = 0.0 - k[918]*y[IDX_CO2I];
    data[2303] = 0.0 - k[1956]*y[IDX_CO2I];
    data[2304] = 0.0 - k[1110]*y[IDX_CO2I];
    data[2305] = 0.0 + k[873]*y[IDX_SO2I];
    data[2306] = 0.0 - k[393] - k[616]*y[IDX_CII] - k[714]*y[IDX_CHII] -
        k[741]*y[IDX_CH2II] - k[796]*y[IDX_CH4II] - k[825]*y[IDX_CH5II] -
        k[891]*y[IDX_HII] - k[918]*y[IDX_H2II] - k[1036]*y[IDX_H3II] -
        k[1110]*y[IDX_HCNII] - k[1173]*y[IDX_HNOII] - k[1209]*y[IDX_HeII] -
        k[1210]*y[IDX_HeII] - k[1211]*y[IDX_HeII] - k[1212]*y[IDX_HeII] -
        k[1296]*y[IDX_NII] - k[1322]*y[IDX_N2HII] - k[1350]*y[IDX_NHII] -
        k[1351]*y[IDX_NHII] - k[1352]*y[IDX_NHII] - k[1470]*y[IDX_OII] -
        k[1489]*y[IDX_O2HII] - k[1520]*y[IDX_OHII] - k[1677]*y[IDX_CHI] -
        k[1744]*y[IDX_HI] - k[1808]*y[IDX_NI] - k[1882]*y[IDX_OI] -
        k[1956]*y[IDX_SiI] - k[2003] - k[2215];
    data[2307] = 0.0 - k[1350]*y[IDX_CO2I] - k[1351]*y[IDX_CO2I] - k[1352]*y[IDX_CO2I];
    data[2308] = 0.0 - k[1296]*y[IDX_CO2I];
    data[2309] = 0.0 - k[1470]*y[IDX_CO2I] + k[1479]*y[IDX_OCSI];
    data[2310] = 0.0 - k[741]*y[IDX_CO2I];
    data[2311] = 0.0 - k[1520]*y[IDX_CO2I];
    data[2312] = 0.0 + k[1632]*y[IDX_O2I] + k[1633]*y[IDX_O2I];
    data[2313] = 0.0 - k[714]*y[IDX_CO2I];
    data[2314] = 0.0 + k[812]*y[IDX_HCO2II];
    data[2315] = 0.0 + k[1784]*y[IDX_O2I] + k[1892]*y[IDX_OI];
    data[2316] = 0.0 + k[1860]*y[IDX_OCNI];
    data[2317] = 0.0 - k[1677]*y[IDX_CO2I] + k[1687]*y[IDX_O2I];
    data[2318] = 0.0 + k[1434]*y[IDX_HCO2II];
    data[2319] = 0.0 + k[1632]*y[IDX_CH2I] + k[1633]*y[IDX_CH2I] + k[1687]*y[IDX_CHI] +
        k[1718]*y[IDX_COI] + k[1784]*y[IDX_HCOI] + k[1863]*y[IDX_OCNI];
    data[2320] = 0.0 - k[616]*y[IDX_CO2I];
    data[2321] = 0.0 + k[1934]*y[IDX_COI];
    data[2322] = 0.0 - k[1808]*y[IDX_CO2I];
    data[2323] = 0.0 - k[1209]*y[IDX_CO2I] - k[1210]*y[IDX_CO2I] - k[1211]*y[IDX_CO2I] -
        k[1212]*y[IDX_CO2I];
    data[2324] = 0.0 - k[1036]*y[IDX_CO2I];
    data[2325] = 0.0 - k[891]*y[IDX_CO2I];
    data[2326] = 0.0 - k[1882]*y[IDX_CO2I] + k[1892]*y[IDX_HCOI] + k[1912]*y[IDX_OCSI];
    data[2327] = 0.0 + k[695]*y[IDX_HCO2II];
    data[2328] = 0.0 + k[1004]*y[IDX_HCO2II];
    data[2329] = 0.0 + k[875]*y[IDX_HCO2II] + k[879]*y[IDX_SO2II] + k[881]*y[IDX_SiOII] +
        k[1716]*y[IDX_HNOI] + k[1717]*y[IDX_NO2I] + k[1718]*y[IDX_O2I] +
        k[1719]*y[IDX_O2HI] + k[1934]*y[IDX_OHI];
    data[2330] = 0.0 + k[543]*y[IDX_HCO2II];
    data[2331] = 0.0 - k[1744]*y[IDX_CO2I];
    data[2332] = 0.0 + k[294]*y[IDX_NHI];
    data[2333] = 0.0 + k[296]*y[IDX_NHI];
    data[2334] = 0.0 + k[168]*y[IDX_NHI] + k[928]*y[IDX_NI];
    data[2335] = 0.0 + k[1244]*y[IDX_HeII] - k[1362]*y[IDX_NHII];
    data[2336] = 0.0 + k[295]*y[IDX_NHI];
    data[2337] = 0.0 - k[1350]*y[IDX_NHII] - k[1351]*y[IDX_NHII] - k[1352]*y[IDX_NHII];
    data[2338] = 0.0 - k[260]*y[IDX_H2COI] - k[261]*y[IDX_H2OI] - k[262]*y[IDX_NH3I] -
        k[263]*y[IDX_NOI] - k[264]*y[IDX_O2I] - k[265]*y[IDX_SI] -
        k[567]*y[IDX_EM] - k[699]*y[IDX_CI] - k[766]*y[IDX_CH2I] -
        k[854]*y[IDX_CHI] - k[954]*y[IDX_H2I] - k[955]*y[IDX_H2I] -
        k[1337]*y[IDX_NI] - k[1345]*y[IDX_C2I] - k[1346]*y[IDX_C2I] -
        k[1347]*y[IDX_C2I] - k[1348]*y[IDX_C2HI] - k[1349]*y[IDX_CNI] -
        k[1350]*y[IDX_CO2I] - k[1351]*y[IDX_CO2I] - k[1352]*y[IDX_CO2I] -
        k[1353]*y[IDX_COI] - k[1354]*y[IDX_H2COI] - k[1355]*y[IDX_H2COI] -
        k[1356]*y[IDX_H2OI] - k[1357]*y[IDX_H2OI] - k[1358]*y[IDX_H2OI] -
        k[1359]*y[IDX_H2OI] - k[1360]*y[IDX_HCNI] - k[1361]*y[IDX_HCOI] -
        k[1362]*y[IDX_HNCI] - k[1363]*y[IDX_N2I] - k[1364]*y[IDX_NH2I] -
        k[1365]*y[IDX_NH3I] - k[1366]*y[IDX_NHI] - k[1367]*y[IDX_NOI] -
        k[1368]*y[IDX_O2I] - k[1369]*y[IDX_O2I] - k[1370]*y[IDX_OI] -
        k[1371]*y[IDX_OHI] - k[1372]*y[IDX_SI] - k[1373]*y[IDX_SI] - k[2048] -
        k[2232];
    data[2339] = 0.0 + k[1253]*y[IDX_HeII] - k[1364]*y[IDX_NHII];
    data[2340] = 0.0 + k[247]*y[IDX_NHI] + k[952]*y[IDX_H2I] + k[1301]*y[IDX_H2SI] +
        k[1303]*y[IDX_HCOI];
    data[2341] = 0.0 - k[1348]*y[IDX_NHII];
    data[2342] = 0.0 + k[297]*y[IDX_NHI];
    data[2343] = 0.0 + k[1301]*y[IDX_NII];
    data[2344] = 0.0 - k[1363]*y[IDX_NHII];
    data[2345] = 0.0 - k[1345]*y[IDX_NHII] - k[1346]*y[IDX_NHII] - k[1347]*y[IDX_NHII];
    data[2346] = 0.0 - k[766]*y[IDX_NHII];
    data[2347] = 0.0 + k[132]*y[IDX_HII] + k[168]*y[IDX_H2II] + k[247]*y[IDX_NII] +
        k[294]*y[IDX_CNII] + k[295]*y[IDX_COII] + k[296]*y[IDX_N2II] +
        k[297]*y[IDX_OII] + k[430] - k[1366]*y[IDX_NHII] + k[2055];
    data[2348] = 0.0 - k[1349]*y[IDX_NHII];
    data[2349] = 0.0 - k[260]*y[IDX_NHII] - k[1354]*y[IDX_NHII] - k[1355]*y[IDX_NHII];
    data[2350] = 0.0 + k[1303]*y[IDX_NII] - k[1361]*y[IDX_NHII];
    data[2351] = 0.0 - k[1360]*y[IDX_NHII];
    data[2352] = 0.0 - k[263]*y[IDX_NHII] - k[1367]*y[IDX_NHII];
    data[2353] = 0.0 - k[854]*y[IDX_NHII];
    data[2354] = 0.0 - k[262]*y[IDX_NHII] + k[1254]*y[IDX_HeII] - k[1365]*y[IDX_NHII];
    data[2355] = 0.0 - k[264]*y[IDX_NHII] - k[1368]*y[IDX_NHII] - k[1369]*y[IDX_NHII];
    data[2356] = 0.0 - k[265]*y[IDX_NHII] - k[1372]*y[IDX_NHII] - k[1373]*y[IDX_NHII];
    data[2357] = 0.0 - k[1371]*y[IDX_NHII];
    data[2358] = 0.0 + k[928]*y[IDX_H2II] - k[1337]*y[IDX_NHII];
    data[2359] = 0.0 + k[1244]*y[IDX_HNCI] + k[1253]*y[IDX_NH2I] + k[1254]*y[IDX_NH3I];
    data[2360] = 0.0 + k[132]*y[IDX_NHI];
    data[2361] = 0.0 - k[1370]*y[IDX_NHII];
    data[2362] = 0.0 - k[699]*y[IDX_NHII];
    data[2363] = 0.0 - k[261]*y[IDX_NHII] - k[1356]*y[IDX_NHII] - k[1357]*y[IDX_NHII] -
        k[1358]*y[IDX_NHII] - k[1359]*y[IDX_NHII];
    data[2364] = 0.0 - k[1353]*y[IDX_NHII];
    data[2365] = 0.0 + k[952]*y[IDX_NII] - k[954]*y[IDX_NHII] - k[955]*y[IDX_NHII];
    data[2366] = 0.0 - k[567]*y[IDX_NHII];
    data[2367] = 0.0 + k[1755]*y[IDX_HI];
    data[2368] = 0.0 - k[271]*y[IDX_NH2I] - k[1394]*y[IDX_NH2I];
    data[2369] = 0.0 - k[1410]*y[IDX_NH2I];
    data[2370] = 0.0 - k[272]*y[IDX_NH2I];
    data[2371] = 0.0 - k[1408]*y[IDX_NH2I];
    data[2372] = 0.0 - k[1407]*y[IDX_NH2I];
    data[2373] = 0.0 - k[1397]*y[IDX_NH2I];
    data[2374] = 0.0 - k[275]*y[IDX_NH2I];
    data[2375] = 0.0 - k[1404]*y[IDX_NH2I] - k[1405]*y[IDX_NH2I];
    data[2376] = 0.0 - k[166]*y[IDX_NH2I];
    data[2377] = 0.0 - k[1401]*y[IDX_NH2I];
    data[2378] = 0.0 - k[1403]*y[IDX_NH2I] + k[1430]*y[IDX_NH3I];
    data[2379] = 0.0 - k[273]*y[IDX_NH2I] - k[1398]*y[IDX_NH2I] + k[1423]*y[IDX_NH3I];
    data[2380] = 0.0 + k[1355]*y[IDX_H2COI] - k[1364]*y[IDX_NH2I];
    data[2381] = 0.0 - k[130]*y[IDX_HII] - k[166]*y[IDX_H2II] - k[245]*y[IDX_NII] -
        k[271]*y[IDX_C2II] - k[272]*y[IDX_CNII] - k[273]*y[IDX_COII] -
        k[274]*y[IDX_H2OII] - k[275]*y[IDX_N2II] - k[276]*y[IDX_O2II] -
        k[277]*y[IDX_OHII] - k[315]*y[IDX_OII] - k[424] - k[425] -
        k[629]*y[IDX_CII] - k[729]*y[IDX_CHII] - k[1056]*y[IDX_H3II] -
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
    data[2382] = 0.0 - k[276]*y[IDX_NH2I];
    data[2383] = 0.0 - k[245]*y[IDX_NH2I];
    data[2384] = 0.0 + k[68]*y[IDX_CH2I] + k[89]*y[IDX_CHI] + k[266]*y[IDX_H2SI] +
        k[267]*y[IDX_HCOI] + k[268]*y[IDX_NH3I] + k[269]*y[IDX_NOI] +
        k[270]*y[IDX_SI] - k[1388]*y[IDX_NH2I];
    data[2385] = 0.0 - k[1395]*y[IDX_NH2I];
    data[2386] = 0.0 + k[572]*y[IDX_EM] + k[573]*y[IDX_EM];
    data[2387] = 0.0 - k[315]*y[IDX_NH2I];
    data[2388] = 0.0 - k[1399]*y[IDX_NH2I];
    data[2389] = 0.0 + k[570]*y[IDX_EM] + k[768]*y[IDX_CH2I] - k[1409]*y[IDX_NH2I] +
        k[1417]*y[IDX_NH3I];
    data[2390] = 0.0 + k[266]*y[IDX_NH2II];
    data[2391] = 0.0 - k[274]*y[IDX_NH2I] - k[1400]*y[IDX_NH2I];
    data[2392] = 0.0 - k[277]*y[IDX_NH2I] - k[1411]*y[IDX_NH2I];
    data[2393] = 0.0 - k[1396]*y[IDX_NH2I];
    data[2394] = 0.0 + k[68]*y[IDX_NH2II] + k[768]*y[IDX_NH3II];
    data[2395] = 0.0 - k[1656]*y[IDX_NH2I] + k[1657]*y[IDX_NH3I];
    data[2396] = 0.0 - k[729]*y[IDX_NH2I];
    data[2397] = 0.0 + k[1730]*y[IDX_H2I] + k[1839]*y[IDX_CH4I] + k[1841]*y[IDX_H2OI] +
        k[1842]*y[IDX_NH3I] + k[1842]*y[IDX_NH3I] + k[1845]*y[IDX_NHI] +
        k[1845]*y[IDX_NHI] + k[1855]*y[IDX_OHI];
    data[2398] = 0.0 + k[1838]*y[IDX_NH3I];
    data[2399] = 0.0 + k[1355]*y[IDX_NHII];
    data[2400] = 0.0 - k[1833]*y[IDX_NH2I] + k[1839]*y[IDX_NHI];
    data[2401] = 0.0 + k[267]*y[IDX_NH2II];
    data[2402] = 0.0 + k[1940]*y[IDX_OHI];
    data[2403] = 0.0 + k[269]*y[IDX_NH2II] - k[1834]*y[IDX_NH2I] - k[1835]*y[IDX_NH2I];
    data[2404] = 0.0 + k[89]*y[IDX_NH2II];
    data[2405] = 0.0 + k[268]*y[IDX_NH2II] + k[426] + k[1417]*y[IDX_NH3II] +
        k[1423]*y[IDX_COII] + k[1430]*y[IDX_HCNII] + k[1657]*y[IDX_CH3I] +
        k[1761]*y[IDX_HI] + k[1838]*y[IDX_CNI] + k[1842]*y[IDX_NHI] +
        k[1842]*y[IDX_NHI] + k[1904]*y[IDX_OI] + k[1944]*y[IDX_OHI] + k[2051];
    data[2406] = 0.0 - k[1402]*y[IDX_NH2I];
    data[2407] = 0.0 - k[629]*y[IDX_NH2I];
    data[2408] = 0.0 + k[270]*y[IDX_NH2II];
    data[2409] = 0.0 - k[1836]*y[IDX_NH2I] - k[1837]*y[IDX_NH2I] + k[1855]*y[IDX_NHI] +
        k[1940]*y[IDX_HCNI] + k[1944]*y[IDX_NH3I];
    data[2410] = 0.0 - k[1252]*y[IDX_NH2I] - k[1253]*y[IDX_NH2I];
    data[2411] = 0.0 - k[1056]*y[IDX_NH2I];
    data[2412] = 0.0 - k[130]*y[IDX_NH2I];
    data[2413] = 0.0 - k[1902]*y[IDX_NH2I] - k[1903]*y[IDX_NH2I] + k[1904]*y[IDX_NH3I];
    data[2414] = 0.0 - k[1599]*y[IDX_NH2I] - k[1600]*y[IDX_NH2I] - k[1601]*y[IDX_NH2I];
    data[2415] = 0.0 - k[1406]*y[IDX_NH2I];
    data[2416] = 0.0 + k[1841]*y[IDX_NHI];
    data[2417] = 0.0 - k[1729]*y[IDX_NH2I] + k[1730]*y[IDX_NHI];
    data[2418] = 0.0 + k[570]*y[IDX_NH3II] + k[572]*y[IDX_NH4II] + k[573]*y[IDX_NH4II];
    data[2419] = 0.0 + k[1755]*y[IDX_HNOI] - k[1760]*y[IDX_NH2I] + k[1761]*y[IDX_NH3I];
    data[2420] = 0.0 + k[324]*y[IDX_O2I];
    data[2421] = 0.0 + k[325]*y[IDX_O2I];
    data[2422] = 0.0 - k[1483]*y[IDX_O2II];
    data[2423] = 0.0 + k[79]*y[IDX_O2I];
    data[2424] = 0.0 - k[228]*y[IDX_O2II];
    data[2425] = 0.0 + k[98]*y[IDX_O2I];
    data[2426] = 0.0 + k[256]*y[IDX_O2I];
    data[2427] = 0.0 + k[170]*y[IDX_O2I];
    data[2428] = 0.0 - k[353]*y[IDX_O2II];
    data[2429] = 0.0 - k[321]*y[IDX_O2II] - k[1482]*y[IDX_O2II];
    data[2430] = 0.0 + k[198]*y[IDX_O2I];
    data[2431] = 0.0 + k[105]*y[IDX_O2I];
    data[2432] = 0.0 + k[1211]*y[IDX_HeII] + k[1470]*y[IDX_OII];
    data[2433] = 0.0 + k[264]*y[IDX_O2I];
    data[2434] = 0.0 - k[276]*y[IDX_O2II];
    data[2435] = 0.0 - k[39]*y[IDX_C2I] - k[54]*y[IDX_CI] - k[70]*y[IDX_CH2I] -
        k[91]*y[IDX_CHI] - k[174]*y[IDX_H2COI] - k[204]*y[IDX_HCOI] -
        k[228]*y[IDX_MgI] - k[276]*y[IDX_NH2I] - k[290]*y[IDX_NH3I] -
        k[302]*y[IDX_NOI] - k[321]*y[IDX_C2H2I] - k[322]*y[IDX_H2SI] -
        k[323]*y[IDX_SI] - k[353]*y[IDX_SiI] - k[577]*y[IDX_EM] -
        k[656]*y[IDX_C2I] - k[700]*y[IDX_CI] - k[769]*y[IDX_CH2I] -
        k[858]*y[IDX_CHI] - k[972]*y[IDX_H2COI] - k[1161]*y[IDX_HCOI] -
        k[1339]*y[IDX_NI] - k[1458]*y[IDX_NHI] - k[1482]*y[IDX_C2H2I] -
        k[1483]*y[IDX_CH3OHI] - k[1484]*y[IDX_SI] - k[2060] - k[2229];
    data[2436] = 0.0 + k[249]*y[IDX_O2I];
    data[2437] = 0.0 + k[317]*y[IDX_O2I] + k[1470]*y[IDX_CO2I] + k[1480]*y[IDX_OHI];
    data[2438] = 0.0 - k[322]*y[IDX_O2II];
    data[2439] = 0.0 + k[183]*y[IDX_O2I] + k[1497]*y[IDX_OI];
    data[2440] = 0.0 + k[337]*y[IDX_O2I] + k[1511]*y[IDX_OI];
    data[2441] = 0.0 - k[39]*y[IDX_O2II] - k[656]*y[IDX_O2II];
    data[2442] = 0.0 - k[70]*y[IDX_O2II] - k[769]*y[IDX_O2II];
    data[2443] = 0.0 - k[1458]*y[IDX_O2II];
    data[2444] = 0.0 - k[174]*y[IDX_O2II] - k[972]*y[IDX_O2II];
    data[2445] = 0.0 - k[204]*y[IDX_O2II] - k[1161]*y[IDX_O2II];
    data[2446] = 0.0 - k[302]*y[IDX_O2II];
    data[2447] = 0.0 - k[91]*y[IDX_O2II] - k[858]*y[IDX_O2II];
    data[2448] = 0.0 - k[290]*y[IDX_O2II];
    data[2449] = 0.0 + k[79]*y[IDX_CH4II] + k[98]*y[IDX_CNII] + k[105]*y[IDX_COII] +
        k[135]*y[IDX_HII] + k[170]*y[IDX_H2II] + k[183]*y[IDX_H2OII] +
        k[198]*y[IDX_HCNII] + k[217]*y[IDX_HeII] + k[249]*y[IDX_NII] +
        k[256]*y[IDX_N2II] + k[264]*y[IDX_NHII] + k[317]*y[IDX_OII] +
        k[324]*y[IDX_ClII] + k[325]*y[IDX_SO2II] + k[337]*y[IDX_OHII] + k[435] +
        k[2061];
    data[2450] = 0.0 - k[323]*y[IDX_O2II] - k[1484]*y[IDX_O2II];
    data[2451] = 0.0 + k[1480]*y[IDX_OII];
    data[2452] = 0.0 - k[1339]*y[IDX_O2II];
    data[2453] = 0.0 + k[217]*y[IDX_O2I] + k[1211]*y[IDX_CO2I];
    data[2454] = 0.0 + k[135]*y[IDX_O2I];
    data[2455] = 0.0 + k[1497]*y[IDX_H2OII] + k[1511]*y[IDX_OHII];
    data[2456] = 0.0 - k[54]*y[IDX_O2II] - k[700]*y[IDX_O2II];
    data[2457] = 0.0 - k[577]*y[IDX_O2II];
    data[2458] = 0.0 - k[1304]*y[IDX_NII] - k[1305]*y[IDX_NII];
    data[2459] = 0.0 + k[1260]*y[IDX_HeII];
    data[2460] = 0.0 - k[1289]*y[IDX_NII] - k[1290]*y[IDX_NII] - k[1291]*y[IDX_NII] -
        k[1292]*y[IDX_NII];
    data[2461] = 0.0 - k[244]*y[IDX_NII];
    data[2462] = 0.0 - k[250]*y[IDX_NII] - k[1312]*y[IDX_NII] - k[1313]*y[IDX_NII];
    data[2463] = 0.0 + k[259]*y[IDX_NI];
    data[2464] = 0.0 - k[1296]*y[IDX_NII];
    data[2465] = 0.0 - k[245]*y[IDX_NII] + k[1252]*y[IDX_HeII];
    data[2466] = 0.0 - k[87]*y[IDX_CHI] - k[233]*y[IDX_C2I] - k[234]*y[IDX_C2HI] -
        k[235]*y[IDX_CH2I] - k[236]*y[IDX_CH4I] - k[237]*y[IDX_CNI] -
        k[238]*y[IDX_COI] - k[239]*y[IDX_H2COI] - k[240]*y[IDX_H2OI] -
        k[241]*y[IDX_H2SI] - k[242]*y[IDX_HCNI] - k[243]*y[IDX_HCOI] -
        k[244]*y[IDX_MgI] - k[245]*y[IDX_NH2I] - k[246]*y[IDX_NH3I] -
        k[247]*y[IDX_NHI] - k[248]*y[IDX_NOI] - k[249]*y[IDX_O2I] -
        k[250]*y[IDX_OCSI] - k[251]*y[IDX_OHI] - k[852]*y[IDX_CHI] -
        k[952]*y[IDX_H2I] - k[1289]*y[IDX_CH3OHI] - k[1290]*y[IDX_CH3OHI] -
        k[1291]*y[IDX_CH3OHI] - k[1292]*y[IDX_CH3OHI] - k[1293]*y[IDX_CH4I] -
        k[1294]*y[IDX_CH4I] - k[1295]*y[IDX_CH4I] - k[1296]*y[IDX_CO2I] -
        k[1297]*y[IDX_COI] - k[1298]*y[IDX_H2COI] - k[1299]*y[IDX_H2COI] -
        k[1300]*y[IDX_H2SI] - k[1301]*y[IDX_H2SI] - k[1302]*y[IDX_H2SI] -
        k[1303]*y[IDX_HCOI] - k[1304]*y[IDX_NCCNI] - k[1305]*y[IDX_NCCNI] -
        k[1306]*y[IDX_NH3I] - k[1307]*y[IDX_NH3I] - k[1308]*y[IDX_NHI] -
        k[1309]*y[IDX_NOI] - k[1310]*y[IDX_O2I] - k[1311]*y[IDX_O2I] -
        k[1312]*y[IDX_OCSI] - k[1313]*y[IDX_OCSI] - k[2127]*y[IDX_NI] -
        k[2141]*y[IDX_EM] - k[2226];
    data[2467] = 0.0 - k[234]*y[IDX_NII];
    data[2468] = 0.0 - k[241]*y[IDX_NII] - k[1300]*y[IDX_NII] - k[1301]*y[IDX_NII] -
        k[1302]*y[IDX_NII];
    data[2469] = 0.0 + k[1250]*y[IDX_HeII];
    data[2470] = 0.0 - k[233]*y[IDX_NII];
    data[2471] = 0.0 - k[235]*y[IDX_NII];
    data[2472] = 0.0 - k[247]*y[IDX_NII] + k[1256]*y[IDX_HeII] - k[1308]*y[IDX_NII];
    data[2473] = 0.0 - k[237]*y[IDX_NII] + k[1207]*y[IDX_HeII];
    data[2474] = 0.0 - k[239]*y[IDX_NII] - k[1298]*y[IDX_NII] - k[1299]*y[IDX_NII];
    data[2475] = 0.0 - k[236]*y[IDX_NII] - k[1293]*y[IDX_NII] - k[1294]*y[IDX_NII] -
        k[1295]*y[IDX_NII];
    data[2476] = 0.0 - k[243]*y[IDX_NII] - k[1303]*y[IDX_NII];
    data[2477] = 0.0 - k[242]*y[IDX_NII] + k[1232]*y[IDX_HeII];
    data[2478] = 0.0 - k[248]*y[IDX_NII] + k[1258]*y[IDX_HeII] - k[1309]*y[IDX_NII];
    data[2479] = 0.0 - k[87]*y[IDX_NII] - k[852]*y[IDX_NII];
    data[2480] = 0.0 - k[246]*y[IDX_NII] - k[1306]*y[IDX_NII] - k[1307]*y[IDX_NII];
    data[2481] = 0.0 - k[249]*y[IDX_NII] - k[1310]*y[IDX_NII] - k[1311]*y[IDX_NII];
    data[2482] = 0.0 - k[251]*y[IDX_NII];
    data[2483] = 0.0 + k[259]*y[IDX_N2II] + k[364] + k[422] - k[2127]*y[IDX_NII];
    data[2484] = 0.0 + k[1207]*y[IDX_CNI] + k[1232]*y[IDX_HCNI] + k[1250]*y[IDX_N2I] +
        k[1252]*y[IDX_NH2I] + k[1256]*y[IDX_NHI] + k[1258]*y[IDX_NOI] +
        k[1260]*y[IDX_NSI];
    data[2485] = 0.0 - k[240]*y[IDX_NII];
    data[2486] = 0.0 - k[238]*y[IDX_NII] - k[1297]*y[IDX_NII];
    data[2487] = 0.0 - k[952]*y[IDX_NII];
    data[2488] = 0.0 - k[2141]*y[IDX_NII];
    data[2489] = 0.0 + k[900]*y[IDX_HII];
    data[2490] = 0.0 + k[271]*y[IDX_NH2I];
    data[2491] = 0.0 + k[1459]*y[IDX_NHI];
    data[2492] = 0.0 + k[272]*y[IDX_NH2I];
    data[2493] = 0.0 + k[1454]*y[IDX_NHI];
    data[2494] = 0.0 + k[1453]*y[IDX_NHI];
    data[2495] = 0.0 + k[1447]*y[IDX_NHI];
    data[2496] = 0.0 + k[275]*y[IDX_NH2I];
    data[2497] = 0.0 + k[166]*y[IDX_NH2I] + k[929]*y[IDX_NHI];
    data[2498] = 0.0 + k[1451]*y[IDX_NHI];
    data[2499] = 0.0 - k[1387]*y[IDX_NH2II];
    data[2500] = 0.0 + k[273]*y[IDX_NH2I];
    data[2501] = 0.0 + k[955]*y[IDX_H2I] + k[1359]*y[IDX_H2OI] + k[1366]*y[IDX_NHI];
    data[2502] = 0.0 + k[130]*y[IDX_HII] + k[166]*y[IDX_H2II] + k[245]*y[IDX_NII] +
        k[271]*y[IDX_C2II] + k[272]*y[IDX_CNII] + k[273]*y[IDX_COII] +
        k[274]*y[IDX_H2OII] + k[275]*y[IDX_N2II] + k[276]*y[IDX_O2II] +
        k[277]*y[IDX_OHII] + k[315]*y[IDX_OII] + k[424] - k[1388]*y[IDX_NH2II] +
        k[2049];
    data[2503] = 0.0 + k[276]*y[IDX_NH2I];
    data[2504] = 0.0 + k[245]*y[IDX_NH2I] + k[1307]*y[IDX_NH3I];
    data[2505] = 0.0 - k[68]*y[IDX_CH2I] - k[89]*y[IDX_CHI] - k[266]*y[IDX_H2SI] -
        k[267]*y[IDX_HCOI] - k[268]*y[IDX_NH3I] - k[269]*y[IDX_NOI] -
        k[270]*y[IDX_SI] - k[568]*y[IDX_EM] - k[569]*y[IDX_EM] -
        k[767]*y[IDX_CH2I] - k[855]*y[IDX_CHI] - k[956]*y[IDX_H2I] -
        k[1338]*y[IDX_NI] - k[1374]*y[IDX_C2I] - k[1375]*y[IDX_C2HI] -
        k[1376]*y[IDX_H2COI] - k[1377]*y[IDX_H2COI] - k[1378]*y[IDX_H2OI] -
        k[1379]*y[IDX_H2OI] - k[1380]*y[IDX_H2OI] - k[1381]*y[IDX_H2SI] -
        k[1382]*y[IDX_H2SI] - k[1383]*y[IDX_H2SI] - k[1384]*y[IDX_H2SI] -
        k[1385]*y[IDX_HCNI] - k[1386]*y[IDX_HCOI] - k[1387]*y[IDX_HNCI] -
        k[1388]*y[IDX_NH2I] - k[1389]*y[IDX_NH3I] - k[1390]*y[IDX_O2I] -
        k[1391]*y[IDX_O2I] - k[1392]*y[IDX_SI] - k[1393]*y[IDX_SI] -
        k[1455]*y[IDX_NHI] - k[1507]*y[IDX_OI] - k[2238];
    data[2506] = 0.0 - k[1375]*y[IDX_NH2II];
    data[2507] = 0.0 + k[315]*y[IDX_NH2I];
    data[2508] = 0.0 - k[266]*y[IDX_NH2II] - k[1381]*y[IDX_NH2II] - k[1382]*y[IDX_NH2II]
        - k[1383]*y[IDX_NH2II] - k[1384]*y[IDX_NH2II];
    data[2509] = 0.0 + k[274]*y[IDX_NH2I];
    data[2510] = 0.0 + k[277]*y[IDX_NH2I] + k[1460]*y[IDX_NHI];
    data[2511] = 0.0 - k[1374]*y[IDX_NH2II];
    data[2512] = 0.0 - k[68]*y[IDX_NH2II] - k[767]*y[IDX_NH2II];
    data[2513] = 0.0 + k[929]*y[IDX_H2II] + k[1058]*y[IDX_H3II] + k[1366]*y[IDX_NHII] +
        k[1447]*y[IDX_CH5II] + k[1451]*y[IDX_HCNII] + k[1452]*y[IDX_HCOII] +
        k[1453]*y[IDX_HNOII] + k[1454]*y[IDX_N2HII] - k[1455]*y[IDX_NH2II] +
        k[1459]*y[IDX_O2HII] + k[1460]*y[IDX_OHII];
    data[2514] = 0.0 - k[1376]*y[IDX_NH2II] - k[1377]*y[IDX_NH2II];
    data[2515] = 0.0 - k[267]*y[IDX_NH2II] - k[1386]*y[IDX_NH2II];
    data[2516] = 0.0 - k[1385]*y[IDX_NH2II];
    data[2517] = 0.0 - k[269]*y[IDX_NH2II];
    data[2518] = 0.0 - k[89]*y[IDX_NH2II] - k[855]*y[IDX_NH2II];
    data[2519] = 0.0 - k[268]*y[IDX_NH2II] + k[1255]*y[IDX_HeII] + k[1307]*y[IDX_NII] -
        k[1389]*y[IDX_NH2II];
    data[2520] = 0.0 - k[1390]*y[IDX_NH2II] - k[1391]*y[IDX_NH2II];
    data[2521] = 0.0 - k[270]*y[IDX_NH2II] - k[1392]*y[IDX_NH2II] - k[1393]*y[IDX_NH2II];
    data[2522] = 0.0 - k[1338]*y[IDX_NH2II];
    data[2523] = 0.0 + k[1255]*y[IDX_NH3I];
    data[2524] = 0.0 + k[1058]*y[IDX_NHI];
    data[2525] = 0.0 + k[130]*y[IDX_NH2I] + k[900]*y[IDX_HNCOI];
    data[2526] = 0.0 - k[1507]*y[IDX_NH2II];
    data[2527] = 0.0 + k[1452]*y[IDX_NHI];
    data[2528] = 0.0 + k[1359]*y[IDX_NHII] - k[1378]*y[IDX_NH2II] - k[1379]*y[IDX_NH2II]
        - k[1380]*y[IDX_NH2II];
    data[2529] = 0.0 + k[955]*y[IDX_NHII] - k[956]*y[IDX_NH2II];
    data[2530] = 0.0 - k[568]*y[IDX_NH2II] - k[569]*y[IDX_NH2II];
    data[2531] = 0.0 + k[1191]*y[IDX_HeII];
    data[2532] = 0.0 + k[1230]*y[IDX_HeII];
    data[2533] = 0.0 + k[1181]*y[IDX_HeII];
    data[2534] = 0.0 + k[1183]*y[IDX_HeII];
    data[2535] = 0.0 + k[648]*y[IDX_HCOI] + k[803]*y[IDX_CH4I] + k[935]*y[IDX_H2I] +
        k[990]*y[IDX_H2OI] + k[1444]*y[IDX_NHI];
    data[2536] = 0.0 + k[657]*y[IDX_C2I];
    data[2537] = 0.0 + k[47]*y[IDX_C2HI];
    data[2538] = 0.0 + k[655]*y[IDX_C2I];
    data[2539] = 0.0 + k[654]*y[IDX_C2I];
    data[2540] = 0.0 + k[823]*y[IDX_C2I];
    data[2541] = 0.0 + k[49]*y[IDX_C2HI];
    data[2542] = 0.0 + k[155]*y[IDX_C2HI] + k[909]*y[IDX_C2I];
    data[2543] = 0.0 + k[1179]*y[IDX_HeII];
    data[2544] = 0.0 + k[652]*y[IDX_C2I];
    data[2545] = 0.0 - k[664]*y[IDX_C2HII];
    data[2546] = 0.0 + k[48]*y[IDX_C2HI];
    data[2547] = 0.0 + k[1345]*y[IDX_C2I];
    data[2548] = 0.0 - k[1395]*y[IDX_C2HII];
    data[2549] = 0.0 + k[234]*y[IDX_C2HI];
    data[2550] = 0.0 + k[1374]*y[IDX_C2I];
    data[2551] = 0.0 - k[40]*y[IDX_NOI] - k[41]*y[IDX_SI] - k[459]*y[IDX_EM] -
        k[460]*y[IDX_EM] - k[660]*y[IDX_H2COI] - k[661]*y[IDX_HCNI] -
        k[662]*y[IDX_HCNI] - k[663]*y[IDX_HCOI] - k[664]*y[IDX_HNCI] -
        k[685]*y[IDX_CI] - k[754]*y[IDX_CH2I] - k[805]*y[IDX_CH4I] -
        k[838]*y[IDX_CHI] - k[936]*y[IDX_H2I] - k[1326]*y[IDX_NI] -
        k[1327]*y[IDX_NI] - k[1395]*y[IDX_NH2I] - k[1418]*y[IDX_NH3I] -
        k[1491]*y[IDX_OI] - k[1963] - k[2242];
    data[2552] = 0.0 + k[47]*y[IDX_CNII] + k[48]*y[IDX_COII] + k[49]*y[IDX_N2II] +
        k[112]*y[IDX_HII] + k[155]*y[IDX_H2II] + k[177]*y[IDX_H2OII] +
        k[234]*y[IDX_NII] + k[308]*y[IDX_OII] + k[330]*y[IDX_OHII] + k[374] +
        k[1971];
    data[2553] = 0.0 + k[308]*y[IDX_C2HI];
    data[2554] = 0.0 + k[651]*y[IDX_C2I];
    data[2555] = 0.0 + k[687]*y[IDX_CI];
    data[2556] = 0.0 + k[177]*y[IDX_C2HI] + k[976]*y[IDX_C2I];
    data[2557] = 0.0 + k[330]*y[IDX_C2HI] + k[1517]*y[IDX_C2I];
    data[2558] = 0.0 + k[688]*y[IDX_CI];
    data[2559] = 0.0 + k[651]*y[IDX_H2COII] + k[652]*y[IDX_HCNII] + k[653]*y[IDX_HCOII] +
        k[654]*y[IDX_HNOII] + k[655]*y[IDX_N2HII] + k[657]*y[IDX_O2HII] +
        k[823]*y[IDX_CH5II] + k[909]*y[IDX_H2II] + k[976]*y[IDX_H2OII] +
        k[1022]*y[IDX_H3II] + k[1080]*y[IDX_H3OII] + k[1345]*y[IDX_NHII] +
        k[1374]*y[IDX_NH2II] + k[1517]*y[IDX_OHII];
    data[2560] = 0.0 + k[608]*y[IDX_CII] + k[707]*y[IDX_CHII] - k[754]*y[IDX_C2HII];
    data[2561] = 0.0 + k[609]*y[IDX_CII];
    data[2562] = 0.0 + k[707]*y[IDX_CH2I];
    data[2563] = 0.0 + k[1444]*y[IDX_C2II];
    data[2564] = 0.0 - k[660]*y[IDX_C2HII];
    data[2565] = 0.0 + k[803]*y[IDX_C2II] - k[805]*y[IDX_C2HII];
    data[2566] = 0.0 + k[648]*y[IDX_C2II] - k[663]*y[IDX_C2HII];
    data[2567] = 0.0 - k[661]*y[IDX_C2HII] - k[662]*y[IDX_C2HII];
    data[2568] = 0.0 - k[40]*y[IDX_C2HII];
    data[2569] = 0.0 - k[838]*y[IDX_C2HII];
    data[2570] = 0.0 - k[1418]*y[IDX_C2HII];
    data[2571] = 0.0 + k[1080]*y[IDX_C2I];
    data[2572] = 0.0 + k[608]*y[IDX_CH2I] + k[609]*y[IDX_CH3I];
    data[2573] = 0.0 - k[41]*y[IDX_C2HII];
    data[2574] = 0.0 - k[1326]*y[IDX_C2HII] - k[1327]*y[IDX_C2HII];
    data[2575] = 0.0 + k[1179]*y[IDX_C2H2I] + k[1181]*y[IDX_C2H3I] + k[1183]*y[IDX_C2H4I]
        + k[1191]*y[IDX_C4HI] + k[1230]*y[IDX_HC3NI];
    data[2576] = 0.0 + k[1022]*y[IDX_C2I];
    data[2577] = 0.0 + k[112]*y[IDX_C2HI];
    data[2578] = 0.0 - k[1491]*y[IDX_C2HII];
    data[2579] = 0.0 - k[685]*y[IDX_C2HII] + k[687]*y[IDX_CH2II] + k[688]*y[IDX_CH3II];
    data[2580] = 0.0 + k[653]*y[IDX_C2I];
    data[2581] = 0.0 + k[990]*y[IDX_C2II];
    data[2582] = 0.0 + k[935]*y[IDX_C2II] - k[936]*y[IDX_C2HII];
    data[2583] = 0.0 - k[459]*y[IDX_C2HII] - k[460]*y[IDX_C2HII];
    data[2584] = 0.0 + k[1420]*y[IDX_NH3I];
    data[2585] = 0.0 + k[1438]*y[IDX_NH3I];
    data[2586] = 0.0 + k[1439]*y[IDX_NH3I];
    data[2587] = 0.0 + k[792]*y[IDX_NH3I];
    data[2588] = 0.0 + k[1443]*y[IDX_NH3I];
    data[2589] = 0.0 + k[1442]*y[IDX_NH3I];
    data[2590] = 0.0 + k[1434]*y[IDX_NH3I];
    data[2591] = 0.0 + k[1435]*y[IDX_NH3I];
    data[2592] = 0.0 + k[1429]*y[IDX_NH3I];
    data[2593] = 0.0 + k[801]*y[IDX_NH3I];
    data[2594] = 0.0 + k[1441]*y[IDX_NH3I];
    data[2595] = 0.0 + k[1440]*y[IDX_NH3I];
    data[2596] = 0.0 + k[1426]*y[IDX_NH3I];
    data[2597] = 0.0 + k[1436]*y[IDX_NH3I];
    data[2598] = 0.0 + k[1422]*y[IDX_NH3I];
    data[2599] = 0.0 + k[1431]*y[IDX_NH3I] + k[1432]*y[IDX_NH3I];
    data[2600] = 0.0 + k[1427]*y[IDX_NH3I];
    data[2601] = 0.0 + k[1437]*y[IDX_NH3I];
    data[2602] = 0.0 + k[1365]*y[IDX_NH3I];
    data[2603] = 0.0 + k[1409]*y[IDX_NH3II];
    data[2604] = 0.0 + k[1380]*y[IDX_H2OI] + k[1384]*y[IDX_H2SI] + k[1389]*y[IDX_NH3I];
    data[2605] = 0.0 + k[1418]*y[IDX_NH3I];
    data[2606] = 0.0 - k[572]*y[IDX_EM] - k[573]*y[IDX_EM] - k[574]*y[IDX_EM] - k[2288];
    data[2607] = 0.0 + k[1424]*y[IDX_NH3I];
    data[2608] = 0.0 + k[748]*y[IDX_NH3I];
    data[2609] = 0.0 + k[818]*y[IDX_CH4I] + k[856]*y[IDX_CHI] + k[957]*y[IDX_H2I] +
        k[1409]*y[IDX_NH2I] + k[1413]*y[IDX_H2COI] + k[1414]*y[IDX_H2OI] +
        k[1415]*y[IDX_H2SI] + k[1416]*y[IDX_HCOI] + k[1417]*y[IDX_NH3I] +
        k[1456]*y[IDX_NHI] + k[1546]*y[IDX_OHI];
    data[2610] = 0.0 + k[1384]*y[IDX_NH2II] + k[1415]*y[IDX_NH3II];
    data[2611] = 0.0 + k[1425]*y[IDX_NH3I];
    data[2612] = 0.0 + k[1530]*y[IDX_NH3I];
    data[2613] = 0.0 + k[781]*y[IDX_NH3I];
    data[2614] = 0.0 + k[1419]*y[IDX_NH3I];
    data[2615] = 0.0 + k[730]*y[IDX_NH3I];
    data[2616] = 0.0 + k[1456]*y[IDX_NH3II];
    data[2617] = 0.0 + k[1413]*y[IDX_NH3II];
    data[2618] = 0.0 + k[818]*y[IDX_NH3II];
    data[2619] = 0.0 + k[1416]*y[IDX_NH3II];
    data[2620] = 0.0 + k[856]*y[IDX_NH3II];
    data[2621] = 0.0 + k[730]*y[IDX_CHII] + k[748]*y[IDX_CH2II] + k[781]*y[IDX_CH3II] +
        k[792]*y[IDX_CH3OH2II] + k[801]*y[IDX_CH4II] + k[1057]*y[IDX_H3II] +
        k[1365]*y[IDX_NHII] + k[1389]*y[IDX_NH2II] + k[1417]*y[IDX_NH3II] +
        k[1418]*y[IDX_C2HII] + k[1419]*y[IDX_C2H2II] + k[1420]*y[IDX_C2H5OH2II]
        + k[1422]*y[IDX_CH5II] + k[1424]*y[IDX_H2COII] + k[1425]*y[IDX_H2OII] +
        k[1426]*y[IDX_H2SII] + k[1427]*y[IDX_H3COII] + k[1428]*y[IDX_H3OII] +
        k[1429]*y[IDX_H3SII] + k[1431]*y[IDX_HCNHII] + k[1432]*y[IDX_HCNHII] +
        k[1433]*y[IDX_HCOII] + k[1434]*y[IDX_HCO2II] + k[1435]*y[IDX_HCSII] +
        k[1436]*y[IDX_HNOII] + k[1437]*y[IDX_HSII] + k[1438]*y[IDX_HSO2II] +
        k[1439]*y[IDX_HSiSII] + k[1440]*y[IDX_N2HII] + k[1441]*y[IDX_O2HII] +
        k[1442]*y[IDX_SiHII] + k[1443]*y[IDX_SiOHII] + k[1530]*y[IDX_OHII];
    data[2622] = 0.0 + k[1428]*y[IDX_NH3I];
    data[2623] = 0.0 + k[1546]*y[IDX_NH3II];
    data[2624] = 0.0 + k[1057]*y[IDX_NH3I];
    data[2625] = 0.0 + k[1433]*y[IDX_NH3I];
    data[2626] = 0.0 + k[1380]*y[IDX_NH2II] + k[1414]*y[IDX_NH3II];
    data[2627] = 0.0 + k[957]*y[IDX_NH3II];
    data[2628] = 0.0 - k[572]*y[IDX_NH4II] - k[573]*y[IDX_NH4II] - k[574]*y[IDX_NH4II];
    data[2629] = 0.0 + k[2323] + k[2324] + k[2325] + k[2326];
    data[2630] = 0.0 + k[378] + k[1975];
    data[2631] = 0.0 + k[407] + k[623]*y[IDX_CII] + k[2027];
    data[2632] = 0.0 - k[1580]*y[IDX_C2HI];
    data[2633] = 0.0 + k[666]*y[IDX_C2H2II];
    data[2634] = 0.0 - k[683]*y[IDX_C2HI];
    data[2635] = 0.0 - k[47]*y[IDX_C2HI];
    data[2636] = 0.0 - k[682]*y[IDX_C2HI];
    data[2637] = 0.0 - k[681]*y[IDX_C2HI];
    data[2638] = 0.0 - k[824]*y[IDX_C2HI];
    data[2639] = 0.0 - k[49]*y[IDX_C2HI];
    data[2640] = 0.0 - k[155]*y[IDX_C2HI] - k[911]*y[IDX_C2HI];
    data[2641] = 0.0 + k[368] + k[1737]*y[IDX_HI] + k[1927]*y[IDX_OHI] + k[1965];
    data[2642] = 0.0 - k[679]*y[IDX_C2HI];
    data[2643] = 0.0 + k[669]*y[IDX_C2H2II] - k[1579]*y[IDX_C2HI];
    data[2644] = 0.0 - k[684]*y[IDX_C2HI];
    data[2645] = 0.0 - k[48]*y[IDX_C2HI] - k[677]*y[IDX_C2HI];
    data[2646] = 0.0 - k[1348]*y[IDX_C2HI];
    data[2647] = 0.0 + k[1396]*y[IDX_C2H2II];
    data[2648] = 0.0 - k[234]*y[IDX_C2HI];
    data[2649] = 0.0 - k[1375]*y[IDX_C2HI];
    data[2650] = 0.0 + k[40]*y[IDX_NOI] + k[41]*y[IDX_SI];
    data[2651] = 0.0 - k[47]*y[IDX_CNII] - k[48]*y[IDX_COII] - k[49]*y[IDX_N2II] -
        k[112]*y[IDX_HII] - k[155]*y[IDX_H2II] - k[177]*y[IDX_H2OII] -
        k[234]*y[IDX_NII] - k[308]*y[IDX_OII] - k[330]*y[IDX_OHII] - k[373] -
        k[374] - k[607]*y[IDX_CII] - k[677]*y[IDX_COII] - k[678]*y[IDX_H2COII] -
        k[679]*y[IDX_HCNII] - k[680]*y[IDX_HCOII] - k[681]*y[IDX_HNOII] -
        k[682]*y[IDX_N2HII] - k[683]*y[IDX_O2HII] - k[684]*y[IDX_SiII] -
        k[706]*y[IDX_CHII] - k[824]*y[IDX_CH5II] - k[884]*y[IDX_HII] -
        k[911]*y[IDX_H2II] - k[977]*y[IDX_H2OII] - k[1025]*y[IDX_H3II] -
        k[1186]*y[IDX_HeII] - k[1187]*y[IDX_HeII] - k[1188]*y[IDX_HeII] -
        k[1348]*y[IDX_NHII] - k[1375]*y[IDX_NH2II] - k[1465]*y[IDX_OII] -
        k[1518]*y[IDX_OHII] - k[1578]*y[IDX_HCNI] - k[1579]*y[IDX_HNCI] -
        k[1580]*y[IDX_NCCNI] - k[1581]*y[IDX_O2I] - k[1721]*y[IDX_H2I] -
        k[1795]*y[IDX_NI] - k[1875]*y[IDX_OI] - k[1970] - k[1971] -
        k[2100]*y[IDX_CNI] - k[2224];
    data[2652] = 0.0 - k[308]*y[IDX_C2HI] - k[1465]*y[IDX_C2HI];
    data[2653] = 0.0 - k[678]*y[IDX_C2HI];
    data[2654] = 0.0 + k[667]*y[IDX_C2H2II];
    data[2655] = 0.0 - k[177]*y[IDX_C2HI] - k[977]*y[IDX_C2HI];
    data[2656] = 0.0 - k[330]*y[IDX_C2HI] - k[1518]*y[IDX_C2HI];
    data[2657] = 0.0 + k[462]*y[IDX_EM] + k[666]*y[IDX_CH3CNI] + k[667]*y[IDX_H2SI] +
        k[668]*y[IDX_HCNI] + k[669]*y[IDX_HNCI] + k[991]*y[IDX_H2OI] +
        k[1396]*y[IDX_NH2I] + k[1419]*y[IDX_NH3I];
    data[2658] = 0.0 + k[1586]*y[IDX_CI];
    data[2659] = 0.0 - k[706]*y[IDX_C2HI];
    data[2660] = 0.0 - k[2100]*y[IDX_C2HI];
    data[2661] = 0.0 + k[668]*y[IDX_C2H2II] - k[1578]*y[IDX_C2HI];
    data[2662] = 0.0 + k[40]*y[IDX_C2HII];
    data[2663] = 0.0 + k[1419]*y[IDX_C2H2II];
    data[2664] = 0.0 - k[1581]*y[IDX_C2HI];
    data[2665] = 0.0 - k[607]*y[IDX_C2HI] + k[623]*y[IDX_HC3NI];
    data[2666] = 0.0 + k[41]*y[IDX_C2HII];
    data[2667] = 0.0 + k[1927]*y[IDX_C2H2I];
    data[2668] = 0.0 - k[1795]*y[IDX_C2HI];
    data[2669] = 0.0 - k[1186]*y[IDX_C2HI] - k[1187]*y[IDX_C2HI] - k[1188]*y[IDX_C2HI];
    data[2670] = 0.0 - k[1025]*y[IDX_C2HI];
    data[2671] = 0.0 - k[112]*y[IDX_C2HI] - k[884]*y[IDX_C2HI];
    data[2672] = 0.0 - k[1875]*y[IDX_C2HI];
    data[2673] = 0.0 + k[1586]*y[IDX_CH2I];
    data[2674] = 0.0 - k[680]*y[IDX_C2HI];
    data[2675] = 0.0 + k[991]*y[IDX_C2H2II];
    data[2676] = 0.0 - k[1721]*y[IDX_C2HI];
    data[2677] = 0.0 + k[462]*y[IDX_C2H2II];
    data[2678] = 0.0 + k[1737]*y[IDX_C2H2I];
    data[2679] = 0.0 - k[1478]*y[IDX_OII];
    data[2680] = 0.0 + k[1263]*y[IDX_HeII];
    data[2681] = 0.0 - k[320]*y[IDX_OII] - k[1481]*y[IDX_OII];
    data[2682] = 0.0 + k[1286]*y[IDX_HeII];
    data[2683] = 0.0 - k[1466]*y[IDX_OII] - k[1467]*y[IDX_OII];
    data[2684] = 0.0 + k[1273]*y[IDX_HeII];
    data[2685] = 0.0 - k[1464]*y[IDX_OII];
    data[2686] = 0.0 + k[326]*y[IDX_OI];
    data[2687] = 0.0 - k[318]*y[IDX_OII] + k[1265]*y[IDX_HeII] - k[1479]*y[IDX_OII];
    data[2688] = 0.0 + k[328]*y[IDX_OI];
    data[2689] = 0.0 - k[307]*y[IDX_OII];
    data[2690] = 0.0 + k[327]*y[IDX_OI];
    data[2691] = 0.0 + k[1210]*y[IDX_HeII] - k[1470]*y[IDX_OII];
    data[2692] = 0.0 - k[315]*y[IDX_OII];
    data[2693] = 0.0 + k[2060];
    data[2694] = 0.0 + k[1311]*y[IDX_O2I];
    data[2695] = 0.0 - k[308]*y[IDX_OII] - k[1465]*y[IDX_OII];
    data[2696] = 0.0 - k[69]*y[IDX_CH2I] - k[90]*y[IDX_CHI] - k[196]*y[IDX_HI] -
        k[297]*y[IDX_NHI] - k[306]*y[IDX_C2I] - k[307]*y[IDX_C2H2I] -
        k[308]*y[IDX_C2HI] - k[309]*y[IDX_CH4I] - k[310]*y[IDX_COI] -
        k[311]*y[IDX_H2COI] - k[312]*y[IDX_H2OI] - k[313]*y[IDX_H2SI] -
        k[314]*y[IDX_HCOI] - k[315]*y[IDX_NH2I] - k[316]*y[IDX_NH3I] -
        k[317]*y[IDX_O2I] - k[318]*y[IDX_OCSI] - k[319]*y[IDX_OHI] -
        k[320]*y[IDX_SO2I] - k[857]*y[IDX_CHI] - k[958]*y[IDX_H2I] -
        k[1457]*y[IDX_NHI] - k[1463]*y[IDX_C2I] - k[1464]*y[IDX_C2H4I] -
        k[1465]*y[IDX_C2HI] - k[1466]*y[IDX_CH3OHI] - k[1467]*y[IDX_CH3OHI] -
        k[1468]*y[IDX_CH4I] - k[1469]*y[IDX_CNI] - k[1470]*y[IDX_CO2I] -
        k[1471]*y[IDX_H2COI] - k[1472]*y[IDX_H2SI] - k[1473]*y[IDX_H2SI] -
        k[1474]*y[IDX_HCNI] - k[1475]*y[IDX_HCNI] - k[1476]*y[IDX_HCOI] -
        k[1477]*y[IDX_N2I] - k[1478]*y[IDX_NO2I] - k[1479]*y[IDX_OCSI] -
        k[1480]*y[IDX_OHI] - k[1481]*y[IDX_SO2I] - k[2103]*y[IDX_CI] -
        k[2142]*y[IDX_EM] - k[2227];
    data[2697] = 0.0 - k[313]*y[IDX_OII] - k[1472]*y[IDX_OII] - k[1473]*y[IDX_OII];
    data[2698] = 0.0 + k[2068];
    data[2699] = 0.0 - k[1477]*y[IDX_OII];
    data[2700] = 0.0 - k[306]*y[IDX_OII] - k[1463]*y[IDX_OII];
    data[2701] = 0.0 - k[69]*y[IDX_OII];
    data[2702] = 0.0 + k[734]*y[IDX_O2I];
    data[2703] = 0.0 - k[297]*y[IDX_OII] - k[1457]*y[IDX_OII];
    data[2704] = 0.0 - k[1469]*y[IDX_OII];
    data[2705] = 0.0 - k[311]*y[IDX_OII] - k[1471]*y[IDX_OII];
    data[2706] = 0.0 - k[309]*y[IDX_OII] - k[1468]*y[IDX_OII];
    data[2707] = 0.0 - k[314]*y[IDX_OII] - k[1476]*y[IDX_OII];
    data[2708] = 0.0 - k[1474]*y[IDX_OII] - k[1475]*y[IDX_OII];
    data[2709] = 0.0 + k[1257]*y[IDX_HeII];
    data[2710] = 0.0 - k[90]*y[IDX_OII] - k[857]*y[IDX_OII];
    data[2711] = 0.0 - k[316]*y[IDX_OII];
    data[2712] = 0.0 - k[317]*y[IDX_OII] + k[634]*y[IDX_CII] + k[734]*y[IDX_CHII] +
        k[1261]*y[IDX_HeII] + k[1311]*y[IDX_NII];
    data[2713] = 0.0 + k[634]*y[IDX_O2I];
    data[2714] = 0.0 - k[319]*y[IDX_OII] + k[1268]*y[IDX_HeII] - k[1480]*y[IDX_OII];
    data[2715] = 0.0 + k[1210]*y[IDX_CO2I] + k[1257]*y[IDX_NOI] + k[1261]*y[IDX_O2I] +
        k[1263]*y[IDX_OCNI] + k[1265]*y[IDX_OCSI] + k[1268]*y[IDX_OHI] +
        k[1273]*y[IDX_SOI] + k[1286]*y[IDX_SiOI];
    data[2716] = 0.0 + k[136]*y[IDX_OI];
    data[2717] = 0.0 + k[136]*y[IDX_HII] + k[326]*y[IDX_CNII] + k[327]*y[IDX_COII] +
        k[328]*y[IDX_N2II] + k[365] + k[438];
    data[2718] = 0.0 - k[2103]*y[IDX_OII];
    data[2719] = 0.0 - k[312]*y[IDX_OII];
    data[2720] = 0.0 - k[310]*y[IDX_OII];
    data[2721] = 0.0 - k[958]*y[IDX_OII];
    data[2722] = 0.0 - k[2142]*y[IDX_OII];
    data[2723] = 0.0 - k[196]*y[IDX_OII];
    data[2724] = 0.0 + k[1509]*y[IDX_OI];
    data[2725] = 0.0 + k[903]*y[IDX_HII] + k[1059]*y[IDX_H3II] + k[1478]*y[IDX_OII];
    data[2726] = 0.0 + k[304]*y[IDX_NOI];
    data[2727] = 0.0 + k[901]*y[IDX_HII] + k[1245]*y[IDX_HeII];
    data[2728] = 0.0 + k[305]*y[IDX_NOI] + k[1343]*y[IDX_NI];
    data[2729] = 0.0 + k[1291]*y[IDX_NII];
    data[2730] = 0.0 - k[227]*y[IDX_NOII];
    data[2731] = 0.0 + k[34]*y[IDX_NOI];
    data[2732] = 0.0 + k[97]*y[IDX_NOI] + k[868]*y[IDX_O2I];
    data[2733] = 0.0 + k[299]*y[IDX_NOI];
    data[2734] = 0.0 + k[300]*y[IDX_NOI];
    data[2735] = 0.0 + k[255]*y[IDX_NOI] + k[1505]*y[IDX_OI];
    data[2736] = 0.0 + k[169]*y[IDX_NOI];
    data[2737] = 0.0 - k[352]*y[IDX_NOII];
    data[2738] = 0.0 + k[197]*y[IDX_NOI];
    data[2739] = 0.0 + k[104]*y[IDX_NOI];
    data[2740] = 0.0 + k[301]*y[IDX_NOI];
    data[2741] = 0.0 + k[1352]*y[IDX_NHII];
    data[2742] = 0.0 + k[263]*y[IDX_NOI] + k[1352]*y[IDX_CO2I] + k[1368]*y[IDX_O2I];
    data[2743] = 0.0 + k[302]*y[IDX_NOI] + k[1339]*y[IDX_NI];
    data[2744] = 0.0 + k[248]*y[IDX_NOI] + k[1291]*y[IDX_CH3OHI] + k[1297]*y[IDX_COI] +
        k[1299]*y[IDX_H2COI] + k[1310]*y[IDX_O2I];
    data[2745] = 0.0 + k[269]*y[IDX_NOI];
    data[2746] = 0.0 + k[40]*y[IDX_NOI];
    data[2747] = 0.0 + k[1457]*y[IDX_NHI] + k[1469]*y[IDX_CNI] + k[1475]*y[IDX_HCNI] +
        k[1477]*y[IDX_N2I] + k[1478]*y[IDX_NO2I];
    data[2748] = 0.0 - k[227]*y[IDX_MgI] - k[352]*y[IDX_SiI] - k[575]*y[IDX_EM] - k[2236];
    data[2749] = 0.0 + k[298]*y[IDX_NOI];
    data[2750] = 0.0 + k[61]*y[IDX_NOI];
    data[2751] = 0.0 + k[280]*y[IDX_NOI];
    data[2752] = 0.0 + k[182]*y[IDX_NOI] + k[1334]*y[IDX_NI];
    data[2753] = 0.0 + k[336]*y[IDX_NOI] + k[1340]*y[IDX_NI];
    data[2754] = 0.0 + k[74]*y[IDX_NOI];
    data[2755] = 0.0 + k[1477]*y[IDX_OII];
    data[2756] = 0.0 + k[45]*y[IDX_NOI];
    data[2757] = 0.0 + k[58]*y[IDX_NOI];
    data[2758] = 0.0 + k[1457]*y[IDX_OII];
    data[2759] = 0.0 + k[303]*y[IDX_NOI];
    data[2760] = 0.0 + k[1469]*y[IDX_OII];
    data[2761] = 0.0 + k[1299]*y[IDX_NII];
    data[2762] = 0.0 + k[1475]*y[IDX_OII];
    data[2763] = 0.0 + k[22]*y[IDX_CII] + k[34]*y[IDX_C2II] + k[40]*y[IDX_C2HII] +
        k[45]*y[IDX_C2H2II] + k[58]*y[IDX_CHII] + k[61]*y[IDX_CH2II] +
        k[74]*y[IDX_CH3II] + k[97]*y[IDX_CNII] + k[104]*y[IDX_COII] +
        k[133]*y[IDX_HII] + k[169]*y[IDX_H2II] + k[182]*y[IDX_H2OII] +
        k[197]*y[IDX_HCNII] + k[248]*y[IDX_NII] + k[255]*y[IDX_N2II] +
        k[263]*y[IDX_NHII] + k[269]*y[IDX_NH2II] + k[280]*y[IDX_NH3II] +
        k[298]*y[IDX_H2COII] + k[299]*y[IDX_H2SII] + k[300]*y[IDX_HNOII] +
        k[301]*y[IDX_HSII] + k[302]*y[IDX_O2II] + k[303]*y[IDX_SII] +
        k[304]*y[IDX_S2II] + k[305]*y[IDX_SiOII] + k[336]*y[IDX_OHII] + k[432] +
        k[2057];
    data[2764] = 0.0 + k[868]*y[IDX_CNII] + k[1310]*y[IDX_NII] + k[1368]*y[IDX_NHII];
    data[2765] = 0.0 + k[22]*y[IDX_NOI];
    data[2766] = 0.0 + k[1334]*y[IDX_H2OII] + k[1339]*y[IDX_O2II] + k[1340]*y[IDX_OHII] +
        k[1343]*y[IDX_SiOII];
    data[2767] = 0.0 + k[1245]*y[IDX_HNOI];
    data[2768] = 0.0 + k[1059]*y[IDX_NO2I];
    data[2769] = 0.0 + k[133]*y[IDX_NOI] + k[901]*y[IDX_HNOI] + k[903]*y[IDX_NO2I];
    data[2770] = 0.0 + k[1505]*y[IDX_N2II] + k[1509]*y[IDX_NSII];
    data[2771] = 0.0 + k[1297]*y[IDX_NII];
    data[2772] = 0.0 - k[575]*y[IDX_NOII];
    data[2773] = 0.0 - k[965]*y[IDX_H2COII] + k[1289]*y[IDX_NII] + k[1466]*y[IDX_OII];
    data[2774] = 0.0 + k[76]*y[IDX_H2COI];
    data[2775] = 0.0 - k[222]*y[IDX_H2COII];
    data[2776] = 0.0 + k[1162]*y[IDX_HCOI];
    data[2777] = 0.0 + k[94]*y[IDX_H2COI];
    data[2778] = 0.0 + k[1160]*y[IDX_HCOI];
    data[2779] = 0.0 + k[1159]*y[IDX_HCOI];
    data[2780] = 0.0 + k[831]*y[IDX_HCOI];
    data[2781] = 0.0 + k[252]*y[IDX_H2COI];
    data[2782] = 0.0 + k[161]*y[IDX_H2COI];
    data[2783] = 0.0 - k[349]*y[IDX_H2COII];
    data[2784] = 0.0 + k[1114]*y[IDX_HCOI];
    data[2785] = 0.0 - k[1165]*y[IDX_H2COII];
    data[2786] = 0.0 + k[101]*y[IDX_H2COI];
    data[2787] = 0.0 + k[741]*y[IDX_CH2II];
    data[2788] = 0.0 + k[260]*y[IDX_H2COI] + k[1361]*y[IDX_HCOI];
    data[2789] = 0.0 - k[1399]*y[IDX_H2COII];
    data[2790] = 0.0 + k[174]*y[IDX_H2COI] + k[769]*y[IDX_CH2I];
    data[2791] = 0.0 + k[239]*y[IDX_H2COI] + k[1289]*y[IDX_CH3OHI];
    data[2792] = 0.0 + k[1386]*y[IDX_HCOI];
    data[2793] = 0.0 - k[678]*y[IDX_H2COII];
    data[2794] = 0.0 + k[311]*y[IDX_H2COI] + k[1466]*y[IDX_CH3OHI];
    data[2795] = 0.0 - k[65]*y[IDX_CH2I] - k[85]*y[IDX_CHI] - k[173]*y[IDX_SI] -
        k[202]*y[IDX_HCOI] - k[222]*y[IDX_MgI] - k[284]*y[IDX_NH3I] -
        k[298]*y[IDX_NOI] - k[349]*y[IDX_SiI] - k[502]*y[IDX_EM] -
        k[503]*y[IDX_EM] - k[504]*y[IDX_EM] - k[505]*y[IDX_EM] -
        k[651]*y[IDX_C2I] - k[678]*y[IDX_C2HI] - k[757]*y[IDX_CH2I] -
        k[809]*y[IDX_CH4I] - k[842]*y[IDX_CHI] - k[965]*y[IDX_CH3OHI] -
        k[966]*y[IDX_H2COI] - k[967]*y[IDX_O2I] - k[968]*y[IDX_SI] -
        k[998]*y[IDX_H2OI] - k[1121]*y[IDX_HCNI] - k[1158]*y[IDX_HCOI] -
        k[1165]*y[IDX_HNCI] - k[1399]*y[IDX_NH2I] - k[1424]*y[IDX_NH3I] -
        k[1449]*y[IDX_NHI] - k[2136]*y[IDX_EM] - k[2244];
    data[2796] = 0.0 + k[741]*y[IDX_CO2I];
    data[2797] = 0.0 + k[178]*y[IDX_H2COI] + k[985]*y[IDX_HCOI];
    data[2798] = 0.0 + k[331]*y[IDX_H2COI] + k[1527]*y[IDX_HCOI];
    data[2799] = 0.0 + k[783]*y[IDX_OI] + k[786]*y[IDX_OHI];
    data[2800] = 0.0 + k[42]*y[IDX_H2COI];
    data[2801] = 0.0 - k[651]*y[IDX_H2COII];
    data[2802] = 0.0 - k[65]*y[IDX_H2COII] - k[757]*y[IDX_H2COII] + k[769]*y[IDX_O2II];
    data[2803] = 0.0 + k[718]*y[IDX_H2OI];
    data[2804] = 0.0 - k[1449]*y[IDX_H2COII];
    data[2805] = 0.0 + k[16]*y[IDX_CII] + k[42]*y[IDX_C2H2II] + k[76]*y[IDX_CH4II] +
        k[94]*y[IDX_CNII] + k[101]*y[IDX_COII] + k[119]*y[IDX_HII] +
        k[161]*y[IDX_H2II] + k[174]*y[IDX_O2II] + k[178]*y[IDX_H2OII] +
        k[212]*y[IDX_HeII] + k[239]*y[IDX_NII] + k[252]*y[IDX_N2II] +
        k[260]*y[IDX_NHII] + k[311]*y[IDX_OII] + k[331]*y[IDX_OHII] -
        k[966]*y[IDX_H2COII] + k[2013];
    data[2806] = 0.0 - k[809]*y[IDX_H2COII];
    data[2807] = 0.0 - k[202]*y[IDX_H2COII] + k[831]*y[IDX_CH5II] + k[985]*y[IDX_H2OII] +
        k[1046]*y[IDX_H3II] + k[1114]*y[IDX_HCNII] + k[1144]*y[IDX_HCOII] -
        k[1158]*y[IDX_H2COII] + k[1159]*y[IDX_HNOII] + k[1160]*y[IDX_N2HII] +
        k[1162]*y[IDX_O2HII] + k[1361]*y[IDX_NHII] + k[1386]*y[IDX_NH2II] +
        k[1527]*y[IDX_OHII];
    data[2808] = 0.0 - k[1121]*y[IDX_H2COII];
    data[2809] = 0.0 - k[298]*y[IDX_H2COII];
    data[2810] = 0.0 - k[85]*y[IDX_H2COII] - k[842]*y[IDX_H2COII];
    data[2811] = 0.0 - k[284]*y[IDX_H2COII] - k[1424]*y[IDX_H2COII];
    data[2812] = 0.0 - k[967]*y[IDX_H2COII];
    data[2813] = 0.0 + k[16]*y[IDX_H2COI];
    data[2814] = 0.0 - k[173]*y[IDX_H2COII] - k[968]*y[IDX_H2COII];
    data[2815] = 0.0 + k[786]*y[IDX_CH3II];
    data[2816] = 0.0 + k[212]*y[IDX_H2COI];
    data[2817] = 0.0 + k[1046]*y[IDX_HCOI];
    data[2818] = 0.0 + k[119]*y[IDX_H2COI];
    data[2819] = 0.0 + k[783]*y[IDX_CH3II];
    data[2820] = 0.0 + k[1144]*y[IDX_HCOI];
    data[2821] = 0.0 + k[718]*y[IDX_CHII] - k[998]*y[IDX_H2COII];
    data[2822] = 0.0 - k[502]*y[IDX_H2COII] - k[503]*y[IDX_H2COII] - k[504]*y[IDX_H2COII]
        - k[505]*y[IDX_H2COII] - k[2136]*y[IDX_H2COII];
    data[2823] = 0.0 + k[1195]*y[IDX_HeII];
    data[2824] = 0.0 + k[619]*y[IDX_CII] + k[1221]*y[IDX_HeII];
    data[2825] = 0.0 + k[863]*y[IDX_CHI];
    data[2826] = 0.0 + k[1185]*y[IDX_HeII];
    data[2827] = 0.0 + k[1993];
    data[2828] = 0.0 + k[62]*y[IDX_CH2I];
    data[2829] = 0.0 + k[859]*y[IDX_CHI];
    data[2830] = 0.0 + k[63]*y[IDX_CH2I];
    data[2831] = 0.0 + k[853]*y[IDX_CHI];
    data[2832] = 0.0 - k[751]*y[IDX_CH2II] - k[752]*y[IDX_CH2II];
    data[2833] = 0.0 + k[850]*y[IDX_CHI];
    data[2834] = 0.0 + k[840]*y[IDX_CHI];
    data[2835] = 0.0 + k[67]*y[IDX_CH2I] + k[815]*y[IDX_CH4I];
    data[2836] = 0.0 + k[847]*y[IDX_CHI] + k[848]*y[IDX_CHI];
    data[2837] = 0.0 + k[156]*y[IDX_CH2I] + k[916]*y[IDX_CHI];
    data[2838] = 0.0 + k[844]*y[IDX_CHI];
    data[2839] = 0.0 + k[846]*y[IDX_CHI];
    data[2840] = 0.0 + k[64]*y[IDX_CH2I];
    data[2841] = 0.0 + k[851]*y[IDX_CHI];
    data[2842] = 0.0 - k[741]*y[IDX_CH2II];
    data[2843] = 0.0 + k[854]*y[IDX_CHI];
    data[2844] = 0.0 + k[70]*y[IDX_CH2I];
    data[2845] = 0.0 + k[235]*y[IDX_CH2I];
    data[2846] = 0.0 + k[68]*y[IDX_CH2I] + k[855]*y[IDX_CHI];
    data[2847] = 0.0 + k[838]*y[IDX_CHI];
    data[2848] = 0.0 + k[69]*y[IDX_CH2I];
    data[2849] = 0.0 + k[65]*y[IDX_CH2I] + k[842]*y[IDX_CHI];
    data[2850] = 0.0 - k[61]*y[IDX_NOI] - k[478]*y[IDX_EM] - k[479]*y[IDX_EM] -
        k[480]*y[IDX_EM] - k[687]*y[IDX_CI] - k[741]*y[IDX_CO2I] -
        k[742]*y[IDX_H2COI] - k[743]*y[IDX_H2OI] - k[744]*y[IDX_H2SI] -
        k[745]*y[IDX_H2SI] - k[746]*y[IDX_H2SI] - k[747]*y[IDX_HCOI] -
        k[748]*y[IDX_NH3I] - k[749]*y[IDX_O2I] - k[750]*y[IDX_OI] -
        k[751]*y[IDX_OCSI] - k[752]*y[IDX_OCSI] - k[753]*y[IDX_SI] -
        k[938]*y[IDX_H2I] - k[1099]*y[IDX_HI] - k[1331]*y[IDX_NI] - k[1978] -
        k[1979] - k[1980] - k[2237];
    data[2851] = 0.0 - k[744]*y[IDX_CH2II] - k[745]*y[IDX_CH2II] - k[746]*y[IDX_CH2II];
    data[2852] = 0.0 + k[66]*y[IDX_CH2I] + k[843]*y[IDX_CHI];
    data[2853] = 0.0 + k[71]*y[IDX_CH2I] + k[860]*y[IDX_CHI];
    data[2854] = 0.0 + k[1100]*y[IDX_HI] + k[1985];
    data[2855] = 0.0 + k[14]*y[IDX_CII] + k[62]*y[IDX_C2II] + k[63]*y[IDX_CNII] +
        k[64]*y[IDX_COII] + k[65]*y[IDX_H2COII] + k[66]*y[IDX_H2OII] +
        k[67]*y[IDX_N2II] + k[68]*y[IDX_NH2II] + k[69]*y[IDX_OII] +
        k[70]*y[IDX_O2II] + k[71]*y[IDX_OHII] + k[114]*y[IDX_HII] +
        k[156]*y[IDX_H2II] + k[235]*y[IDX_NII] + k[381] + k[1981];
    data[2856] = 0.0 + k[726]*y[IDX_HCOI] + k[937]*y[IDX_H2I];
    data[2857] = 0.0 + k[617]*y[IDX_CII] - k[742]*y[IDX_CH2II] + k[1218]*y[IDX_HeII];
    data[2858] = 0.0 + k[815]*y[IDX_N2II] + k[1203]*y[IDX_HeII];
    data[2859] = 0.0 + k[726]*y[IDX_CHII] - k[747]*y[IDX_CH2II];
    data[2860] = 0.0 - k[61]*y[IDX_CH2II];
    data[2861] = 0.0 + k[838]*y[IDX_C2HII] + k[840]*y[IDX_CH5II] + k[842]*y[IDX_H2COII] +
        k[843]*y[IDX_H2OII] + k[844]*y[IDX_H3COII] + k[845]*y[IDX_H3OII] +
        k[846]*y[IDX_HCNII] + k[847]*y[IDX_HCNHII] + k[848]*y[IDX_HCNHII] +
        k[849]*y[IDX_HCOII] + k[850]*y[IDX_HNOII] + k[851]*y[IDX_HSII] +
        k[853]*y[IDX_N2HII] + k[854]*y[IDX_NHII] + k[855]*y[IDX_NH2II] +
        k[859]*y[IDX_O2HII] + k[860]*y[IDX_OHII] + k[863]*y[IDX_SiHII] +
        k[916]*y[IDX_H2II] + k[1034]*y[IDX_H3II];
    data[2862] = 0.0 - k[748]*y[IDX_CH2II];
    data[2863] = 0.0 + k[845]*y[IDX_CHI];
    data[2864] = 0.0 - k[749]*y[IDX_CH2II];
    data[2865] = 0.0 + k[14]*y[IDX_CH2I] + k[617]*y[IDX_H2COI] + k[619]*y[IDX_H2CSI] +
        k[2112]*y[IDX_H2I];
    data[2866] = 0.0 - k[753]*y[IDX_CH2II];
    data[2867] = 0.0 - k[1331]*y[IDX_CH2II];
    data[2868] = 0.0 + k[1185]*y[IDX_C2H4I] + k[1195]*y[IDX_CH2COI] + k[1203]*y[IDX_CH4I]
        + k[1218]*y[IDX_H2COI] + k[1221]*y[IDX_H2CSI];
    data[2869] = 0.0 + k[1034]*y[IDX_CHI];
    data[2870] = 0.0 + k[114]*y[IDX_CH2I];
    data[2871] = 0.0 - k[750]*y[IDX_CH2II];
    data[2872] = 0.0 - k[687]*y[IDX_CH2II];
    data[2873] = 0.0 + k[849]*y[IDX_CHI];
    data[2874] = 0.0 - k[743]*y[IDX_CH2II];
    data[2875] = 0.0 + k[937]*y[IDX_CHII] - k[938]*y[IDX_CH2II] + k[2112]*y[IDX_CII];
    data[2876] = 0.0 - k[478]*y[IDX_CH2II] - k[479]*y[IDX_CH2II] - k[480]*y[IDX_CH2II];
    data[2877] = 0.0 - k[1099]*y[IDX_CH2II] + k[1100]*y[IDX_CH3II];
    data[2878] = 0.0 + k[291]*y[IDX_NH3I];
    data[2879] = 0.0 + k[78]*y[IDX_NH3I];
    data[2880] = 0.0 - k[279]*y[IDX_NH3II];
    data[2881] = 0.0 + k[293]*y[IDX_NH3I];
    data[2882] = 0.0 + k[1410]*y[IDX_NH2I];
    data[2883] = 0.0 + k[1408]*y[IDX_NH2I];
    data[2884] = 0.0 + k[286]*y[IDX_NH3I];
    data[2885] = 0.0 + k[1407]*y[IDX_NH2I];
    data[2886] = 0.0 + k[1397]*y[IDX_NH2I];
    data[2887] = 0.0 + k[289]*y[IDX_NH3I];
    data[2888] = 0.0 + k[1404]*y[IDX_NH2I] + k[1405]*y[IDX_NH2I];
    data[2889] = 0.0 + k[167]*y[IDX_NH3I];
    data[2890] = 0.0 + k[1401]*y[IDX_NH2I];
    data[2891] = 0.0 - k[281]*y[IDX_NH3II];
    data[2892] = 0.0 + k[287]*y[IDX_NH3I] + k[1403]*y[IDX_NH2I];
    data[2893] = 0.0 + k[283]*y[IDX_NH3I];
    data[2894] = 0.0 + k[288]*y[IDX_NH3I];
    data[2895] = 0.0 + k[262]*y[IDX_NH3I] + k[1358]*y[IDX_H2OI] + k[1364]*y[IDX_NH2I];
    data[2896] = 0.0 + k[1056]*y[IDX_H3II] + k[1364]*y[IDX_NHII] + k[1388]*y[IDX_NH2II] +
        k[1395]*y[IDX_C2HII] + k[1396]*y[IDX_C2H2II] + k[1397]*y[IDX_CH5II] +
        k[1399]*y[IDX_H2COII] + k[1400]*y[IDX_H2OII] + k[1401]*y[IDX_H3COII] +
        k[1402]*y[IDX_H3OII] + k[1403]*y[IDX_HCNII] + k[1404]*y[IDX_HCNHII] +
        k[1405]*y[IDX_HCNHII] + k[1406]*y[IDX_HCOII] + k[1407]*y[IDX_HNOII] +
        k[1408]*y[IDX_N2HII] - k[1409]*y[IDX_NH3II] + k[1410]*y[IDX_O2HII] +
        k[1411]*y[IDX_OHII];
    data[2897] = 0.0 + k[290]*y[IDX_NH3I];
    data[2898] = 0.0 + k[246]*y[IDX_NH3I];
    data[2899] = 0.0 + k[268]*y[IDX_NH3I] + k[956]*y[IDX_H2I] + k[1377]*y[IDX_H2COI] +
        k[1379]*y[IDX_H2OI] + k[1383]*y[IDX_H2SI] + k[1388]*y[IDX_NH2I] +
        k[1455]*y[IDX_NHI];
    data[2900] = 0.0 + k[1395]*y[IDX_NH2I];
    data[2901] = 0.0 + k[316]*y[IDX_NH3I];
    data[2902] = 0.0 + k[284]*y[IDX_NH3I] + k[1399]*y[IDX_NH2I];
    data[2903] = 0.0 - k[278]*y[IDX_HCOI] - k[279]*y[IDX_MgI] - k[280]*y[IDX_NOI] -
        k[281]*y[IDX_SiI] - k[570]*y[IDX_EM] - k[571]*y[IDX_EM] -
        k[768]*y[IDX_CH2I] - k[818]*y[IDX_CH4I] - k[856]*y[IDX_CHI] -
        k[957]*y[IDX_H2I] - k[1409]*y[IDX_NH2I] - k[1412]*y[IDX_C2I] -
        k[1413]*y[IDX_H2COI] - k[1414]*y[IDX_H2OI] - k[1415]*y[IDX_H2SI] -
        k[1416]*y[IDX_HCOI] - k[1417]*y[IDX_NH3I] - k[1456]*y[IDX_NHI] -
        k[1508]*y[IDX_OI] - k[1546]*y[IDX_OHI] - k[2243];
    data[2904] = 0.0 + k[1383]*y[IDX_NH2II] - k[1415]*y[IDX_NH3II];
    data[2905] = 0.0 + k[285]*y[IDX_NH3I] + k[1400]*y[IDX_NH2I];
    data[2906] = 0.0 + k[335]*y[IDX_NH3I] + k[1411]*y[IDX_NH2I];
    data[2907] = 0.0 + k[282]*y[IDX_NH3I] + k[1396]*y[IDX_NH2I];
    data[2908] = 0.0 - k[1412]*y[IDX_NH3II];
    data[2909] = 0.0 - k[768]*y[IDX_NH3II];
    data[2910] = 0.0 + k[57]*y[IDX_NH3I];
    data[2911] = 0.0 + k[1455]*y[IDX_NH2II] - k[1456]*y[IDX_NH3II];
    data[2912] = 0.0 + k[292]*y[IDX_NH3I];
    data[2913] = 0.0 + k[1377]*y[IDX_NH2II] - k[1413]*y[IDX_NH3II];
    data[2914] = 0.0 - k[818]*y[IDX_NH3II];
    data[2915] = 0.0 - k[278]*y[IDX_NH3II] - k[1416]*y[IDX_NH3II];
    data[2916] = 0.0 - k[280]*y[IDX_NH3II];
    data[2917] = 0.0 - k[856]*y[IDX_NH3II];
    data[2918] = 0.0 + k[21]*y[IDX_CII] + k[57]*y[IDX_CHII] + k[78]*y[IDX_CH4II] +
        k[131]*y[IDX_HII] + k[167]*y[IDX_H2II] + k[216]*y[IDX_HeII] +
        k[246]*y[IDX_NII] + k[262]*y[IDX_NHII] + k[268]*y[IDX_NH2II] +
        k[282]*y[IDX_C2H2II] + k[283]*y[IDX_COII] + k[284]*y[IDX_H2COII] +
        k[285]*y[IDX_H2OII] + k[286]*y[IDX_H2SII] + k[287]*y[IDX_HCNII] +
        k[288]*y[IDX_HSII] + k[289]*y[IDX_N2II] + k[290]*y[IDX_O2II] +
        k[291]*y[IDX_OCSII] + k[292]*y[IDX_SII] + k[293]*y[IDX_SOII] +
        k[316]*y[IDX_OII] + k[335]*y[IDX_OHII] + k[427] - k[1417]*y[IDX_NH3II] +
        k[2052];
    data[2919] = 0.0 + k[1402]*y[IDX_NH2I];
    data[2920] = 0.0 + k[21]*y[IDX_NH3I];
    data[2921] = 0.0 - k[1546]*y[IDX_NH3II];
    data[2922] = 0.0 + k[216]*y[IDX_NH3I];
    data[2923] = 0.0 + k[1056]*y[IDX_NH2I];
    data[2924] = 0.0 + k[131]*y[IDX_NH3I];
    data[2925] = 0.0 - k[1508]*y[IDX_NH3II];
    data[2926] = 0.0 + k[1406]*y[IDX_NH2I];
    data[2927] = 0.0 + k[1358]*y[IDX_NHII] + k[1379]*y[IDX_NH2II] - k[1414]*y[IDX_NH3II];
    data[2928] = 0.0 + k[956]*y[IDX_NH2II] - k[957]*y[IDX_NH3II];
    data[2929] = 0.0 - k[570]*y[IDX_NH3II] - k[571]*y[IDX_NH3II];
    data[2930] = 0.0 + k[2391] + k[2392] + k[2393] + k[2394];
    data[2931] = 0.0 + k[1009]*y[IDX_H2OI] - k[1020]*y[IDX_H2SI];
    data[2932] = 0.0 - k[1019]*y[IDX_H2SI];
    data[2933] = 0.0 - k[190]*y[IDX_H2SI];
    data[2934] = 0.0 - k[1018]*y[IDX_H2SI];
    data[2935] = 0.0 + k[532]*y[IDX_EM] + k[970]*y[IDX_H2COI] + k[1123]*y[IDX_HCNI] +
        k[1167]*y[IDX_HNCI] + k[1429]*y[IDX_NH3I];
    data[2936] = 0.0 - k[77]*y[IDX_H2SI] - k[800]*y[IDX_H2SI];
    data[2937] = 0.0 + k[223]*y[IDX_H2SII];
    data[2938] = 0.0 - k[1021]*y[IDX_H2SI];
    data[2939] = 0.0 + k[203]*y[IDX_HCOI] + k[223]*y[IDX_MgI] + k[286]*y[IDX_NH3I] +
        k[299]*y[IDX_NOI] + k[346]*y[IDX_SI] + k[350]*y[IDX_SiI] -
        k[1017]*y[IDX_H2SI] + k[2138]*y[IDX_EM];
    data[2940] = 0.0 - k[829]*y[IDX_H2SI];
    data[2941] = 0.0 - k[253]*y[IDX_H2SI] - k[1315]*y[IDX_H2SI] - k[1316]*y[IDX_H2SI];
    data[2942] = 0.0 + k[1727]*y[IDX_H2I] + k[1789]*y[IDX_HSI] + k[1789]*y[IDX_HSI];
    data[2943] = 0.0 - k[1134]*y[IDX_H2SI] - k[1135]*y[IDX_H2SI];
    data[2944] = 0.0 - k[163]*y[IDX_H2SI] - k[923]*y[IDX_H2SI] - k[924]*y[IDX_H2SI];
    data[2945] = 0.0 - k[1079]*y[IDX_H2SI];
    data[2946] = 0.0 + k[350]*y[IDX_H2SII];
    data[2947] = 0.0 + k[1167]*y[IDX_H3SII];
    data[2948] = 0.0 - k[102]*y[IDX_H2SI] - k[872]*y[IDX_H2SI];
    data[2949] = 0.0 - k[1175]*y[IDX_H2SI] - k[1176]*y[IDX_H2SI];
    data[2950] = 0.0 - k[322]*y[IDX_H2SI];
    data[2951] = 0.0 - k[241]*y[IDX_H2SI] - k[1300]*y[IDX_H2SI] - k[1301]*y[IDX_H2SI] -
        k[1302]*y[IDX_H2SI];
    data[2952] = 0.0 - k[266]*y[IDX_H2SI] - k[1381]*y[IDX_H2SI] - k[1382]*y[IDX_H2SI] -
        k[1383]*y[IDX_H2SI] - k[1384]*y[IDX_H2SI];
    data[2953] = 0.0 - k[313]*y[IDX_H2SI] - k[1472]*y[IDX_H2SI] - k[1473]*y[IDX_H2SI];
    data[2954] = 0.0 - k[744]*y[IDX_H2SI] - k[745]*y[IDX_H2SI] - k[746]*y[IDX_H2SI];
    data[2955] = 0.0 - k[1415]*y[IDX_H2SI];
    data[2956] = 0.0 - k[17]*y[IDX_CII] - k[43]*y[IDX_C2H2II] - k[77]*y[IDX_CH4II] -
        k[102]*y[IDX_COII] - k[123]*y[IDX_HII] - k[163]*y[IDX_H2II] -
        k[179]*y[IDX_H2OII] - k[190]*y[IDX_OCSII] - k[214]*y[IDX_HeII] -
        k[241]*y[IDX_NII] - k[253]*y[IDX_N2II] - k[266]*y[IDX_NH2II] -
        k[313]*y[IDX_OII] - k[322]*y[IDX_O2II] - k[333]*y[IDX_OHII] - k[403] -
        k[404] - k[622]*y[IDX_CII] - k[667]*y[IDX_C2H2II] - k[721]*y[IDX_CHII] -
        k[722]*y[IDX_CHII] - k[744]*y[IDX_CH2II] - k[745]*y[IDX_CH2II] -
        k[746]*y[IDX_CH2II] - k[778]*y[IDX_CH3II] - k[800]*y[IDX_CH4II] -
        k[829]*y[IDX_CH5II] - k[872]*y[IDX_COII] - k[894]*y[IDX_HII] -
        k[895]*y[IDX_HII] - k[923]*y[IDX_H2II] - k[924]*y[IDX_H2II] -
        k[981]*y[IDX_H2OII] - k[982]*y[IDX_H2OII] - k[1017]*y[IDX_H2SII] -
        k[1018]*y[IDX_C2NII] - k[1019]*y[IDX_HS2II] - k[1020]*y[IDX_HSiSII] -
        k[1021]*y[IDX_SOII] - k[1044]*y[IDX_H3II] - k[1079]*y[IDX_H3COII] -
        k[1087]*y[IDX_H3OII] - k[1134]*y[IDX_HCNHII] - k[1135]*y[IDX_HCNHII] -
        k[1143]*y[IDX_HCOII] - k[1175]*y[IDX_HSII] - k[1176]*y[IDX_HSII] -
        k[1226]*y[IDX_HeII] - k[1227]*y[IDX_HeII] - k[1300]*y[IDX_NII] -
        k[1301]*y[IDX_NII] - k[1302]*y[IDX_NII] - k[1315]*y[IDX_N2II] -
        k[1316]*y[IDX_N2II] - k[1381]*y[IDX_NH2II] - k[1382]*y[IDX_NH2II] -
        k[1383]*y[IDX_NH2II] - k[1384]*y[IDX_NH2II] - k[1415]*y[IDX_NH3II] -
        k[1472]*y[IDX_OII] - k[1473]*y[IDX_OII] - k[1524]*y[IDX_OHII] -
        k[1550]*y[IDX_SII] - k[1551]*y[IDX_SII] - k[1653]*y[IDX_CH3I] -
        k[1749]*y[IDX_HI] - k[1888]*y[IDX_OI] - k[1938]*y[IDX_OHI] - k[2020] -
        k[2021] - k[2022] - k[2257];
    data[2957] = 0.0 - k[179]*y[IDX_H2SI] - k[981]*y[IDX_H2SI] - k[982]*y[IDX_H2SI];
    data[2958] = 0.0 - k[333]*y[IDX_H2SI] - k[1524]*y[IDX_H2SI];
    data[2959] = 0.0 - k[778]*y[IDX_H2SI];
    data[2960] = 0.0 - k[43]*y[IDX_H2SI] - k[667]*y[IDX_H2SI];
    data[2961] = 0.0 - k[1653]*y[IDX_H2SI];
    data[2962] = 0.0 - k[721]*y[IDX_H2SI] - k[722]*y[IDX_H2SI];
    data[2963] = 0.0 - k[1550]*y[IDX_H2SI] - k[1551]*y[IDX_H2SI];
    data[2964] = 0.0 + k[970]*y[IDX_H3SII];
    data[2965] = 0.0 + k[203]*y[IDX_H2SII];
    data[2966] = 0.0 + k[1123]*y[IDX_H3SII];
    data[2967] = 0.0 + k[299]*y[IDX_H2SII];
    data[2968] = 0.0 + k[286]*y[IDX_H2SII] + k[1429]*y[IDX_H3SII];
    data[2969] = 0.0 - k[1087]*y[IDX_H2SI];
    data[2970] = 0.0 - k[17]*y[IDX_H2SI] - k[622]*y[IDX_H2SI];
    data[2971] = 0.0 + k[346]*y[IDX_H2SII];
    data[2972] = 0.0 - k[1938]*y[IDX_H2SI];
    data[2973] = 0.0 - k[214]*y[IDX_H2SI] - k[1226]*y[IDX_H2SI] - k[1227]*y[IDX_H2SI];
    data[2974] = 0.0 - k[1044]*y[IDX_H2SI];
    data[2975] = 0.0 - k[123]*y[IDX_H2SI] - k[894]*y[IDX_H2SI] - k[895]*y[IDX_H2SI];
    data[2976] = 0.0 - k[1888]*y[IDX_H2SI];
    data[2977] = 0.0 - k[1143]*y[IDX_H2SI];
    data[2978] = 0.0 + k[1009]*y[IDX_HSiSII];
    data[2979] = 0.0 + k[1727]*y[IDX_HSI];
    data[2980] = 0.0 + k[532]*y[IDX_H3SII] + k[2138]*y[IDX_H2SII];
    data[2981] = 0.0 - k[1749]*y[IDX_H2SI];
    data[2982] = 0.0 - k[989]*y[IDX_H2OII];
    data[2983] = 0.0 - k[181]*y[IDX_H2OII];
    data[2984] = 0.0 + k[1547]*y[IDX_OHI];
    data[2985] = 0.0 + k[1545]*y[IDX_OHI];
    data[2986] = 0.0 - k[184]*y[IDX_H2OII];
    data[2987] = 0.0 + k[1544]*y[IDX_OHI];
    data[2988] = 0.0 + k[1538]*y[IDX_OHI];
    data[2989] = 0.0 + k[189]*y[IDX_H2OI];
    data[2990] = 0.0 + k[162]*y[IDX_H2OI] + k[933]*y[IDX_OHI];
    data[2991] = 0.0 - k[186]*y[IDX_H2OII];
    data[2992] = 0.0 - k[176]*y[IDX_H2OII];
    data[2993] = 0.0 + k[188]*y[IDX_H2OI] + k[1541]*y[IDX_OHI];
    data[2994] = 0.0 - k[986]*y[IDX_H2OII];
    data[2995] = 0.0 + k[187]*y[IDX_H2OI];
    data[2996] = 0.0 + k[261]*y[IDX_H2OI] + k[1371]*y[IDX_OHI];
    data[2997] = 0.0 - k[274]*y[IDX_H2OII] - k[1400]*y[IDX_H2OII];
    data[2998] = 0.0 + k[240]*y[IDX_H2OI];
    data[2999] = 0.0 - k[177]*y[IDX_H2OII] - k[977]*y[IDX_H2OII];
    data[3000] = 0.0 + k[312]*y[IDX_H2OI];
    data[3001] = 0.0 - k[179]*y[IDX_H2OII] - k[981]*y[IDX_H2OII] - k[982]*y[IDX_H2OII];
    data[3002] = 0.0 - k[66]*y[IDX_CH2I] - k[86]*y[IDX_CHI] - k[175]*y[IDX_C2I] -
        k[176]*y[IDX_C2H2I] - k[177]*y[IDX_C2HI] - k[178]*y[IDX_H2COI] -
        k[179]*y[IDX_H2SI] - k[180]*y[IDX_HCOI] - k[181]*y[IDX_MgI] -
        k[182]*y[IDX_NOI] - k[183]*y[IDX_O2I] - k[184]*y[IDX_OCSI] -
        k[185]*y[IDX_SI] - k[186]*y[IDX_SiI] - k[274]*y[IDX_NH2I] -
        k[285]*y[IDX_NH3I] - k[512]*y[IDX_EM] - k[513]*y[IDX_EM] -
        k[514]*y[IDX_EM] - k[690]*y[IDX_CI] - k[758]*y[IDX_CH2I] -
        k[810]*y[IDX_CH4I] - k[843]*y[IDX_CHI] - k[945]*y[IDX_H2I] -
        k[976]*y[IDX_C2I] - k[977]*y[IDX_C2HI] - k[978]*y[IDX_COI] -
        k[979]*y[IDX_H2COI] - k[980]*y[IDX_H2OI] - k[981]*y[IDX_H2SI] -
        k[982]*y[IDX_H2SI] - k[983]*y[IDX_HCNI] - k[984]*y[IDX_HCOI] -
        k[985]*y[IDX_HCOI] - k[986]*y[IDX_HNCI] - k[987]*y[IDX_SI] -
        k[988]*y[IDX_SI] - k[989]*y[IDX_SO2I] - k[1333]*y[IDX_NI] -
        k[1334]*y[IDX_NI] - k[1400]*y[IDX_NH2I] - k[1425]*y[IDX_NH3I] -
        k[1450]*y[IDX_NHI] - k[1497]*y[IDX_OI] - k[1540]*y[IDX_OHI] - k[2016] -
        k[2239];
    data[3003] = 0.0 + k[332]*y[IDX_H2OI] + k[960]*y[IDX_H2I] + k[1526]*y[IDX_HCOI] +
        k[1532]*y[IDX_OHI];
    data[3004] = 0.0 - k[175]*y[IDX_H2OII] - k[976]*y[IDX_H2OII];
    data[3005] = 0.0 - k[66]*y[IDX_H2OII] - k[758]*y[IDX_H2OII];
    data[3006] = 0.0 - k[1450]*y[IDX_H2OII];
    data[3007] = 0.0 - k[178]*y[IDX_H2OII] - k[979]*y[IDX_H2OII];
    data[3008] = 0.0 - k[810]*y[IDX_H2OII];
    data[3009] = 0.0 - k[180]*y[IDX_H2OII] - k[984]*y[IDX_H2OII] - k[985]*y[IDX_H2OII] +
        k[1526]*y[IDX_OHII];
    data[3010] = 0.0 - k[983]*y[IDX_H2OII];
    data[3011] = 0.0 - k[182]*y[IDX_H2OII];
    data[3012] = 0.0 - k[86]*y[IDX_H2OII] - k[843]*y[IDX_H2OII];
    data[3013] = 0.0 - k[285]*y[IDX_H2OII] - k[1425]*y[IDX_H2OII];
    data[3014] = 0.0 - k[183]*y[IDX_H2OII];
    data[3015] = 0.0 - k[185]*y[IDX_H2OII] - k[987]*y[IDX_H2OII] - k[988]*y[IDX_H2OII];
    data[3016] = 0.0 + k[933]*y[IDX_H2II] + k[1066]*y[IDX_H3II] + k[1371]*y[IDX_NHII] +
        k[1532]*y[IDX_OHII] + k[1538]*y[IDX_CH5II] - k[1540]*y[IDX_H2OII] +
        k[1541]*y[IDX_HCNII] + k[1542]*y[IDX_HCOII] + k[1544]*y[IDX_HNOII] +
        k[1545]*y[IDX_N2HII] + k[1547]*y[IDX_O2HII];
    data[3017] = 0.0 - k[1333]*y[IDX_H2OII] - k[1334]*y[IDX_H2OII];
    data[3018] = 0.0 + k[213]*y[IDX_H2OI];
    data[3019] = 0.0 + k[1063]*y[IDX_OI] + k[1066]*y[IDX_OHI];
    data[3020] = 0.0 + k[121]*y[IDX_H2OI];
    data[3021] = 0.0 + k[1063]*y[IDX_H3II] - k[1497]*y[IDX_H2OII];
    data[3022] = 0.0 - k[690]*y[IDX_H2OII];
    data[3023] = 0.0 + k[1542]*y[IDX_OHI];
    data[3024] = 0.0 + k[121]*y[IDX_HII] + k[162]*y[IDX_H2II] + k[187]*y[IDX_COII] +
        k[188]*y[IDX_HCNII] + k[189]*y[IDX_N2II] + k[213]*y[IDX_HeII] +
        k[240]*y[IDX_NII] + k[261]*y[IDX_NHII] + k[312]*y[IDX_OII] +
        k[332]*y[IDX_OHII] - k[980]*y[IDX_H2OII] + k[2017];
    data[3025] = 0.0 - k[978]*y[IDX_H2OII];
    data[3026] = 0.0 - k[945]*y[IDX_H2OII] + k[960]*y[IDX_OHII];
    data[3027] = 0.0 - k[512]*y[IDX_H2OII] - k[513]*y[IDX_H2OII] - k[514]*y[IDX_H2OII];
    data[3028] = 0.0 - k[1536]*y[IDX_OHII];
    data[3029] = 0.0 - k[1537]*y[IDX_OHII];
    data[3030] = 0.0 + k[1200]*y[IDX_HeII];
    data[3031] = 0.0 + k[339]*y[IDX_OHI];
    data[3032] = 0.0 + k[1510]*y[IDX_OI];
    data[3033] = 0.0 + k[340]*y[IDX_OHI];
    data[3034] = 0.0 + k[1506]*y[IDX_OI];
    data[3035] = 0.0 + k[342]*y[IDX_OHI];
    data[3036] = 0.0 + k[171]*y[IDX_OHI] + k[932]*y[IDX_OI];
    data[3037] = 0.0 - k[1535]*y[IDX_OHII];
    data[3038] = 0.0 - k[1528]*y[IDX_OHII];
    data[3039] = 0.0 + k[341]*y[IDX_OHI];
    data[3040] = 0.0 - k[1520]*y[IDX_OHII];
    data[3041] = 0.0 + k[1370]*y[IDX_OI];
    data[3042] = 0.0 - k[277]*y[IDX_OHII] - k[1411]*y[IDX_OHII];
    data[3043] = 0.0 + k[251]*y[IDX_OHI];
    data[3044] = 0.0 - k[330]*y[IDX_OHII] - k[1518]*y[IDX_OHII];
    data[3045] = 0.0 + k[319]*y[IDX_OHI] + k[958]*y[IDX_H2I] + k[1476]*y[IDX_HCOI];
    data[3046] = 0.0 - k[333]*y[IDX_OHII] - k[1524]*y[IDX_OHII];
    data[3047] = 0.0 + k[2016];
    data[3048] = 0.0 - k[71]*y[IDX_CH2I] - k[92]*y[IDX_CHI] - k[277]*y[IDX_NH2I] -
        k[329]*y[IDX_C2I] - k[330]*y[IDX_C2HI] - k[331]*y[IDX_H2COI] -
        k[332]*y[IDX_H2OI] - k[333]*y[IDX_H2SI] - k[334]*y[IDX_HCOI] -
        k[335]*y[IDX_NH3I] - k[336]*y[IDX_NOI] - k[337]*y[IDX_O2I] -
        k[338]*y[IDX_SI] - k[582]*y[IDX_EM] - k[702]*y[IDX_CI] -
        k[771]*y[IDX_CH2I] - k[819]*y[IDX_CH4I] - k[820]*y[IDX_CH4I] -
        k[860]*y[IDX_CHI] - k[960]*y[IDX_H2I] - k[1340]*y[IDX_NI] -
        k[1411]*y[IDX_NH2I] - k[1460]*y[IDX_NHI] - k[1511]*y[IDX_OI] -
        k[1517]*y[IDX_C2I] - k[1518]*y[IDX_C2HI] - k[1519]*y[IDX_CNI] -
        k[1520]*y[IDX_CO2I] - k[1521]*y[IDX_COI] - k[1522]*y[IDX_H2COI] -
        k[1523]*y[IDX_H2OI] - k[1524]*y[IDX_H2SI] - k[1525]*y[IDX_HCNI] -
        k[1526]*y[IDX_HCOI] - k[1527]*y[IDX_HCOI] - k[1528]*y[IDX_HNCI] -
        k[1529]*y[IDX_N2I] - k[1530]*y[IDX_NH3I] - k[1531]*y[IDX_NOI] -
        k[1532]*y[IDX_OHI] - k[1533]*y[IDX_SI] - k[1534]*y[IDX_SI] -
        k[1535]*y[IDX_SiI] - k[1536]*y[IDX_SiHI] - k[1537]*y[IDX_SiOI] - k[2068]
        - k[2233];
    data[3049] = 0.0 - k[1529]*y[IDX_OHII];
    data[3050] = 0.0 - k[329]*y[IDX_OHII] - k[1517]*y[IDX_OHII];
    data[3051] = 0.0 - k[71]*y[IDX_OHII] - k[771]*y[IDX_OHII];
    data[3052] = 0.0 - k[1460]*y[IDX_OHII];
    data[3053] = 0.0 - k[1519]*y[IDX_OHII];
    data[3054] = 0.0 - k[331]*y[IDX_OHII] - k[1522]*y[IDX_OHII];
    data[3055] = 0.0 - k[819]*y[IDX_OHII] - k[820]*y[IDX_OHII];
    data[3056] = 0.0 - k[334]*y[IDX_OHII] + k[1476]*y[IDX_OII] - k[1526]*y[IDX_OHII] -
        k[1527]*y[IDX_OHII];
    data[3057] = 0.0 - k[1525]*y[IDX_OHII];
    data[3058] = 0.0 - k[336]*y[IDX_OHII] - k[1531]*y[IDX_OHII];
    data[3059] = 0.0 - k[92]*y[IDX_OHII] - k[860]*y[IDX_OHII];
    data[3060] = 0.0 - k[335]*y[IDX_OHII] - k[1530]*y[IDX_OHII];
    data[3061] = 0.0 - k[337]*y[IDX_OHII];
    data[3062] = 0.0 - k[338]*y[IDX_OHII] - k[1533]*y[IDX_OHII] - k[1534]*y[IDX_OHII];
    data[3063] = 0.0 + k[138]*y[IDX_HII] + k[171]*y[IDX_H2II] + k[251]*y[IDX_NII] +
        k[319]*y[IDX_OII] + k[339]*y[IDX_C2II] + k[340]*y[IDX_CNII] +
        k[341]*y[IDX_COII] + k[342]*y[IDX_N2II] - k[1532]*y[IDX_OHII] + k[2070];
    data[3064] = 0.0 - k[1340]*y[IDX_OHII];
    data[3065] = 0.0 + k[1200]*y[IDX_CH3OHI] + k[1222]*y[IDX_H2OI];
    data[3066] = 0.0 + k[1064]*y[IDX_OI];
    data[3067] = 0.0 + k[138]*y[IDX_OHI];
    data[3068] = 0.0 + k[932]*y[IDX_H2II] + k[1064]*y[IDX_H3II] + k[1370]*y[IDX_NHII] +
        k[1506]*y[IDX_N2HII] + k[1510]*y[IDX_O2HII] - k[1511]*y[IDX_OHII];
    data[3069] = 0.0 - k[702]*y[IDX_OHII];
    data[3070] = 0.0 - k[332]*y[IDX_OHII] + k[1222]*y[IDX_HeII] - k[1523]*y[IDX_OHII];
    data[3071] = 0.0 - k[1521]*y[IDX_OHII];
    data[3072] = 0.0 + k[958]*y[IDX_OII] - k[960]*y[IDX_OHII];
    data[3073] = 0.0 - k[582]*y[IDX_OHII];
    data[3074] = 0.0 + k[1023]*y[IDX_H3II];
    data[3075] = 0.0 - k[775]*y[IDX_CH3II] + k[886]*y[IDX_HII] + k[1199]*y[IDX_HeII];
    data[3076] = 0.0 - k[789]*y[IDX_CH3II];
    data[3077] = 0.0 + k[613]*y[IDX_CII] + k[709]*y[IDX_CHII] - k[776]*y[IDX_CH3II] +
        k[887]*y[IDX_HII] + k[1031]*y[IDX_H3II] + k[1201]*y[IDX_HeII] +
        k[1292]*y[IDX_NII];
    data[3078] = 0.0 - k[788]*y[IDX_CH3II];
    data[3079] = 0.0 - k[774]*y[IDX_CH3II];
    data[3080] = 0.0 + k[1101]*y[IDX_HI] + k[1493]*y[IDX_OI] + k[1994];
    data[3081] = 0.0 - k[73]*y[IDX_CH3II];
    data[3082] = 0.0 + k[770]*y[IDX_CH2I];
    data[3083] = 0.0 + k[765]*y[IDX_CH2I];
    data[3084] = 0.0 - k[785]*y[IDX_CH3II];
    data[3085] = 0.0 + k[764]*y[IDX_CH2I];
    data[3086] = 0.0 + k[755]*y[IDX_CH2I];
    data[3087] = 0.0 + k[816]*y[IDX_CH4I];
    data[3088] = 0.0 - k[780]*y[IDX_CH3II];
    data[3089] = 0.0 + k[761]*y[IDX_CH2I] + k[762]*y[IDX_CH2I];
    data[3090] = 0.0 + k[913]*y[IDX_CH2I] + k[914]*y[IDX_CH4I];
    data[3091] = 0.0 + k[760]*y[IDX_CH2I];
    data[3092] = 0.0 + k[766]*y[IDX_CH2I];
    data[3093] = 0.0 + k[1292]*y[IDX_CH3OHI] + k[1293]*y[IDX_CH4I];
    data[3094] = 0.0 + k[767]*y[IDX_CH2I];
    data[3095] = 0.0 + k[754]*y[IDX_CH2I];
    data[3096] = 0.0 + k[1468]*y[IDX_CH4I];
    data[3097] = 0.0 + k[757]*y[IDX_CH2I];
    data[3098] = 0.0 + k[747]*y[IDX_HCOI] + k[938]*y[IDX_H2I];
    data[3099] = 0.0 + k[768]*y[IDX_CH2I];
    data[3100] = 0.0 - k[778]*y[IDX_CH3II];
    data[3101] = 0.0 + k[758]*y[IDX_CH2I];
    data[3102] = 0.0 + k[771]*y[IDX_CH2I];
    data[3103] = 0.0 - k[72]*y[IDX_HCOI] - k[73]*y[IDX_MgI] - k[74]*y[IDX_NOI] -
        k[481]*y[IDX_EM] - k[482]*y[IDX_EM] - k[483]*y[IDX_EM] -
        k[688]*y[IDX_CI] - k[774]*y[IDX_C2H4I] - k[775]*y[IDX_CH3CNI] -
        k[776]*y[IDX_CH3OHI] - k[777]*y[IDX_H2COI] - k[778]*y[IDX_H2SI] -
        k[779]*y[IDX_HCOI] - k[780]*y[IDX_HSI] - k[781]*y[IDX_NH3I] -
        k[782]*y[IDX_O2I] - k[783]*y[IDX_OI] - k[784]*y[IDX_OI] -
        k[785]*y[IDX_OCSI] - k[786]*y[IDX_OHI] - k[787]*y[IDX_SI] -
        k[788]*y[IDX_SOI] - k[789]*y[IDX_SiH4I] - k[839]*y[IDX_CHI] -
        k[1100]*y[IDX_HI] - k[1446]*y[IDX_NHI] - k[1984] - k[1985] -
        k[2107]*y[IDX_H2OI] - k[2108]*y[IDX_HCNI] - k[2114]*y[IDX_H2I] -
        k[2133]*y[IDX_EM] - k[2245];
    data[3104] = 0.0 + k[754]*y[IDX_C2HII] + k[755]*y[IDX_CH5II] + k[757]*y[IDX_H2COII] +
        k[758]*y[IDX_H2OII] + k[759]*y[IDX_H3OII] + k[760]*y[IDX_HCNII] +
        k[761]*y[IDX_HCNHII] + k[762]*y[IDX_HCNHII] + k[763]*y[IDX_HCOII] +
        k[764]*y[IDX_HNOII] + k[765]*y[IDX_N2HII] + k[766]*y[IDX_NHII] +
        k[767]*y[IDX_NH2II] + k[768]*y[IDX_NH3II] + k[770]*y[IDX_O2HII] +
        k[771]*y[IDX_OHII] + k[913]*y[IDX_H2II] + k[1028]*y[IDX_H3II];
    data[3105] = 0.0 + k[115]*y[IDX_HII] + k[385] + k[1987];
    data[3106] = 0.0 + k[709]*y[IDX_CH3OHI] + k[715]*y[IDX_H2COI];
    data[3107] = 0.0 - k[1446]*y[IDX_CH3II];
    data[3108] = 0.0 + k[715]*y[IDX_CHII] - k[777]*y[IDX_CH3II];
    data[3109] = 0.0 + k[816]*y[IDX_N2II] + k[890]*y[IDX_HII] + k[914]*y[IDX_H2II] +
        k[1204]*y[IDX_HeII] + k[1293]*y[IDX_NII] + k[1468]*y[IDX_OII];
    data[3110] = 0.0 - k[72]*y[IDX_CH3II] + k[747]*y[IDX_CH2II] - k[779]*y[IDX_CH3II];
    data[3111] = 0.0 - k[2108]*y[IDX_CH3II];
    data[3112] = 0.0 - k[74]*y[IDX_CH3II];
    data[3113] = 0.0 - k[839]*y[IDX_CH3II];
    data[3114] = 0.0 - k[781]*y[IDX_CH3II];
    data[3115] = 0.0 + k[759]*y[IDX_CH2I];
    data[3116] = 0.0 - k[782]*y[IDX_CH3II];
    data[3117] = 0.0 + k[613]*y[IDX_CH3OHI];
    data[3118] = 0.0 - k[787]*y[IDX_CH3II];
    data[3119] = 0.0 - k[786]*y[IDX_CH3II];
    data[3120] = 0.0 + k[1199]*y[IDX_CH3CNI] + k[1201]*y[IDX_CH3OHI] +
        k[1204]*y[IDX_CH4I];
    data[3121] = 0.0 + k[1023]*y[IDX_C2H5OHI] + k[1028]*y[IDX_CH2I] +
        k[1031]*y[IDX_CH3OHI];
    data[3122] = 0.0 + k[115]*y[IDX_CH3I] + k[886]*y[IDX_CH3CNI] + k[887]*y[IDX_CH3OHI] +
        k[890]*y[IDX_CH4I];
    data[3123] = 0.0 - k[783]*y[IDX_CH3II] - k[784]*y[IDX_CH3II] + k[1493]*y[IDX_CH4II];
    data[3124] = 0.0 - k[688]*y[IDX_CH3II];
    data[3125] = 0.0 + k[763]*y[IDX_CH2I];
    data[3126] = 0.0 - k[2107]*y[IDX_CH3II];
    data[3127] = 0.0 + k[938]*y[IDX_CH2II] - k[2114]*y[IDX_CH3II];
    data[3128] = 0.0 - k[481]*y[IDX_CH3II] - k[482]*y[IDX_CH3II] - k[483]*y[IDX_CH3II] -
        k[2133]*y[IDX_CH3II];
    data[3129] = 0.0 - k[1100]*y[IDX_CH3II] + k[1101]*y[IDX_CH4II];
    data[3130] = 0.0 + k[2347] + k[2348] + k[2349] + k[2350];
    data[3131] = 0.0 + k[1820]*y[IDX_NI] + k[1822]*y[IDX_NI];
    data[3132] = 0.0 + k[1304]*y[IDX_NII] + k[1818]*y[IDX_NI];
    data[3133] = 0.0 + k[1860]*y[IDX_NOI];
    data[3134] = 0.0 + k[1321]*y[IDX_N2HII];
    data[3135] = 0.0 + k[1824]*y[IDX_NI];
    data[3136] = 0.0 + k[226]*y[IDX_N2II];
    data[3137] = 0.0 - k[1320]*y[IDX_N2I];
    data[3138] = 0.0 + k[565]*y[IDX_EM] + k[655]*y[IDX_C2I] + k[682]*y[IDX_C2HI] +
        k[698]*y[IDX_CI] + k[765]*y[IDX_CH2I] + k[817]*y[IDX_CH4I] +
        k[853]*y[IDX_CHI] + k[877]*y[IDX_COI] + k[1011]*y[IDX_H2OI] +
        k[1128]*y[IDX_HCNI] + k[1160]*y[IDX_HCOI] + k[1171]*y[IDX_HNCI] +
        k[1321]*y[IDX_CH3CNI] + k[1322]*y[IDX_CO2I] + k[1323]*y[IDX_H2COI] +
        k[1324]*y[IDX_SI] + k[1408]*y[IDX_NH2I] + k[1440]*y[IDX_NH3I] +
        k[1454]*y[IDX_NHI] + k[1506]*y[IDX_OI] + k[1545]*y[IDX_OHI];
    data[3139] = 0.0 + k[257]*y[IDX_N2II] + k[1318]*y[IDX_N2II];
    data[3140] = 0.0 - k[1319]*y[IDX_N2I];
    data[3141] = 0.0 + k[38]*y[IDX_C2I] + k[49]*y[IDX_C2HI] + k[53]*y[IDX_CI] +
        k[67]*y[IDX_CH2I] + k[88]*y[IDX_CHI] + k[100]*y[IDX_CNI] +
        k[107]*y[IDX_COI] + k[189]*y[IDX_H2OI] + k[201]*y[IDX_HCNI] +
        k[226]*y[IDX_MgI] + k[252]*y[IDX_H2COI] + k[253]*y[IDX_H2SI] +
        k[254]*y[IDX_HCOI] + k[255]*y[IDX_NOI] + k[256]*y[IDX_O2I] +
        k[257]*y[IDX_OCSI] + k[258]*y[IDX_SI] + k[259]*y[IDX_NI] +
        k[275]*y[IDX_NH2I] + k[289]*y[IDX_NH3I] + k[296]*y[IDX_NHI] +
        k[328]*y[IDX_OI] + k[342]*y[IDX_OHI] + k[815]*y[IDX_CH4I] +
        k[816]*y[IDX_CH4I] + k[1314]*y[IDX_H2COI] + k[1315]*y[IDX_H2SI] +
        k[1316]*y[IDX_H2SI] + k[1318]*y[IDX_OCSI];
    data[3142] = 0.0 - k[927]*y[IDX_N2I];
    data[3143] = 0.0 + k[1171]*y[IDX_N2HII];
    data[3144] = 0.0 + k[1322]*y[IDX_N2HII];
    data[3145] = 0.0 - k[1363]*y[IDX_N2I];
    data[3146] = 0.0 + k[275]*y[IDX_N2II] + k[1408]*y[IDX_N2HII] + k[1834]*y[IDX_NOI] +
        k[1835]*y[IDX_NOI];
    data[3147] = 0.0 + k[1304]*y[IDX_NCCNI];
    data[3148] = 0.0 + k[49]*y[IDX_N2II] + k[682]*y[IDX_N2HII];
    data[3149] = 0.0 - k[1477]*y[IDX_N2I];
    data[3150] = 0.0 + k[253]*y[IDX_N2II] + k[1315]*y[IDX_N2II] + k[1316]*y[IDX_N2II];
    data[3151] = 0.0 - k[1529]*y[IDX_N2I];
    data[3152] = 0.0 - k[215]*y[IDX_HeII] - k[421] - k[927]*y[IDX_H2II] -
        k[1055]*y[IDX_H3II] - k[1250]*y[IDX_HeII] - k[1319]*y[IDX_HNOII] -
        k[1320]*y[IDX_O2HII] - k[1363]*y[IDX_NHII] - k[1477]*y[IDX_OII] -
        k[1529]*y[IDX_OHII] - k[1597]*y[IDX_CI] - k[1627]*y[IDX_CH2I] -
        k[1681]*y[IDX_CHI] - k[1901]*y[IDX_OI] - k[2045] - k[2219];
    data[3153] = 0.0 + k[38]*y[IDX_N2II] + k[655]*y[IDX_N2HII];
    data[3154] = 0.0 + k[67]*y[IDX_N2II] + k[765]*y[IDX_N2HII] - k[1627]*y[IDX_N2I];
    data[3155] = 0.0 + k[296]*y[IDX_N2II] + k[1454]*y[IDX_N2HII] + k[1819]*y[IDX_NI] +
        k[1843]*y[IDX_NHI] + k[1843]*y[IDX_NHI] + k[1844]*y[IDX_NHI] +
        k[1844]*y[IDX_NHI] + k[1847]*y[IDX_NOI] + k[1848]*y[IDX_NOI];
    data[3156] = 0.0 + k[100]*y[IDX_N2II] + k[1703]*y[IDX_CNI] + k[1703]*y[IDX_CNI] +
        k[1710]*y[IDX_NOI] + k[1807]*y[IDX_NI];
    data[3157] = 0.0 + k[252]*y[IDX_N2II] + k[1314]*y[IDX_N2II] + k[1323]*y[IDX_N2HII];
    data[3158] = 0.0 + k[815]*y[IDX_N2II] + k[816]*y[IDX_N2II] + k[817]*y[IDX_N2HII];
    data[3159] = 0.0 + k[254]*y[IDX_N2II] + k[1160]*y[IDX_N2HII];
    data[3160] = 0.0 + k[201]*y[IDX_N2II] + k[1128]*y[IDX_N2HII];
    data[3161] = 0.0 + k[255]*y[IDX_N2II] + k[1710]*y[IDX_CNI] + k[1823]*y[IDX_NI] +
        k[1834]*y[IDX_NH2I] + k[1835]*y[IDX_NH2I] + k[1847]*y[IDX_NHI] +
        k[1848]*y[IDX_NHI] + k[1858]*y[IDX_NOI] + k[1858]*y[IDX_NOI] +
        k[1860]*y[IDX_OCNI];
    data[3162] = 0.0 + k[88]*y[IDX_N2II] + k[853]*y[IDX_N2HII] - k[1681]*y[IDX_N2I];
    data[3163] = 0.0 + k[289]*y[IDX_N2II] + k[1440]*y[IDX_N2HII];
    data[3164] = 0.0 + k[256]*y[IDX_N2II];
    data[3165] = 0.0 + k[258]*y[IDX_N2II] + k[1324]*y[IDX_N2HII];
    data[3166] = 0.0 + k[342]*y[IDX_N2II] + k[1545]*y[IDX_N2HII];
    data[3167] = 0.0 + k[259]*y[IDX_N2II] + k[1807]*y[IDX_CNI] + k[1818]*y[IDX_NCCNI] +
        k[1819]*y[IDX_NHI] + k[1820]*y[IDX_NO2I] + k[1822]*y[IDX_NO2I] +
        k[1823]*y[IDX_NOI] + k[1824]*y[IDX_NSI];
    data[3168] = 0.0 - k[215]*y[IDX_N2I] - k[1250]*y[IDX_N2I];
    data[3169] = 0.0 - k[1055]*y[IDX_N2I];
    data[3170] = 0.0 + k[328]*y[IDX_N2II] + k[1506]*y[IDX_N2HII] - k[1901]*y[IDX_N2I];
    data[3171] = 0.0 + k[53]*y[IDX_N2II] + k[698]*y[IDX_N2HII] - k[1597]*y[IDX_N2I];
    data[3172] = 0.0 + k[189]*y[IDX_N2II] + k[1011]*y[IDX_N2HII];
    data[3173] = 0.0 + k[107]*y[IDX_N2II] + k[877]*y[IDX_N2HII];
    data[3174] = 0.0 + k[565]*y[IDX_N2HII];
    data[3175] = 0.0 + k[675]*y[IDX_C2H2I];
    data[3176] = 0.0 + k[611]*y[IDX_CII];
    data[3177] = 0.0 - k[665]*y[IDX_C2H2II] - k[666]*y[IDX_C2H2II];
    data[3178] = 0.0 - k[671]*y[IDX_C2H2II] - k[672]*y[IDX_C2H2II] - k[673]*y[IDX_C2H2II]
        - k[674]*y[IDX_C2H2II];
    data[3179] = 0.0 + k[882]*y[IDX_HII] + k[1182]*y[IDX_HeII];
    data[3180] = 0.0 + k[883]*y[IDX_HII] + k[910]*y[IDX_H2II] + k[1184]*y[IDX_HeII] +
        k[1464]*y[IDX_OII];
    data[3181] = 0.0 + k[75]*y[IDX_C2H2I];
    data[3182] = 0.0 - k[220]*y[IDX_C2H2II];
    data[3183] = 0.0 + k[804]*y[IDX_CH4I];
    data[3184] = 0.0 + k[683]*y[IDX_C2HI];
    data[3185] = 0.0 + k[682]*y[IDX_C2HI];
    data[3186] = 0.0 + k[681]*y[IDX_C2HI];
    data[3187] = 0.0 + k[824]*y[IDX_C2HI];
    data[3188] = 0.0 + k[154]*y[IDX_C2H2I] + k[910]*y[IDX_C2H4I] + k[911]*y[IDX_C2HI];
    data[3189] = 0.0 - k[670]*y[IDX_C2H2II];
    data[3190] = 0.0 + k[46]*y[IDX_HCNII] + k[75]*y[IDX_CH4II] + k[111]*y[IDX_HII] +
        k[154]*y[IDX_H2II] + k[176]*y[IDX_H2OII] + k[208]*y[IDX_HeII] +
        k[307]*y[IDX_OII] + k[321]*y[IDX_O2II] + k[367] + k[675]*y[IDX_C2N2II] +
        k[1964];
    data[3191] = 0.0 + k[46]*y[IDX_C2H2I] + k[679]*y[IDX_C2HI];
    data[3192] = 0.0 - k[669]*y[IDX_C2H2II];
    data[3193] = 0.0 + k[1348]*y[IDX_C2HI];
    data[3194] = 0.0 - k[1396]*y[IDX_C2H2II];
    data[3195] = 0.0 + k[321]*y[IDX_C2H2I];
    data[3196] = 0.0 + k[1375]*y[IDX_C2HI];
    data[3197] = 0.0 + k[661]*y[IDX_HCNI] + k[663]*y[IDX_HCOI] + k[805]*y[IDX_CH4I] +
        k[936]*y[IDX_H2I];
    data[3198] = 0.0 + k[678]*y[IDX_H2COII] + k[679]*y[IDX_HCNII] + k[680]*y[IDX_HCOII] +
        k[681]*y[IDX_HNOII] + k[682]*y[IDX_N2HII] + k[683]*y[IDX_O2HII] +
        k[824]*y[IDX_CH5II] + k[911]*y[IDX_H2II] + k[977]*y[IDX_H2OII] +
        k[1025]*y[IDX_H3II] + k[1348]*y[IDX_NHII] + k[1375]*y[IDX_NH2II] +
        k[1518]*y[IDX_OHII];
    data[3199] = 0.0 + k[307]*y[IDX_C2H2I] + k[1464]*y[IDX_C2H4I];
    data[3200] = 0.0 + k[678]*y[IDX_C2HI];
    data[3201] = 0.0 + k[1412]*y[IDX_C2I];
    data[3202] = 0.0 - k[43]*y[IDX_C2H2II] - k[667]*y[IDX_C2H2II];
    data[3203] = 0.0 + k[176]*y[IDX_C2H2I] + k[977]*y[IDX_C2HI];
    data[3204] = 0.0 + k[1518]*y[IDX_C2HI];
    data[3205] = 0.0 + k[839]*y[IDX_CHI];
    data[3206] = 0.0 - k[42]*y[IDX_H2COI] - k[43]*y[IDX_H2SI] - k[44]*y[IDX_HCOI] -
        k[45]*y[IDX_NOI] - k[220]*y[IDX_MgI] - k[282]*y[IDX_NH3I] -
        k[461]*y[IDX_EM] - k[462]*y[IDX_EM] - k[463]*y[IDX_EM] -
        k[665]*y[IDX_CH3CNI] - k[666]*y[IDX_CH3CNI] - k[667]*y[IDX_H2SI] -
        k[668]*y[IDX_HCNI] - k[669]*y[IDX_HNCI] - k[670]*y[IDX_SiI] -
        k[671]*y[IDX_SiH4I] - k[672]*y[IDX_SiH4I] - k[673]*y[IDX_SiH4I] -
        k[674]*y[IDX_SiH4I] - k[806]*y[IDX_CH4I] - k[991]*y[IDX_H2OI] -
        k[1328]*y[IDX_NI] - k[1329]*y[IDX_NI] - k[1330]*y[IDX_NI] -
        k[1396]*y[IDX_NH2I] - k[1419]*y[IDX_NH3I] - k[1492]*y[IDX_OI] - k[2277];
    data[3207] = 0.0 + k[1412]*y[IDX_NH3II];
    data[3208] = 0.0 + k[610]*y[IDX_CII];
    data[3209] = 0.0 + k[711]*y[IDX_CH4I];
    data[3210] = 0.0 - k[42]*y[IDX_C2H2II];
    data[3211] = 0.0 + k[614]*y[IDX_CII] + k[711]*y[IDX_CHII] + k[804]*y[IDX_C2II] +
        k[805]*y[IDX_C2HII] - k[806]*y[IDX_C2H2II];
    data[3212] = 0.0 - k[44]*y[IDX_C2H2II] + k[663]*y[IDX_C2HII];
    data[3213] = 0.0 + k[661]*y[IDX_C2HII] - k[668]*y[IDX_C2H2II];
    data[3214] = 0.0 - k[45]*y[IDX_C2H2II];
    data[3215] = 0.0 + k[839]*y[IDX_CH3II];
    data[3216] = 0.0 - k[282]*y[IDX_C2H2II] - k[1419]*y[IDX_C2H2II];
    data[3217] = 0.0 + k[610]*y[IDX_CH3I] + k[611]*y[IDX_CH3CCHI] + k[614]*y[IDX_CH4I];
    data[3218] = 0.0 - k[1328]*y[IDX_C2H2II] - k[1329]*y[IDX_C2H2II] -
        k[1330]*y[IDX_C2H2II];
    data[3219] = 0.0 + k[208]*y[IDX_C2H2I] + k[1182]*y[IDX_C2H3I] + k[1184]*y[IDX_C2H4I];
    data[3220] = 0.0 + k[1025]*y[IDX_C2HI];
    data[3221] = 0.0 + k[111]*y[IDX_C2H2I] + k[882]*y[IDX_C2H3I] + k[883]*y[IDX_C2H4I];
    data[3222] = 0.0 - k[1492]*y[IDX_C2H2II];
    data[3223] = 0.0 + k[680]*y[IDX_C2HI];
    data[3224] = 0.0 - k[991]*y[IDX_C2H2II];
    data[3225] = 0.0 + k[936]*y[IDX_C2HII];
    data[3226] = 0.0 - k[461]*y[IDX_C2H2II] - k[462]*y[IDX_C2H2II] - k[463]*y[IDX_C2H2II];
    data[3227] = 0.0 + k[2315] + k[2316] + k[2317] + k[2318];
    data[3228] = 0.0 + k[591]*y[IDX_EM];
    data[3229] = 0.0 + k[378] + k[1191]*y[IDX_HeII] + k[1975];
    data[3230] = 0.0 + k[475]*y[IDX_EM];
    data[3231] = 0.0 + k[2079];
    data[3232] = 0.0 + k[377] + k[1974];
    data[3233] = 0.0 + k[588]*y[IDX_EM];
    data[3234] = 0.0 + k[473]*y[IDX_EM];
    data[3235] = 0.0 + k[1274]*y[IDX_HeII] + k[2078];
    data[3236] = 0.0 + k[375] + k[1584]*y[IDX_CI] + k[1972];
    data[3237] = 0.0 + k[642]*y[IDX_CII];
    data[3238] = 0.0 - k[659]*y[IDX_C2I];
    data[3239] = 0.0 + k[468]*y[IDX_EM];
    data[3240] = 0.0 + k[1592]*y[IDX_CI];
    data[3241] = 0.0 + k[33]*y[IDX_HCOI] + k[34]*y[IDX_NOI] + k[35]*y[IDX_SI] +
        k[50]*y[IDX_CI] + k[62]*y[IDX_CH2I] + k[82]*y[IDX_CHI] +
        k[271]*y[IDX_NH2I] + k[339]*y[IDX_OHI] - k[647]*y[IDX_C2I];
    data[3242] = 0.0 - k[657]*y[IDX_C2I];
    data[3243] = 0.0 - k[36]*y[IDX_C2I];
    data[3244] = 0.0 - k[655]*y[IDX_C2I];
    data[3245] = 0.0 - k[654]*y[IDX_C2I];
    data[3246] = 0.0 - k[823]*y[IDX_C2I];
    data[3247] = 0.0 - k[38]*y[IDX_C2I];
    data[3248] = 0.0 - k[153]*y[IDX_C2I] - k[909]*y[IDX_C2I];
    data[3249] = 0.0 - k[1570]*y[IDX_C2I];
    data[3250] = 0.0 - k[652]*y[IDX_C2I];
    data[3251] = 0.0 + k[664]*y[IDX_C2HII];
    data[3252] = 0.0 - k[37]*y[IDX_C2I] + k[677]*y[IDX_C2HI];
    data[3253] = 0.0 - k[1345]*y[IDX_C2I] - k[1346]*y[IDX_C2I] - k[1347]*y[IDX_C2I];
    data[3254] = 0.0 + k[271]*y[IDX_C2II] + k[1395]*y[IDX_C2HII];
    data[3255] = 0.0 - k[39]*y[IDX_C2I] - k[656]*y[IDX_C2I];
    data[3256] = 0.0 - k[233]*y[IDX_C2I];
    data[3257] = 0.0 - k[1374]*y[IDX_C2I];
    data[3258] = 0.0 + k[459]*y[IDX_EM] + k[660]*y[IDX_H2COI] + k[662]*y[IDX_HCNI] +
        k[664]*y[IDX_HNCI] + k[754]*y[IDX_CH2I] + k[838]*y[IDX_CHI] +
        k[1395]*y[IDX_NH2I] + k[1418]*y[IDX_NH3I];
    data[3259] = 0.0 + k[373] + k[677]*y[IDX_COII] + k[1970];
    data[3260] = 0.0 - k[306]*y[IDX_C2I] - k[1463]*y[IDX_C2I];
    data[3261] = 0.0 - k[651]*y[IDX_C2I];
    data[3262] = 0.0 - k[1412]*y[IDX_C2I];
    data[3263] = 0.0 - k[175]*y[IDX_C2I] - k[976]*y[IDX_C2I];
    data[3264] = 0.0 - k[329]*y[IDX_C2I] - k[1517]*y[IDX_C2I];
    data[3265] = 0.0 + k[461]*y[IDX_EM];
    data[3266] = 0.0 - k[36]*y[IDX_CNII] - k[37]*y[IDX_COII] - k[38]*y[IDX_N2II] -
        k[39]*y[IDX_O2II] - k[110]*y[IDX_HII] - k[153]*y[IDX_H2II] -
        k[175]*y[IDX_H2OII] - k[207]*y[IDX_HeII] - k[233]*y[IDX_NII] -
        k[306]*y[IDX_OII] - k[329]*y[IDX_OHII] - k[366] - k[647]*y[IDX_C2II] -
        k[651]*y[IDX_H2COII] - k[652]*y[IDX_HCNII] - k[653]*y[IDX_HCOII] -
        k[654]*y[IDX_HNOII] - k[655]*y[IDX_N2HII] - k[656]*y[IDX_O2II] -
        k[657]*y[IDX_O2HII] - k[658]*y[IDX_SII] - k[659]*y[IDX_SiOII] -
        k[705]*y[IDX_CHII] - k[823]*y[IDX_CH5II] - k[909]*y[IDX_H2II] -
        k[976]*y[IDX_H2OII] - k[1022]*y[IDX_H3II] - k[1080]*y[IDX_H3OII] -
        k[1177]*y[IDX_HeII] - k[1345]*y[IDX_NHII] - k[1346]*y[IDX_NHII] -
        k[1347]*y[IDX_NHII] - k[1374]*y[IDX_NH2II] - k[1412]*y[IDX_NH3II] -
        k[1463]*y[IDX_OII] - k[1517]*y[IDX_OHII] - k[1570]*y[IDX_C2H2I] -
        k[1571]*y[IDX_HCNI] - k[1572]*y[IDX_O2I] - k[1573]*y[IDX_SI] -
        k[1736]*y[IDX_HI] - k[1790]*y[IDX_NI] - k[1867]*y[IDX_OI] - k[1961] -
        k[1962] - k[2209];
    data[3267] = 0.0 + k[62]*y[IDX_C2II] + k[754]*y[IDX_C2HII];
    data[3268] = 0.0 - k[705]*y[IDX_C2I];
    data[3269] = 0.0 - k[658]*y[IDX_C2I];
    data[3270] = 0.0 + k[1590]*y[IDX_CI] + k[1703]*y[IDX_CNI] + k[1703]*y[IDX_CNI];
    data[3271] = 0.0 + k[660]*y[IDX_C2HII];
    data[3272] = 0.0 + k[33]*y[IDX_C2II];
    data[3273] = 0.0 + k[662]*y[IDX_C2HII] - k[1571]*y[IDX_C2I];
    data[3274] = 0.0 + k[34]*y[IDX_C2II];
    data[3275] = 0.0 + k[82]*y[IDX_C2II] + k[838]*y[IDX_C2HII] + k[1589]*y[IDX_CI];
    data[3276] = 0.0 + k[1418]*y[IDX_C2HII];
    data[3277] = 0.0 - k[1080]*y[IDX_C2I];
    data[3278] = 0.0 - k[1572]*y[IDX_C2I];
    data[3279] = 0.0 + k[642]*y[IDX_SiCI];
    data[3280] = 0.0 + k[35]*y[IDX_C2II] - k[1573]*y[IDX_C2I];
    data[3281] = 0.0 + k[339]*y[IDX_C2II];
    data[3282] = 0.0 - k[1790]*y[IDX_C2I];
    data[3283] = 0.0 - k[207]*y[IDX_C2I] - k[1177]*y[IDX_C2I] + k[1191]*y[IDX_C4HI] +
        k[1274]*y[IDX_SiC2I];
    data[3284] = 0.0 - k[1022]*y[IDX_C2I];
    data[3285] = 0.0 - k[110]*y[IDX_C2I];
    data[3286] = 0.0 - k[1867]*y[IDX_C2I];
    data[3287] = 0.0 + k[50]*y[IDX_C2II] + k[1584]*y[IDX_C2NI] + k[1589]*y[IDX_CHI] +
        k[1590]*y[IDX_CNI] + k[1591]*y[IDX_COI] + k[1592]*y[IDX_CSI] +
        k[2101]*y[IDX_CI] + k[2101]*y[IDX_CI];
    data[3288] = 0.0 - k[653]*y[IDX_C2I];
    data[3289] = 0.0 + k[1591]*y[IDX_CI];
    data[3290] = 0.0 + k[459]*y[IDX_C2HII] + k[461]*y[IDX_C2H2II] + k[468]*y[IDX_C2NII] +
        k[473]*y[IDX_C3II] + k[475]*y[IDX_C4NII] + k[588]*y[IDX_SiC2II] +
        k[591]*y[IDX_SiC3II];
    data[3291] = 0.0 - k[1736]*y[IDX_C2I];
    data[3292] = 0.0 + k[383] + k[1194]*y[IDX_HeII] + k[1983];
    data[3293] = 0.0 + k[1220]*y[IDX_HeII];
    data[3294] = 0.0 + k[1692]*y[IDX_CHI];
    data[3295] = 0.0 - k[1628]*y[IDX_CH2I];
    data[3296] = 0.0 + k[486]*y[IDX_EM];
    data[3297] = 0.0 - k[1626]*y[IDX_CH2I] + k[1680]*y[IDX_CHI];
    data[3298] = 0.0 - k[773]*y[IDX_CH2I];
    data[3299] = 0.0 + k[710]*y[IDX_CHII];
    data[3300] = 0.0 + k[1185]*y[IDX_HeII] + k[1872]*y[IDX_OI];
    data[3301] = 0.0 + k[491]*y[IDX_EM];
    data[3302] = 0.0 - k[62]*y[IDX_CH2I] + k[804]*y[IDX_CH4I];
    data[3303] = 0.0 - k[770]*y[IDX_CH2I];
    data[3304] = 0.0 - k[63]*y[IDX_CH2I];
    data[3305] = 0.0 - k[765]*y[IDX_CH2I];
    data[3306] = 0.0 - k[764]*y[IDX_CH2I];
    data[3307] = 0.0 + k[493]*y[IDX_EM] - k[755]*y[IDX_CH2I] + k[1495]*y[IDX_OI];
    data[3308] = 0.0 - k[67]*y[IDX_CH2I];
    data[3309] = 0.0 - k[761]*y[IDX_CH2I] - k[762]*y[IDX_CH2I];
    data[3310] = 0.0 - k[156]*y[IDX_CH2I] - k[913]*y[IDX_CH2I];
    data[3311] = 0.0 + k[521]*y[IDX_EM];
    data[3312] = 0.0 + k[1868]*y[IDX_OI];
    data[3313] = 0.0 - k[760]*y[IDX_CH2I];
    data[3314] = 0.0 - k[64]*y[IDX_CH2I] - k[756]*y[IDX_CH2I];
    data[3315] = 0.0 - k[766]*y[IDX_CH2I];
    data[3316] = 0.0 - k[70]*y[IDX_CH2I] - k[769]*y[IDX_CH2I];
    data[3317] = 0.0 - k[235]*y[IDX_CH2I] + k[1299]*y[IDX_H2COI];
    data[3318] = 0.0 - k[68]*y[IDX_CH2I] - k[767]*y[IDX_CH2I];
    data[3319] = 0.0 - k[754]*y[IDX_CH2I];
    data[3320] = 0.0 - k[69]*y[IDX_CH2I];
    data[3321] = 0.0 - k[65]*y[IDX_CH2I] + k[502]*y[IDX_EM] - k[757]*y[IDX_CH2I];
    data[3322] = 0.0 + k[61]*y[IDX_NOI];
    data[3323] = 0.0 - k[768]*y[IDX_CH2I];
    data[3324] = 0.0 - k[66]*y[IDX_CH2I] - k[758]*y[IDX_CH2I];
    data[3325] = 0.0 - k[71]*y[IDX_CH2I] - k[771]*y[IDX_CH2I] + k[820]*y[IDX_CH4I];
    data[3326] = 0.0 + k[481]*y[IDX_EM] + k[781]*y[IDX_NH3I];
    data[3327] = 0.0 - k[1627]*y[IDX_CH2I];
    data[3328] = 0.0 - k[14]*y[IDX_CII] - k[62]*y[IDX_C2II] - k[63]*y[IDX_CNII] -
        k[64]*y[IDX_COII] - k[65]*y[IDX_H2COII] - k[66]*y[IDX_H2OII] -
        k[67]*y[IDX_N2II] - k[68]*y[IDX_NH2II] - k[69]*y[IDX_OII] -
        k[70]*y[IDX_O2II] - k[71]*y[IDX_OHII] - k[114]*y[IDX_HII] -
        k[156]*y[IDX_H2II] - k[235]*y[IDX_NII] - k[381] - k[382] -
        k[608]*y[IDX_CII] - k[707]*y[IDX_CHII] - k[754]*y[IDX_C2HII] -
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
    data[3329] = 0.0 + k[384] + k[1649]*y[IDX_CH3I] + k[1649]*y[IDX_CH3I] +
        k[1650]*y[IDX_CNI] + k[1662]*y[IDX_O2I] + k[1668]*y[IDX_OHI] +
        k[1741]*y[IDX_HI] + k[1986];
    data[3330] = 0.0 - k[707]*y[IDX_CH2I] + k[710]*y[IDX_CH3OHI] + k[717]*y[IDX_H2COI];
    data[3331] = 0.0 - k[772]*y[IDX_CH2I];
    data[3332] = 0.0 - k[1623]*y[IDX_CH2I] + k[1650]*y[IDX_CH3I];
    data[3333] = 0.0 + k[717]*y[IDX_CHII] + k[1299]*y[IDX_NII] - k[1624]*y[IDX_CH2I] +
        k[1678]*y[IDX_CHI];
    data[3334] = 0.0 + k[390] + k[804]*y[IDX_C2II] + k[820]*y[IDX_OHII] -
        k[1622]*y[IDX_CH2I] + k[1995];
    data[3335] = 0.0 - k[1625]*y[IDX_CH2I] + k[1679]*y[IDX_CHI] + k[1752]*y[IDX_HI];
    data[3336] = 0.0 + k[61]*y[IDX_CH2II] - k[1629]*y[IDX_CH2I] - k[1630]*y[IDX_CH2I] -
        k[1631]*y[IDX_CH2I];
    data[3337] = 0.0 + k[1678]*y[IDX_H2COI] + k[1679]*y[IDX_HCOI] + k[1680]*y[IDX_HNOI] +
        k[1692]*y[IDX_O2HI] + k[1725]*y[IDX_H2I];
    data[3338] = 0.0 + k[781]*y[IDX_CH3II];
    data[3339] = 0.0 - k[759]*y[IDX_CH2I];
    data[3340] = 0.0 - k[1632]*y[IDX_CH2I] - k[1633]*y[IDX_CH2I] - k[1634]*y[IDX_CH2I] -
        k[1635]*y[IDX_CH2I] - k[1636]*y[IDX_CH2I] + k[1662]*y[IDX_CH3I];
    data[3341] = 0.0 - k[14]*y[IDX_CH2I] - k[608]*y[IDX_CH2I];
    data[3342] = 0.0 - k[1644]*y[IDX_CH2I] - k[1645]*y[IDX_CH2I];
    data[3343] = 0.0 - k[1641]*y[IDX_CH2I] - k[1642]*y[IDX_CH2I] - k[1643]*y[IDX_CH2I] +
        k[1668]*y[IDX_CH3I];
    data[3344] = 0.0 - k[1801]*y[IDX_CH2I] - k[1802]*y[IDX_CH2I] - k[1803]*y[IDX_CH2I];
    data[3345] = 0.0 + k[1185]*y[IDX_C2H4I] - k[1192]*y[IDX_CH2I] - k[1193]*y[IDX_CH2I] +
        k[1194]*y[IDX_CH2COI] + k[1220]*y[IDX_H2CSI];
    data[3346] = 0.0 - k[1028]*y[IDX_CH2I];
    data[3347] = 0.0 - k[114]*y[IDX_CH2I] - k[885]*y[IDX_CH2I];
    data[3348] = 0.0 + k[1495]*y[IDX_CH5II] - k[1637]*y[IDX_CH2I] - k[1638]*y[IDX_CH2I] -
        k[1639]*y[IDX_CH2I] - k[1640]*y[IDX_CH2I] + k[1868]*y[IDX_C2H2I] +
        k[1872]*y[IDX_C2H4I];
    data[3349] = 0.0 - k[1586]*y[IDX_CH2I] - k[1587]*y[IDX_CH2I] + k[2113]*y[IDX_H2I];
    data[3350] = 0.0 - k[763]*y[IDX_CH2I];
    data[3351] = 0.0 - k[1723]*y[IDX_CH2I] + k[1725]*y[IDX_CHI] + k[2113]*y[IDX_CI];
    data[3352] = 0.0 + k[481]*y[IDX_CH3II] + k[486]*y[IDX_CH3OH2II] + k[491]*y[IDX_CH4II]
        + k[493]*y[IDX_CH5II] + k[502]*y[IDX_H2COII] + k[521]*y[IDX_H3COII];
    data[3353] = 0.0 - k[1739]*y[IDX_CH2I] + k[1741]*y[IDX_CH3I] + k[1752]*y[IDX_HCOI];
    data[3354] = 0.0 + k[1740]*y[IDX_HI];
    data[3355] = 0.0 + k[1238]*y[IDX_HeII];
    data[3356] = 0.0 - k[1663]*y[IDX_CH3I];
    data[3357] = 0.0 + k[485]*y[IDX_EM];
    data[3358] = 0.0 - k[1658]*y[IDX_CH3I];
    data[3359] = 0.0 + k[1794]*y[IDX_NI] + k[1874]*y[IDX_OI];
    data[3360] = 0.0 + k[487]*y[IDX_EM] + k[488]*y[IDX_EM];
    data[3361] = 0.0 + k[387] + k[1198]*y[IDX_HeII] + k[1989];
    data[3362] = 0.0 + k[1626]*y[IDX_CH2I] - k[1655]*y[IDX_CH3I];
    data[3363] = 0.0 - k[1646]*y[IDX_CH3I];
    data[3364] = 0.0 + k[808]*y[IDX_CH4I];
    data[3365] = 0.0 + k[389] + k[794]*y[IDX_CH4II] + k[1200]*y[IDX_HeII] +
        k[1291]*y[IDX_NII] + k[1564]*y[IDX_SiII] + k[1992];
    data[3366] = 0.0 + k[676]*y[IDX_SII] + k[1792]*y[IDX_NI] + k[1873]*y[IDX_OI];
    data[3367] = 0.0 + k[492]*y[IDX_EM] + k[794]*y[IDX_CH3OHI] + k[795]*y[IDX_CH4I] +
        k[796]*y[IDX_CO2I] + k[797]*y[IDX_COI] + k[798]*y[IDX_H2COI] +
        k[799]*y[IDX_H2OI] + k[800]*y[IDX_H2SI] + k[801]*y[IDX_NH3I] +
        k[802]*y[IDX_OCSI];
    data[3368] = 0.0 + k[73]*y[IDX_CH3II];
    data[3369] = 0.0 + k[803]*y[IDX_CH4I];
    data[3370] = 0.0 + k[802]*y[IDX_CH4II];
    data[3371] = 0.0 + k[494]*y[IDX_EM] + k[495]*y[IDX_EM];
    data[3372] = 0.0 + k[1929]*y[IDX_OHI];
    data[3373] = 0.0 + k[811]*y[IDX_CH4I];
    data[3374] = 0.0 + k[1564]*y[IDX_CH3OHI];
    data[3375] = 0.0 + k[807]*y[IDX_CH4I];
    data[3376] = 0.0 + k[796]*y[IDX_CH4II];
    data[3377] = 0.0 - k[1656]*y[IDX_CH3I] + k[1833]*y[IDX_CH4I];
    data[3378] = 0.0 + k[1291]*y[IDX_CH3OHI];
    data[3379] = 0.0 + k[805]*y[IDX_CH4I];
    data[3380] = 0.0 + k[809]*y[IDX_CH4I];
    data[3381] = 0.0 + k[742]*y[IDX_H2COI];
    data[3382] = 0.0 + k[818]*y[IDX_CH4I];
    data[3383] = 0.0 + k[800]*y[IDX_CH4II] - k[1653]*y[IDX_CH3I];
    data[3384] = 0.0 + k[810]*y[IDX_CH4I];
    data[3385] = 0.0 + k[72]*y[IDX_HCOI] + k[73]*y[IDX_MgI] + k[74]*y[IDX_NOI] +
        k[2133]*y[IDX_EM];
    data[3386] = 0.0 + k[1621]*y[IDX_CH2I] + k[1621]*y[IDX_CH2I] + k[1622]*y[IDX_CH4I] +
        k[1622]*y[IDX_CH4I] + k[1624]*y[IDX_H2COI] + k[1625]*y[IDX_HCOI] +
        k[1626]*y[IDX_HNOI] + k[1643]*y[IDX_OHI] + k[1723]*y[IDX_H2I];
    data[3387] = 0.0 - k[115]*y[IDX_HII] - k[384] - k[385] - k[386] - k[609]*y[IDX_CII] -
        k[610]*y[IDX_CII] - k[790]*y[IDX_SII] - k[1029]*y[IDX_H3II] -
        k[1196]*y[IDX_HeII] - k[1588]*y[IDX_CI] - k[1646]*y[IDX_C2H3I] -
        k[1647]*y[IDX_CH3I] - k[1647]*y[IDX_CH3I] - k[1647]*y[IDX_CH3I] -
        k[1647]*y[IDX_CH3I] - k[1648]*y[IDX_CH3I] - k[1648]*y[IDX_CH3I] -
        k[1648]*y[IDX_CH3I] - k[1648]*y[IDX_CH3I] - k[1649]*y[IDX_CH3I] -
        k[1649]*y[IDX_CH3I] - k[1649]*y[IDX_CH3I] - k[1649]*y[IDX_CH3I] -
        k[1650]*y[IDX_CNI] - k[1651]*y[IDX_H2COI] - k[1652]*y[IDX_H2OI] -
        k[1653]*y[IDX_H2SI] - k[1654]*y[IDX_HCOI] - k[1655]*y[IDX_HNOI] -
        k[1656]*y[IDX_NH2I] - k[1657]*y[IDX_NH3I] - k[1658]*y[IDX_NO2I] -
        k[1659]*y[IDX_NOI] - k[1660]*y[IDX_O2I] - k[1661]*y[IDX_O2I] -
        k[1662]*y[IDX_O2I] - k[1663]*y[IDX_O2HI] - k[1664]*y[IDX_OI] -
        k[1665]*y[IDX_OI] - k[1666]*y[IDX_OHI] - k[1667]*y[IDX_OHI] -
        k[1668]*y[IDX_OHI] - k[1669]*y[IDX_SI] - k[1724]*y[IDX_H2I] -
        k[1741]*y[IDX_HI] - k[1804]*y[IDX_NI] - k[1805]*y[IDX_NI] -
        k[1806]*y[IDX_NI] - k[1986] - k[1987] - k[1988] - k[2109]*y[IDX_CNI] -
        k[2216];
    data[3388] = 0.0 + k[1839]*y[IDX_CH4I];
    data[3389] = 0.0 + k[676]*y[IDX_C2H4I] - k[790]*y[IDX_CH3I];
    data[3390] = 0.0 - k[1650]*y[IDX_CH3I] + k[1670]*y[IDX_CH4I] - k[2109]*y[IDX_CH3I];
    data[3391] = 0.0 + k[742]*y[IDX_CH2II] + k[798]*y[IDX_CH4II] + k[1624]*y[IDX_CH2I] -
        k[1651]*y[IDX_CH3I];
    data[3392] = 0.0 + k[795]*y[IDX_CH4II] + k[803]*y[IDX_C2II] + k[805]*y[IDX_C2HII] +
        k[807]*y[IDX_COII] + k[808]*y[IDX_CSII] + k[809]*y[IDX_H2COII] +
        k[810]*y[IDX_H2OII] + k[811]*y[IDX_HCNII] + k[818]*y[IDX_NH3II] +
        k[1205]*y[IDX_HeII] + k[1622]*y[IDX_CH2I] + k[1622]*y[IDX_CH2I] +
        k[1670]*y[IDX_CNI] + k[1671]*y[IDX_O2I] + k[1672]*y[IDX_OHI] +
        k[1673]*y[IDX_SI] + k[1742]*y[IDX_HI] + k[1833]*y[IDX_NH2I] +
        k[1839]*y[IDX_NHI] + k[1879]*y[IDX_OI] + k[1996];
    data[3393] = 0.0 + k[72]*y[IDX_CH3II] + k[1625]*y[IDX_CH2I] - k[1654]*y[IDX_CH3I];
    data[3394] = 0.0 + k[74]*y[IDX_CH3II] - k[1659]*y[IDX_CH3I];
    data[3395] = 0.0 + k[2115]*y[IDX_H2I];
    data[3396] = 0.0 + k[801]*y[IDX_CH4II] - k[1657]*y[IDX_CH3I];
    data[3397] = 0.0 - k[1660]*y[IDX_CH3I] - k[1661]*y[IDX_CH3I] - k[1662]*y[IDX_CH3I] +
        k[1671]*y[IDX_CH4I];
    data[3398] = 0.0 - k[609]*y[IDX_CH3I] - k[610]*y[IDX_CH3I];
    data[3399] = 0.0 - k[1669]*y[IDX_CH3I] + k[1673]*y[IDX_CH4I];
    data[3400] = 0.0 + k[1643]*y[IDX_CH2I] - k[1666]*y[IDX_CH3I] - k[1667]*y[IDX_CH3I] -
        k[1668]*y[IDX_CH3I] + k[1672]*y[IDX_CH4I] + k[1929]*y[IDX_C2H2I];
    data[3401] = 0.0 + k[1792]*y[IDX_C2H4I] + k[1794]*y[IDX_C2H5I] - k[1804]*y[IDX_CH3I]
        - k[1805]*y[IDX_CH3I] - k[1806]*y[IDX_CH3I];
    data[3402] = 0.0 - k[1196]*y[IDX_CH3I] + k[1198]*y[IDX_CH3CNI] +
        k[1200]*y[IDX_CH3OHI] + k[1205]*y[IDX_CH4I] + k[1238]*y[IDX_HCOOCH3I];
    data[3403] = 0.0 - k[1029]*y[IDX_CH3I];
    data[3404] = 0.0 - k[115]*y[IDX_CH3I];
    data[3405] = 0.0 - k[1664]*y[IDX_CH3I] - k[1665]*y[IDX_CH3I] + k[1873]*y[IDX_C2H4I] +
        k[1874]*y[IDX_C2H5I] + k[1879]*y[IDX_CH4I];
    data[3406] = 0.0 - k[1588]*y[IDX_CH3I];
    data[3407] = 0.0 + k[799]*y[IDX_CH4II] - k[1652]*y[IDX_CH3I];
    data[3408] = 0.0 + k[797]*y[IDX_CH4II];
    data[3409] = 0.0 + k[1723]*y[IDX_CH2I] - k[1724]*y[IDX_CH3I] + k[2115]*y[IDX_CHI];
    data[3410] = 0.0 + k[485]*y[IDX_CH3CNHII] + k[487]*y[IDX_CH3OH2II] +
        k[488]*y[IDX_CH3OH2II] + k[492]*y[IDX_CH4II] + k[494]*y[IDX_CH5II] +
        k[495]*y[IDX_CH5II] + k[2133]*y[IDX_CH3II];
    data[3411] = 0.0 + k[1740]*y[IDX_CH2COI] - k[1741]*y[IDX_CH3I] + k[1742]*y[IDX_CH4I];
    data[3412] = 0.0 + k[695]*y[IDX_CI];
    data[3413] = 0.0 - k[708]*y[IDX_CHII] - k[709]*y[IDX_CHII] - k[710]*y[IDX_CHII];
    data[3414] = 0.0 - k[56]*y[IDX_CHII];
    data[3415] = 0.0 + k[82]*y[IDX_CHI];
    data[3416] = 0.0 + k[701]*y[IDX_CI];
    data[3417] = 0.0 + k[83]*y[IDX_CHI];
    data[3418] = 0.0 + k[698]*y[IDX_CI];
    data[3419] = 0.0 - k[736]*y[IDX_CHII] - k[737]*y[IDX_CHII];
    data[3420] = 0.0 + k[696]*y[IDX_CI];
    data[3421] = 0.0 + k[689]*y[IDX_CI];
    data[3422] = 0.0 + k[88]*y[IDX_CHI];
    data[3423] = 0.0 + k[158]*y[IDX_CHI] + k[912]*y[IDX_CI];
    data[3424] = 0.0 - k[60]*y[IDX_CHII];
    data[3425] = 0.0 + k[1180]*y[IDX_HeII];
    data[3426] = 0.0 + k[693]*y[IDX_CI];
    data[3427] = 0.0 - k[727]*y[IDX_CHII];
    data[3428] = 0.0 + k[84]*y[IDX_CHI];
    data[3429] = 0.0 - k[714]*y[IDX_CHII];
    data[3430] = 0.0 + k[699]*y[IDX_CI];
    data[3431] = 0.0 - k[729]*y[IDX_CHII];
    data[3432] = 0.0 + k[91]*y[IDX_CHI];
    data[3433] = 0.0 + k[87]*y[IDX_CHI];
    data[3434] = 0.0 + k[89]*y[IDX_CHI];
    data[3435] = 0.0 + k[1327]*y[IDX_NI];
    data[3436] = 0.0 - k[706]*y[IDX_CHII] + k[1187]*y[IDX_HeII];
    data[3437] = 0.0 + k[90]*y[IDX_CHI];
    data[3438] = 0.0 + k[85]*y[IDX_CHI];
    data[3439] = 0.0 + k[1099]*y[IDX_HI] + k[1979];
    data[3440] = 0.0 - k[721]*y[IDX_CHII] - k[722]*y[IDX_CHII];
    data[3441] = 0.0 + k[86]*y[IDX_CHI] + k[690]*y[IDX_CI];
    data[3442] = 0.0 + k[92]*y[IDX_CHI] + k[702]*y[IDX_CI];
    data[3443] = 0.0 + k[1984];
    data[3444] = 0.0 + k[1330]*y[IDX_NI];
    data[3445] = 0.0 - k[705]*y[IDX_CHII];
    data[3446] = 0.0 - k[707]*y[IDX_CHII] + k[885]*y[IDX_HII] + k[1193]*y[IDX_HeII];
    data[3447] = 0.0 + k[1196]*y[IDX_HeII];
    data[3448] = 0.0 - k[55]*y[IDX_HCOI] - k[56]*y[IDX_MgI] - k[57]*y[IDX_NH3I] -
        k[58]*y[IDX_NOI] - k[59]*y[IDX_SI] - k[60]*y[IDX_SiI] - k[380] -
        k[477]*y[IDX_EM] - k[686]*y[IDX_CI] - k[705]*y[IDX_C2I] -
        k[706]*y[IDX_C2HI] - k[707]*y[IDX_CH2I] - k[708]*y[IDX_CH3OHI] -
        k[709]*y[IDX_CH3OHI] - k[710]*y[IDX_CH3OHI] - k[711]*y[IDX_CH4I] -
        k[712]*y[IDX_CHI] - k[713]*y[IDX_CNI] - k[714]*y[IDX_CO2I] -
        k[715]*y[IDX_H2COI] - k[716]*y[IDX_H2COI] - k[717]*y[IDX_H2COI] -
        k[718]*y[IDX_H2OI] - k[719]*y[IDX_H2OI] - k[720]*y[IDX_H2OI] -
        k[721]*y[IDX_H2SI] - k[722]*y[IDX_H2SI] - k[723]*y[IDX_HCNI] -
        k[724]*y[IDX_HCNI] - k[725]*y[IDX_HCNI] - k[726]*y[IDX_HCOI] -
        k[727]*y[IDX_HNCI] - k[728]*y[IDX_NI] - k[729]*y[IDX_NH2I] -
        k[730]*y[IDX_NH3I] - k[731]*y[IDX_NHI] - k[732]*y[IDX_O2I] -
        k[733]*y[IDX_O2I] - k[734]*y[IDX_O2I] - k[735]*y[IDX_OI] -
        k[736]*y[IDX_OCSI] - k[737]*y[IDX_OCSI] - k[738]*y[IDX_OHI] -
        k[739]*y[IDX_SI] - k[740]*y[IDX_SI] - k[937]*y[IDX_H2I] -
        k[1098]*y[IDX_HI] - k[1977] - k[2231];
    data[3449] = 0.0 - k[731]*y[IDX_CHII];
    data[3450] = 0.0 - k[713]*y[IDX_CHII];
    data[3451] = 0.0 - k[715]*y[IDX_CHII] - k[716]*y[IDX_CHII] - k[717]*y[IDX_CHII];
    data[3452] = 0.0 - k[711]*y[IDX_CHII] + k[1202]*y[IDX_HeII];
    data[3453] = 0.0 - k[55]*y[IDX_CHII] + k[626]*y[IDX_CII] - k[726]*y[IDX_CHII] +
        k[1237]*y[IDX_HeII];
    data[3454] = 0.0 - k[723]*y[IDX_CHII] - k[724]*y[IDX_CHII] - k[725]*y[IDX_CHII] +
        k[1234]*y[IDX_HeII];
    data[3455] = 0.0 - k[58]*y[IDX_CHII];
    data[3456] = 0.0 + k[15]*y[IDX_CII] + k[82]*y[IDX_C2II] + k[83]*y[IDX_CNII] +
        k[84]*y[IDX_COII] + k[85]*y[IDX_H2COII] + k[86]*y[IDX_H2OII] +
        k[87]*y[IDX_NII] + k[88]*y[IDX_N2II] + k[89]*y[IDX_NH2II] +
        k[90]*y[IDX_OII] + k[91]*y[IDX_O2II] + k[92]*y[IDX_OHII] +
        k[117]*y[IDX_HII] + k[158]*y[IDX_H2II] + k[211]*y[IDX_HeII] -
        k[712]*y[IDX_CHII] + k[2000];
    data[3457] = 0.0 - k[57]*y[IDX_CHII] - k[730]*y[IDX_CHII];
    data[3458] = 0.0 - k[732]*y[IDX_CHII] - k[733]*y[IDX_CHII] - k[734]*y[IDX_CHII];
    data[3459] = 0.0 + k[15]*y[IDX_CHI] + k[626]*y[IDX_HCOI] + k[934]*y[IDX_H2I] +
        k[2122]*y[IDX_HI];
    data[3460] = 0.0 - k[59]*y[IDX_CHII] - k[739]*y[IDX_CHII] - k[740]*y[IDX_CHII];
    data[3461] = 0.0 - k[738]*y[IDX_CHII];
    data[3462] = 0.0 - k[728]*y[IDX_CHII] + k[1327]*y[IDX_C2HII] + k[1330]*y[IDX_C2H2II];
    data[3463] = 0.0 + k[211]*y[IDX_CHI] + k[1180]*y[IDX_C2H2I] + k[1187]*y[IDX_C2HI] +
        k[1193]*y[IDX_CH2I] + k[1196]*y[IDX_CH3I] + k[1202]*y[IDX_CH4I] +
        k[1234]*y[IDX_HCNI] + k[1237]*y[IDX_HCOI];
    data[3464] = 0.0 + k[1027]*y[IDX_CI];
    data[3465] = 0.0 + k[117]*y[IDX_CHI] + k[885]*y[IDX_CH2I];
    data[3466] = 0.0 - k[735]*y[IDX_CHII];
    data[3467] = 0.0 - k[686]*y[IDX_CHII] + k[689]*y[IDX_CH5II] + k[690]*y[IDX_H2OII] +
        k[693]*y[IDX_HCNII] + k[694]*y[IDX_HCOII] + k[695]*y[IDX_HCO2II] +
        k[696]*y[IDX_HNOII] + k[698]*y[IDX_N2HII] + k[699]*y[IDX_NHII] +
        k[701]*y[IDX_O2HII] + k[702]*y[IDX_OHII] + k[912]*y[IDX_H2II] +
        k[1027]*y[IDX_H3II];
    data[3468] = 0.0 + k[694]*y[IDX_CI];
    data[3469] = 0.0 - k[718]*y[IDX_CHII] - k[719]*y[IDX_CHII] - k[720]*y[IDX_CHII];
    data[3470] = 0.0 + k[934]*y[IDX_CII] - k[937]*y[IDX_CHII];
    data[3471] = 0.0 - k[477]*y[IDX_CHII];
    data[3472] = 0.0 - k[1098]*y[IDX_CHII] + k[1099]*y[IDX_CH2II] + k[2122]*y[IDX_CII];
    data[3473] = 0.0 + k[415] + k[2037];
    data[3474] = 0.0 + k[1810]*y[IDX_NI];
    data[3475] = 0.0 + k[1826]*y[IDX_NI];
    data[3476] = 0.0 - k[1846]*y[IDX_NHI];
    data[3477] = 0.0 + k[1793]*y[IDX_NI];
    data[3478] = 0.0 + k[1773]*y[IDX_HI];
    data[3479] = 0.0 + k[1767]*y[IDX_HI];
    data[3480] = 0.0 + k[1716]*y[IDX_COI] + k[1757]*y[IDX_HI] + k[1815]*y[IDX_NI] +
        k[1898]*y[IDX_OI];
    data[3481] = 0.0 + k[1791]*y[IDX_NI];
    data[3482] = 0.0 + k[1289]*y[IDX_NII] + k[1290]*y[IDX_NII];
    data[3483] = 0.0 - k[1444]*y[IDX_NHI] - k[1445]*y[IDX_NHI];
    data[3484] = 0.0 - k[1459]*y[IDX_NHI];
    data[3485] = 0.0 - k[294]*y[IDX_NHI] + k[996]*y[IDX_H2OI];
    data[3486] = 0.0 + k[566]*y[IDX_EM] - k[1454]*y[IDX_NHI];
    data[3487] = 0.0 - k[1453]*y[IDX_NHI];
    data[3488] = 0.0 - k[1447]*y[IDX_NHI];
    data[3489] = 0.0 - k[296]*y[IDX_NHI];
    data[3490] = 0.0 + k[1817]*y[IDX_NI];
    data[3491] = 0.0 - k[168]*y[IDX_NHI] - k[929]*y[IDX_NHI];
    data[3492] = 0.0 - k[1451]*y[IDX_NHI];
    data[3493] = 0.0 + k[1387]*y[IDX_NH2II];
    data[3494] = 0.0 - k[295]*y[IDX_NHI] + k[1398]*y[IDX_NH2I] - k[1448]*y[IDX_NHI];
    data[3495] = 0.0 + k[260]*y[IDX_H2COI] + k[261]*y[IDX_H2OI] + k[262]*y[IDX_NH3I] +
        k[263]*y[IDX_NOI] + k[264]*y[IDX_O2I] + k[265]*y[IDX_SI] -
        k[1366]*y[IDX_NHI];
    data[3496] = 0.0 + k[425] + k[1388]*y[IDX_NH2II] + k[1398]*y[IDX_COII] +
        k[1409]*y[IDX_NH3II] + k[1601]*y[IDX_CI] + k[1656]*y[IDX_CH3I] +
        k[1760]*y[IDX_HI] + k[1836]*y[IDX_OHI] + k[1903]*y[IDX_OI] + k[2050];
    data[3497] = 0.0 - k[1458]*y[IDX_NHI];
    data[3498] = 0.0 - k[247]*y[IDX_NHI] + k[1289]*y[IDX_CH3OHI] + k[1290]*y[IDX_CH3OHI]
        + k[1298]*y[IDX_H2COI] + k[1300]*y[IDX_H2SI] + k[1302]*y[IDX_H2SI] +
        k[1307]*y[IDX_NH3I] - k[1308]*y[IDX_NHI];
    data[3499] = 0.0 + k[569]*y[IDX_EM] + k[767]*y[IDX_CH2I] + k[855]*y[IDX_CHI] +
        k[1374]*y[IDX_C2I] + k[1375]*y[IDX_C2HI] + k[1376]*y[IDX_H2COI] +
        k[1378]*y[IDX_H2OI] + k[1381]*y[IDX_H2SI] + k[1385]*y[IDX_HCNI] +
        k[1386]*y[IDX_HCOI] + k[1387]*y[IDX_HNCI] + k[1388]*y[IDX_NH2I] +
        k[1389]*y[IDX_NH3I] + k[1393]*y[IDX_SI] - k[1455]*y[IDX_NHI];
    data[3500] = 0.0 + k[1375]*y[IDX_NH2II];
    data[3501] = 0.0 - k[297]*y[IDX_NHI] - k[1457]*y[IDX_NHI];
    data[3502] = 0.0 - k[1449]*y[IDX_NHI];
    data[3503] = 0.0 + k[571]*y[IDX_EM] + k[1409]*y[IDX_NH2I] + k[1412]*y[IDX_C2I] -
        k[1456]*y[IDX_NHI];
    data[3504] = 0.0 + k[1300]*y[IDX_NII] + k[1302]*y[IDX_NII] + k[1381]*y[IDX_NH2II];
    data[3505] = 0.0 - k[1450]*y[IDX_NHI];
    data[3506] = 0.0 - k[1460]*y[IDX_NHI];
    data[3507] = 0.0 - k[1446]*y[IDX_NHI];
    data[3508] = 0.0 + k[1627]*y[IDX_CH2I];
    data[3509] = 0.0 + k[1374]*y[IDX_NH2II] + k[1412]*y[IDX_NH3II];
    data[3510] = 0.0 + k[767]*y[IDX_NH2II] + k[1627]*y[IDX_N2I] + k[1803]*y[IDX_NI];
    data[3511] = 0.0 + k[1656]*y[IDX_NH2I];
    data[3512] = 0.0 - k[731]*y[IDX_NHI];
    data[3513] = 0.0 - k[132]*y[IDX_HII] - k[168]*y[IDX_H2II] - k[247]*y[IDX_NII] -
        k[294]*y[IDX_CNII] - k[295]*y[IDX_COII] - k[296]*y[IDX_N2II] -
        k[297]*y[IDX_OII] - k[429] - k[430] - k[631]*y[IDX_CII] -
        k[731]*y[IDX_CHII] - k[929]*y[IDX_H2II] - k[1058]*y[IDX_H3II] -
        k[1256]*y[IDX_HeII] - k[1308]*y[IDX_NII] - k[1366]*y[IDX_NHII] -
        k[1444]*y[IDX_C2II] - k[1445]*y[IDX_C2II] - k[1446]*y[IDX_CH3II] -
        k[1447]*y[IDX_CH5II] - k[1448]*y[IDX_COII] - k[1449]*y[IDX_H2COII] -
        k[1450]*y[IDX_H2OII] - k[1451]*y[IDX_HCNII] - k[1452]*y[IDX_HCOII] -
        k[1453]*y[IDX_HNOII] - k[1454]*y[IDX_N2HII] - k[1455]*y[IDX_NH2II] -
        k[1456]*y[IDX_NH3II] - k[1457]*y[IDX_OII] - k[1458]*y[IDX_O2II] -
        k[1459]*y[IDX_O2HII] - k[1460]*y[IDX_OHII] - k[1461]*y[IDX_SII] -
        k[1602]*y[IDX_CI] - k[1603]*y[IDX_CI] - k[1730]*y[IDX_H2I] -
        k[1762]*y[IDX_HI] - k[1819]*y[IDX_NI] - k[1839]*y[IDX_CH4I] -
        k[1840]*y[IDX_CNI] - k[1841]*y[IDX_H2OI] - k[1842]*y[IDX_NH3I] -
        k[1843]*y[IDX_NHI] - k[1843]*y[IDX_NHI] - k[1843]*y[IDX_NHI] -
        k[1843]*y[IDX_NHI] - k[1844]*y[IDX_NHI] - k[1844]*y[IDX_NHI] -
        k[1844]*y[IDX_NHI] - k[1844]*y[IDX_NHI] - k[1845]*y[IDX_NHI] -
        k[1845]*y[IDX_NHI] - k[1845]*y[IDX_NHI] - k[1845]*y[IDX_NHI] -
        k[1846]*y[IDX_NO2I] - k[1847]*y[IDX_NOI] - k[1848]*y[IDX_NOI] -
        k[1849]*y[IDX_O2I] - k[1850]*y[IDX_O2I] - k[1851]*y[IDX_OI] -
        k[1852]*y[IDX_OI] - k[1853]*y[IDX_OHI] - k[1854]*y[IDX_OHI] -
        k[1855]*y[IDX_OHI] - k[1856]*y[IDX_SI] - k[1857]*y[IDX_SI] - k[2054] -
        k[2055] - k[2222];
    data[3514] = 0.0 - k[1461]*y[IDX_NHI];
    data[3515] = 0.0 - k[1840]*y[IDX_NHI];
    data[3516] = 0.0 + k[260]*y[IDX_NHII] + k[1298]*y[IDX_NII] + k[1376]*y[IDX_NH2II];
    data[3517] = 0.0 - k[1839]*y[IDX_NHI];
    data[3518] = 0.0 + k[1386]*y[IDX_NH2II] + k[1811]*y[IDX_NI];
    data[3519] = 0.0 + k[1385]*y[IDX_NH2II] + k[1890]*y[IDX_OI];
    data[3520] = 0.0 + k[263]*y[IDX_NHII] + k[1764]*y[IDX_HI] - k[1847]*y[IDX_NHI] -
        k[1848]*y[IDX_NHI];
    data[3521] = 0.0 + k[855]*y[IDX_NH2II] + k[1683]*y[IDX_NI];
    data[3522] = 0.0 + k[262]*y[IDX_NHII] + k[428] + k[1307]*y[IDX_NII] +
        k[1389]*y[IDX_NH2II] - k[1842]*y[IDX_NHI] + k[2053];
    data[3523] = 0.0 + k[264]*y[IDX_NHII] - k[1849]*y[IDX_NHI] - k[1850]*y[IDX_NHI];
    data[3524] = 0.0 - k[631]*y[IDX_NHI];
    data[3525] = 0.0 + k[265]*y[IDX_NHII] + k[1393]*y[IDX_NH2II] - k[1856]*y[IDX_NHI] -
        k[1857]*y[IDX_NHI];
    data[3526] = 0.0 + k[1828]*y[IDX_NI] + k[1836]*y[IDX_NH2I] - k[1853]*y[IDX_NHI] -
        k[1854]*y[IDX_NHI] - k[1855]*y[IDX_NHI];
    data[3527] = 0.0 + k[1683]*y[IDX_CHI] + k[1728]*y[IDX_H2I] + k[1791]*y[IDX_C2H3I] +
        k[1793]*y[IDX_C2H5I] + k[1803]*y[IDX_CH2I] + k[1810]*y[IDX_H2CNI] +
        k[1811]*y[IDX_HCOI] + k[1815]*y[IDX_HNOI] + k[1817]*y[IDX_HSI] -
        k[1819]*y[IDX_NHI] + k[1826]*y[IDX_O2HI] + k[1828]*y[IDX_OHI];
    data[3528] = 0.0 - k[1256]*y[IDX_NHI];
    data[3529] = 0.0 - k[1058]*y[IDX_NHI];
    data[3530] = 0.0 - k[132]*y[IDX_NHI];
    data[3531] = 0.0 - k[1851]*y[IDX_NHI] - k[1852]*y[IDX_NHI] + k[1890]*y[IDX_HCNI] +
        k[1898]*y[IDX_HNOI] + k[1903]*y[IDX_NH2I];
    data[3532] = 0.0 + k[1601]*y[IDX_NH2I] - k[1602]*y[IDX_NHI] - k[1603]*y[IDX_NHI];
    data[3533] = 0.0 - k[1452]*y[IDX_NHI];
    data[3534] = 0.0 + k[261]*y[IDX_NHII] + k[996]*y[IDX_CNII] + k[1378]*y[IDX_NH2II] -
        k[1841]*y[IDX_NHI];
    data[3535] = 0.0 + k[1716]*y[IDX_HNOI];
    data[3536] = 0.0 + k[1728]*y[IDX_NI] - k[1730]*y[IDX_NHI];
    data[3537] = 0.0 + k[566]*y[IDX_N2HII] + k[569]*y[IDX_NH2II] + k[571]*y[IDX_NH3II];
    data[3538] = 0.0 + k[1757]*y[IDX_HNOI] + k[1760]*y[IDX_NH2I] - k[1762]*y[IDX_NHI] +
        k[1764]*y[IDX_NOI] + k[1767]*y[IDX_NSI] + k[1773]*y[IDX_OCNI];
    data[3539] = 0.0 + k[1247]*y[IDX_HeII];
    data[3540] = 0.0 + k[1220]*y[IDX_HeII];
    data[3541] = 0.0 - k[343]*y[IDX_SII];
    data[3542] = 0.0 - k[344]*y[IDX_SII] + k[1287]*y[IDX_HeII];
    data[3543] = 0.0 + k[1269]*y[IDX_HeII];
    data[3544] = 0.0 + k[1259]*y[IDX_HeII];
    data[3545] = 0.0 + k[1270]*y[IDX_HeII];
    data[3546] = 0.0 - k[355]*y[IDX_SII] - k[1569]*y[IDX_SII];
    data[3547] = 0.0 + k[2005];
    data[3548] = 0.0 + k[640]*y[IDX_CII] + k[1272]*y[IDX_HeII];
    data[3549] = 0.0 - k[676]*y[IDX_SII];
    data[3550] = 0.0 + k[1214]*y[IDX_HeII];
    data[3551] = 0.0 - k[229]*y[IDX_SII];
    data[3552] = 0.0 + k[35]*y[IDX_SI];
    data[3553] = 0.0 + k[99]*y[IDX_SI];
    data[3554] = 0.0 + k[1266]*y[IDX_HeII] + k[1313]*y[IDX_NII] + k[1318]*y[IDX_N2II] +
        k[1479]*y[IDX_OII] - k[1552]*y[IDX_SII];
    data[3555] = 0.0 + k[346]*y[IDX_SI];
    data[3556] = 0.0 + k[258]*y[IDX_SI] + k[1316]*y[IDX_H2SI] + k[1318]*y[IDX_OCSI];
    data[3557] = 0.0 + k[902]*y[IDX_HII] + k[1249]*y[IDX_HeII];
    data[3558] = 0.0 + k[924]*y[IDX_H2SI];
    data[3559] = 0.0 - k[354]*y[IDX_SII];
    data[3560] = 0.0 + k[199]*y[IDX_SI];
    data[3561] = 0.0 + k[106]*y[IDX_SI];
    data[3562] = 0.0 + k[347]*y[IDX_SI] + k[1105]*y[IDX_HI] + k[1503]*y[IDX_OI] + k[2039];
    data[3563] = 0.0 + k[265]*y[IDX_SI];
    data[3564] = 0.0 + k[323]*y[IDX_SI];
    data[3565] = 0.0 + k[1302]*y[IDX_H2SI] + k[1313]*y[IDX_OCSI];
    data[3566] = 0.0 + k[270]*y[IDX_SI];
    data[3567] = 0.0 + k[41]*y[IDX_SI];
    data[3568] = 0.0 + k[1473]*y[IDX_H2SI] + k[1479]*y[IDX_OCSI];
    data[3569] = 0.0 + k[173]*y[IDX_SI];
    data[3570] = 0.0 + k[895]*y[IDX_HII] + k[924]*y[IDX_H2II] + k[1227]*y[IDX_HeII] +
        k[1302]*y[IDX_NII] + k[1316]*y[IDX_N2II] + k[1473]*y[IDX_OII] -
        k[1550]*y[IDX_SII] - k[1551]*y[IDX_SII];
    data[3571] = 0.0 + k[185]*y[IDX_SI];
    data[3572] = 0.0 + k[338]*y[IDX_SI];
    data[3573] = 0.0 - k[658]*y[IDX_SII];
    data[3574] = 0.0 - k[772]*y[IDX_SII];
    data[3575] = 0.0 - k[790]*y[IDX_SII];
    data[3576] = 0.0 + k[59]*y[IDX_SI];
    data[3577] = 0.0 - k[1461]*y[IDX_SII];
    data[3578] = 0.0 - k[205]*y[IDX_HCOI] - k[229]*y[IDX_MgI] - k[292]*y[IDX_NH3I] -
        k[303]*y[IDX_NOI] - k[343]*y[IDX_SiCI] - k[344]*y[IDX_SiSI] -
        k[354]*y[IDX_SiI] - k[355]*y[IDX_SiHI] - k[658]*y[IDX_C2I] -
        k[676]*y[IDX_C2H4I] - k[772]*y[IDX_CH2I] - k[790]*y[IDX_CH3I] -
        k[821]*y[IDX_CH4I] - k[822]*y[IDX_CH4I] - k[861]*y[IDX_CHI] -
        k[961]*y[IDX_H2I] - k[974]*y[IDX_H2COI] - k[975]*y[IDX_H2COI] -
        k[1163]*y[IDX_HCOI] - k[1461]*y[IDX_NHI] - k[1486]*y[IDX_O2I] -
        k[1548]*y[IDX_OHI] - k[1550]*y[IDX_H2SI] - k[1551]*y[IDX_H2SI] -
        k[1552]*y[IDX_OCSI] - k[1569]*y[IDX_SiHI] - k[2105]*y[IDX_CI] -
        k[2117]*y[IDX_H2I] - k[2143]*y[IDX_EM] - k[2264];
    data[3579] = 0.0 - k[974]*y[IDX_SII] - k[975]*y[IDX_SII];
    data[3580] = 0.0 - k[821]*y[IDX_SII] - k[822]*y[IDX_SII];
    data[3581] = 0.0 - k[205]*y[IDX_SII] - k[1163]*y[IDX_SII];
    data[3582] = 0.0 - k[303]*y[IDX_SII];
    data[3583] = 0.0 - k[861]*y[IDX_SII];
    data[3584] = 0.0 - k[292]*y[IDX_SII];
    data[3585] = 0.0 - k[1486]*y[IDX_SII];
    data[3586] = 0.0 + k[345]*y[IDX_SI] + k[640]*y[IDX_SOI];
    data[3587] = 0.0 + k[35]*y[IDX_C2II] + k[41]*y[IDX_C2HII] + k[59]*y[IDX_CHII] +
        k[99]*y[IDX_CNII] + k[106]*y[IDX_COII] + k[140]*y[IDX_HII] +
        k[173]*y[IDX_H2COII] + k[185]*y[IDX_H2OII] + k[199]*y[IDX_HCNII] +
        k[258]*y[IDX_N2II] + k[265]*y[IDX_NHII] + k[270]*y[IDX_NH2II] +
        k[323]*y[IDX_O2II] + k[338]*y[IDX_OHII] + k[345]*y[IDX_CII] +
        k[346]*y[IDX_H2SII] + k[347]*y[IDX_HSII] + k[444] + k[2073];
    data[3588] = 0.0 - k[1548]*y[IDX_SII];
    data[3589] = 0.0 + k[1214]*y[IDX_CSI] + k[1220]*y[IDX_H2CSI] + k[1227]*y[IDX_H2SI] +
        k[1247]*y[IDX_HS2I] + k[1249]*y[IDX_HSI] + k[1259]*y[IDX_NSI] +
        k[1266]*y[IDX_OCSI] + k[1269]*y[IDX_S2I] + k[1270]*y[IDX_SO2I] +
        k[1272]*y[IDX_SOI] + k[1287]*y[IDX_SiSI];
    data[3590] = 0.0 + k[140]*y[IDX_SI] + k[895]*y[IDX_H2SI] + k[902]*y[IDX_HSI];
    data[3591] = 0.0 + k[1503]*y[IDX_HSII];
    data[3592] = 0.0 - k[2105]*y[IDX_SII];
    data[3593] = 0.0 - k[961]*y[IDX_SII] - k[2117]*y[IDX_SII];
    data[3594] = 0.0 - k[2143]*y[IDX_SII];
    data[3595] = 0.0 + k[1105]*y[IDX_HSII];
    data[3596] = 0.0 + k[1800]*y[IDX_NI];
    data[3597] = 0.0 + k[471]*y[IDX_EM] + k[471]*y[IDX_EM] + k[1097]*y[IDX_HI];
    data[3598] = 0.0 + k[377] + k[1190]*y[IDX_HeII] + k[1798]*y[IDX_NI] + k[1974];
    data[3599] = 0.0 + k[407] + k[1230]*y[IDX_HeII] + k[2027];
    data[3600] = 0.0 - k[1709]*y[IDX_CNI];
    data[3601] = 0.0 + k[1342]*y[IDX_NI];
    data[3602] = 0.0 + k[376] + k[1189]*y[IDX_HeII] + k[1584]*y[IDX_CI] +
        k[1796]*y[IDX_NI] + k[1796]*y[IDX_NI] + k[1876]*y[IDX_OI] + k[1973];
    data[3603] = 0.0 + k[20]*y[IDX_CII] + k[423] + k[423] + k[1251]*y[IDX_HeII] +
        k[1580]*y[IDX_C2HI] + k[1598]*y[IDX_CI] + k[1759]*y[IDX_HI] + k[2047] +
        k[2047];
    data[3604] = 0.0 + k[1832]*y[IDX_NI];
    data[3605] = 0.0 + k[439] + k[635]*y[IDX_CII] + k[1263]*y[IDX_HeII] +
        k[1609]*y[IDX_CI] + k[1774]*y[IDX_HI] + k[1911]*y[IDX_OI] + k[2065];
    data[3606] = 0.0 + k[387] + k[665]*y[IDX_C2H2II] + k[1199]*y[IDX_HeII] + k[1989];
    data[3607] = 0.0 + k[1607]*y[IDX_CI];
    data[3608] = 0.0 - k[1708]*y[IDX_CNI];
    data[3609] = 0.0 - k[1715]*y[IDX_CNI];
    data[3610] = 0.0 + k[469]*y[IDX_EM];
    data[3611] = 0.0 - k[1702]*y[IDX_CNI];
    data[3612] = 0.0 + k[1809]*y[IDX_NI];
    data[3613] = 0.0 + k[1325]*y[IDX_NI];
    data[3614] = 0.0 - k[870]*y[IDX_CNI];
    data[3615] = 0.0 + k[36]*y[IDX_C2I] + k[47]*y[IDX_C2HI] + k[51]*y[IDX_CI] +
        k[63]*y[IDX_CH2I] + k[83]*y[IDX_CHI] + k[93]*y[IDX_COI] +
        k[94]*y[IDX_H2COI] + k[95]*y[IDX_HCNI] + k[96]*y[IDX_HCOI] +
        k[97]*y[IDX_NOI] + k[98]*y[IDX_O2I] + k[99]*y[IDX_SI] + k[191]*y[IDX_HI]
        + k[272]*y[IDX_NH2I] + k[294]*y[IDX_NHI] + k[326]*y[IDX_OI] +
        k[340]*y[IDX_OHI];
    data[3616] = 0.0 - k[869]*y[IDX_CNI];
    data[3617] = 0.0 - k[100]*y[IDX_CNI];
    data[3618] = 0.0 + k[539]*y[IDX_EM];
    data[3619] = 0.0 - k[159]*y[IDX_CNI] - k[917]*y[IDX_CNI];
    data[3620] = 0.0 - k[1701]*y[IDX_CNI];
    data[3621] = 0.0 + k[538]*y[IDX_EM] + k[652]*y[IDX_C2I] + k[679]*y[IDX_C2HI] +
        k[693]*y[IDX_CI] + k[760]*y[IDX_CH2I] + k[846]*y[IDX_CHI] +
        k[1002]*y[IDX_H2OI] + k[1110]*y[IDX_CO2I] + k[1111]*y[IDX_COI] +
        k[1112]*y[IDX_H2COI] + k[1113]*y[IDX_HCNI] + k[1114]*y[IDX_HCOI] +
        k[1116]*y[IDX_HNCI] + k[1117]*y[IDX_SI] + k[1403]*y[IDX_NH2I] +
        k[1451]*y[IDX_NHI] + k[1541]*y[IDX_OHI];
    data[3622] = 0.0 + k[414] + k[1116]*y[IDX_HCNII] - k[1707]*y[IDX_CNI] + k[2036];
    data[3623] = 0.0 + k[1110]*y[IDX_HCNII];
    data[3624] = 0.0 - k[1349]*y[IDX_CNI];
    data[3625] = 0.0 + k[272]*y[IDX_CNII] + k[1403]*y[IDX_HCNII];
    data[3626] = 0.0 - k[237]*y[IDX_CNI];
    data[3627] = 0.0 + k[661]*y[IDX_HCNI] + k[1327]*y[IDX_NI];
    data[3628] = 0.0 + k[47]*y[IDX_CNII] + k[679]*y[IDX_HCNII] + k[1580]*y[IDX_NCCNI] -
        k[2100]*y[IDX_CNI];
    data[3629] = 0.0 - k[1469]*y[IDX_CNI];
    data[3630] = 0.0 - k[1519]*y[IDX_CNI];
    data[3631] = 0.0 + k[1597]*y[IDX_CI];
    data[3632] = 0.0 + k[665]*y[IDX_CH3CNI];
    data[3633] = 0.0 + k[36]*y[IDX_CNII] + k[652]*y[IDX_HCNII] + k[1790]*y[IDX_NI];
    data[3634] = 0.0 + k[63]*y[IDX_CNII] + k[760]*y[IDX_HCNII] - k[1623]*y[IDX_CNI];
    data[3635] = 0.0 - k[1650]*y[IDX_CNI] - k[2109]*y[IDX_CNI];
    data[3636] = 0.0 - k[713]*y[IDX_CNI];
    data[3637] = 0.0 + k[294]*y[IDX_CNII] + k[1451]*y[IDX_HCNII] + k[1602]*y[IDX_CI] -
        k[1840]*y[IDX_CNI];
    data[3638] = 0.0 - k[100]*y[IDX_N2II] - k[159]*y[IDX_H2II] - k[237]*y[IDX_NII] -
        k[392] - k[713]*y[IDX_CHII] - k[869]*y[IDX_HNOII] - k[870]*y[IDX_O2HII]
        - k[917]*y[IDX_H2II] - k[1035]*y[IDX_H3II] - k[1207]*y[IDX_HeII] -
        k[1208]*y[IDX_HeII] - k[1349]*y[IDX_NHII] - k[1469]*y[IDX_OII] -
        k[1519]*y[IDX_OHII] - k[1590]*y[IDX_CI] - k[1623]*y[IDX_CH2I] -
        k[1650]*y[IDX_CH3I] - k[1670]*y[IDX_CH4I] - k[1701]*y[IDX_C2H2I] -
        k[1702]*y[IDX_C2H4I] - k[1703]*y[IDX_CNI] - k[1703]*y[IDX_CNI] -
        k[1703]*y[IDX_CNI] - k[1703]*y[IDX_CNI] - k[1704]*y[IDX_H2COI] -
        k[1705]*y[IDX_HCNI] - k[1706]*y[IDX_HCOI] - k[1707]*y[IDX_HNCI] -
        k[1708]*y[IDX_HNOI] - k[1709]*y[IDX_NO2I] - k[1710]*y[IDX_NOI] -
        k[1711]*y[IDX_NOI] - k[1712]*y[IDX_O2I] - k[1713]*y[IDX_O2I] -
        k[1714]*y[IDX_SI] - k[1715]*y[IDX_SiH4I] - k[1726]*y[IDX_H2I] -
        k[1807]*y[IDX_NI] - k[1838]*y[IDX_NH3I] - k[1840]*y[IDX_NHI] -
        k[1880]*y[IDX_OI] - k[1881]*y[IDX_OI] - k[1932]*y[IDX_OHI] -
        k[1933]*y[IDX_OHI] - k[2001] - k[2100]*y[IDX_C2HI] - k[2109]*y[IDX_CH3I]
        - k[2220];
    data[3639] = 0.0 + k[94]*y[IDX_CNII] + k[1112]*y[IDX_HCNII] - k[1704]*y[IDX_CNI];
    data[3640] = 0.0 - k[1670]*y[IDX_CNI];
    data[3641] = 0.0 + k[96]*y[IDX_CNII] + k[1114]*y[IDX_HCNII] - k[1706]*y[IDX_CNI];
    data[3642] = 0.0 + k[95]*y[IDX_CNII] + k[408] + k[661]*y[IDX_C2HII] +
        k[1113]*y[IDX_HCNII] - k[1705]*y[IDX_CNI] + k[1750]*y[IDX_HI] +
        k[1889]*y[IDX_OI] + k[1939]*y[IDX_OHI] + k[2028];
    data[3643] = 0.0 + k[97]*y[IDX_CNII] + k[1604]*y[IDX_CI] - k[1710]*y[IDX_CNI] -
        k[1711]*y[IDX_CNI];
    data[3644] = 0.0 + k[83]*y[IDX_CNII] + k[846]*y[IDX_HCNII] + k[1682]*y[IDX_NI];
    data[3645] = 0.0 - k[1838]*y[IDX_CNI];
    data[3646] = 0.0 + k[98]*y[IDX_CNII] - k[1712]*y[IDX_CNI] - k[1713]*y[IDX_CNI];
    data[3647] = 0.0 + k[20]*y[IDX_NCCNI] + k[635]*y[IDX_OCNI];
    data[3648] = 0.0 + k[99]*y[IDX_CNII] + k[1117]*y[IDX_HCNII] - k[1714]*y[IDX_CNI];
    data[3649] = 0.0 + k[340]*y[IDX_CNII] + k[1541]*y[IDX_HCNII] - k[1932]*y[IDX_CNI] -
        k[1933]*y[IDX_CNI] + k[1939]*y[IDX_HCNI];
    data[3650] = 0.0 + k[1325]*y[IDX_C2II] + k[1327]*y[IDX_C2HII] + k[1342]*y[IDX_SiCII]
        + k[1682]*y[IDX_CHI] + k[1790]*y[IDX_C2I] + k[1796]*y[IDX_C2NI] +
        k[1796]*y[IDX_C2NI] + k[1798]*y[IDX_C3NI] + k[1800]*y[IDX_C4NI] -
        k[1807]*y[IDX_CNI] + k[1809]*y[IDX_CSI] + k[1832]*y[IDX_SiCI] +
        k[2102]*y[IDX_CI];
    data[3651] = 0.0 + k[1189]*y[IDX_C2NI] + k[1190]*y[IDX_C3NI] + k[1199]*y[IDX_CH3CNI]
        - k[1207]*y[IDX_CNI] - k[1208]*y[IDX_CNI] + k[1230]*y[IDX_HC3NI] +
        k[1251]*y[IDX_NCCNI] + k[1263]*y[IDX_OCNI];
    data[3652] = 0.0 - k[1035]*y[IDX_CNI];
    data[3653] = 0.0 + k[326]*y[IDX_CNII] + k[1876]*y[IDX_C2NI] - k[1880]*y[IDX_CNI] -
        k[1881]*y[IDX_CNI] + k[1889]*y[IDX_HCNI] + k[1911]*y[IDX_OCNI];
    data[3654] = 0.0 + k[51]*y[IDX_CNII] + k[693]*y[IDX_HCNII] + k[1584]*y[IDX_C2NI] -
        k[1590]*y[IDX_CNI] + k[1597]*y[IDX_N2I] + k[1598]*y[IDX_NCCNI] +
        k[1602]*y[IDX_NHI] + k[1604]*y[IDX_NOI] + k[1607]*y[IDX_NSI] +
        k[1609]*y[IDX_OCNI] + k[2102]*y[IDX_NI];
    data[3655] = 0.0 + k[1002]*y[IDX_HCNII];
    data[3656] = 0.0 + k[93]*y[IDX_CNII] + k[1111]*y[IDX_HCNII];
    data[3657] = 0.0 - k[1726]*y[IDX_CNI];
    data[3658] = 0.0 + k[469]*y[IDX_C2NII] + k[471]*y[IDX_C2N2II] + k[471]*y[IDX_C2N2II]
        + k[538]*y[IDX_HCNII] + k[539]*y[IDX_HCNHII];
    data[3659] = 0.0 + k[191]*y[IDX_CNII] + k[1097]*y[IDX_C2N2II] + k[1750]*y[IDX_HCNI] +
        k[1759]*y[IDX_NCCNI] + k[1774]*y[IDX_OCNI];
    data[3660] = 0.0 + k[2367] + k[2368] + k[2369] + k[2370];
    data[3661] = 0.0 + k[2032] + k[2032];
    data[3662] = 0.0 + k[1786]*y[IDX_HCOI];
    data[3663] = 0.0 + k[1628]*y[IDX_CH2I] + k[1658]*y[IDX_CH3I];
    data[3664] = 0.0 + k[793]*y[IDX_CH3OHI];
    data[3665] = 0.0 + k[1874]*y[IDX_OI];
    data[3666] = 0.0 + k[490]*y[IDX_EM] - k[969]*y[IDX_H2COI];
    data[3667] = 0.0 + k[1782]*y[IDX_HCOI];
    data[3668] = 0.0 + k[773]*y[IDX_CH2I];
    data[3669] = 0.0 + k[1576]*y[IDX_O2I];
    data[3670] = 0.0 + k[388] + k[709]*y[IDX_CHII] + k[793]*y[IDX_S2II] +
        k[1078]*y[IDX_H3COII] + k[1990];
    data[3671] = 0.0 - k[970]*y[IDX_H2COI];
    data[3672] = 0.0 + k[1559]*y[IDX_SOII] + k[1872]*y[IDX_OI];
    data[3673] = 0.0 - k[76]*y[IDX_H2COI] - k[798]*y[IDX_H2COI];
    data[3674] = 0.0 + k[222]*y[IDX_H2COII];
    data[3675] = 0.0 + k[1559]*y[IDX_C2H4I];
    data[3676] = 0.0 - k[973]*y[IDX_H2COI];
    data[3677] = 0.0 - k[94]*y[IDX_H2COI] - k[865]*y[IDX_H2COI];
    data[3678] = 0.0 - k[1323]*y[IDX_H2COI];
    data[3679] = 0.0 - k[971]*y[IDX_H2COI];
    data[3680] = 0.0 - k[827]*y[IDX_H2COI];
    data[3681] = 0.0 - k[252]*y[IDX_H2COI] - k[1314]*y[IDX_H2COI];
    data[3682] = 0.0 - k[1132]*y[IDX_H2COI] - k[1133]*y[IDX_H2COI];
    data[3683] = 0.0 - k[161]*y[IDX_H2COI] - k[921]*y[IDX_H2COI];
    data[3684] = 0.0 + k[524]*y[IDX_EM] + k[844]*y[IDX_CHI] + k[1001]*y[IDX_H2OI] +
        k[1078]*y[IDX_CH3OHI] + k[1079]*y[IDX_H2SI] + k[1122]*y[IDX_HCNI] +
        k[1166]*y[IDX_HNCI] + k[1401]*y[IDX_NH2I] + k[1427]*y[IDX_NH3I];
    data[3685] = 0.0 + k[349]*y[IDX_H2COII];
    data[3686] = 0.0 - k[1112]*y[IDX_H2COI];
    data[3687] = 0.0 + k[1166]*y[IDX_H3COII];
    data[3688] = 0.0 - k[101]*y[IDX_H2COI] - k[871]*y[IDX_H2COI];
    data[3689] = 0.0 - k[260]*y[IDX_H2COI] - k[1354]*y[IDX_H2COI] - k[1355]*y[IDX_H2COI];
    data[3690] = 0.0 + k[1401]*y[IDX_H3COII];
    data[3691] = 0.0 - k[174]*y[IDX_H2COI] - k[972]*y[IDX_H2COI];
    data[3692] = 0.0 - k[239]*y[IDX_H2COI] - k[1298]*y[IDX_H2COI] - k[1299]*y[IDX_H2COI];
    data[3693] = 0.0 - k[1376]*y[IDX_H2COI] - k[1377]*y[IDX_H2COI];
    data[3694] = 0.0 - k[660]*y[IDX_H2COI];
    data[3695] = 0.0 - k[311]*y[IDX_H2COI] - k[1471]*y[IDX_H2COI];
    data[3696] = 0.0 + k[65]*y[IDX_CH2I] + k[85]*y[IDX_CHI] + k[173]*y[IDX_SI] +
        k[202]*y[IDX_HCOI] + k[222]*y[IDX_MgI] + k[284]*y[IDX_NH3I] +
        k[298]*y[IDX_NOI] + k[349]*y[IDX_SiI] - k[966]*y[IDX_H2COI] +
        k[2136]*y[IDX_EM];
    data[3697] = 0.0 - k[742]*y[IDX_H2COI];
    data[3698] = 0.0 - k[1413]*y[IDX_H2COI];
    data[3699] = 0.0 + k[1079]*y[IDX_H3COII];
    data[3700] = 0.0 - k[178]*y[IDX_H2COI] - k[979]*y[IDX_H2COI];
    data[3701] = 0.0 - k[331]*y[IDX_H2COI] - k[1522]*y[IDX_H2COI];
    data[3702] = 0.0 - k[777]*y[IDX_H2COI];
    data[3703] = 0.0 - k[42]*y[IDX_H2COI];
    data[3704] = 0.0 + k[65]*y[IDX_H2COII] + k[773]*y[IDX_SiOII] - k[1624]*y[IDX_H2COI] +
        k[1628]*y[IDX_NO2I] + k[1629]*y[IDX_NOI] + k[1635]*y[IDX_O2I] +
        k[1641]*y[IDX_OHI];
    data[3705] = 0.0 - k[1651]*y[IDX_H2COI] + k[1658]*y[IDX_NO2I] + k[1660]*y[IDX_O2I] +
        k[1665]*y[IDX_OI] + k[1667]*y[IDX_OHI];
    data[3706] = 0.0 + k[709]*y[IDX_CH3OHI] - k[715]*y[IDX_H2COI] - k[716]*y[IDX_H2COI] -
        k[717]*y[IDX_H2COI];
    data[3707] = 0.0 - k[974]*y[IDX_H2COI] - k[975]*y[IDX_H2COI];
    data[3708] = 0.0 - k[1704]*y[IDX_H2COI];
    data[3709] = 0.0 - k[16]*y[IDX_CII] - k[42]*y[IDX_C2H2II] - k[76]*y[IDX_CH4II] -
        k[94]*y[IDX_CNII] - k[101]*y[IDX_COII] - k[119]*y[IDX_HII] -
        k[161]*y[IDX_H2II] - k[174]*y[IDX_O2II] - k[178]*y[IDX_H2OII] -
        k[212]*y[IDX_HeII] - k[239]*y[IDX_NII] - k[252]*y[IDX_N2II] -
        k[260]*y[IDX_NHII] - k[311]*y[IDX_OII] - k[331]*y[IDX_OHII] - k[399] -
        k[617]*y[IDX_CII] - k[618]*y[IDX_CII] - k[660]*y[IDX_C2HII] -
        k[715]*y[IDX_CHII] - k[716]*y[IDX_CHII] - k[717]*y[IDX_CHII] -
        k[742]*y[IDX_CH2II] - k[777]*y[IDX_CH3II] - k[798]*y[IDX_CH4II] -
        k[827]*y[IDX_CH5II] - k[865]*y[IDX_CNII] - k[871]*y[IDX_COII] -
        k[892]*y[IDX_HII] - k[893]*y[IDX_HII] - k[921]*y[IDX_H2II] -
        k[966]*y[IDX_H2COII] - k[969]*y[IDX_CH3OH2II] - k[970]*y[IDX_H3SII] -
        k[971]*y[IDX_HNOII] - k[972]*y[IDX_O2II] - k[973]*y[IDX_O2HII] -
        k[974]*y[IDX_SII] - k[975]*y[IDX_SII] - k[979]*y[IDX_H2OII] -
        k[1041]*y[IDX_H3II] - k[1086]*y[IDX_H3OII] - k[1112]*y[IDX_HCNII] -
        k[1132]*y[IDX_HCNHII] - k[1133]*y[IDX_HCNHII] - k[1141]*y[IDX_HCOII] -
        k[1216]*y[IDX_HeII] - k[1217]*y[IDX_HeII] - k[1218]*y[IDX_HeII] -
        k[1298]*y[IDX_NII] - k[1299]*y[IDX_NII] - k[1314]*y[IDX_N2II] -
        k[1323]*y[IDX_N2HII] - k[1354]*y[IDX_NHII] - k[1355]*y[IDX_NHII] -
        k[1376]*y[IDX_NH2II] - k[1377]*y[IDX_NH2II] - k[1413]*y[IDX_NH3II] -
        k[1471]*y[IDX_OII] - k[1522]*y[IDX_OHII] - k[1624]*y[IDX_CH2I] -
        k[1651]*y[IDX_CH3I] - k[1678]*y[IDX_CHI] - k[1704]*y[IDX_CNI] -
        k[1747]*y[IDX_HI] - k[1886]*y[IDX_OI] - k[1937]*y[IDX_OHI] - k[2011] -
        k[2012] - k[2013] - k[2014] - k[2208];
    data[3710] = 0.0 + k[202]*y[IDX_H2COII] + k[1781]*y[IDX_HCOI] + k[1781]*y[IDX_HCOI] +
        k[1782]*y[IDX_HNOI] + k[1786]*y[IDX_O2HI];
    data[3711] = 0.0 + k[1122]*y[IDX_H3COII];
    data[3712] = 0.0 + k[298]*y[IDX_H2COII] + k[1629]*y[IDX_CH2I];
    data[3713] = 0.0 + k[85]*y[IDX_H2COII] + k[844]*y[IDX_H3COII] - k[1678]*y[IDX_H2COI];
    data[3714] = 0.0 + k[284]*y[IDX_H2COII] + k[1427]*y[IDX_H3COII];
    data[3715] = 0.0 - k[1086]*y[IDX_H2COI];
    data[3716] = 0.0 + k[1576]*y[IDX_C2H3I] + k[1635]*y[IDX_CH2I] + k[1660]*y[IDX_CH3I];
    data[3717] = 0.0 - k[16]*y[IDX_H2COI] - k[617]*y[IDX_H2COI] - k[618]*y[IDX_H2COI];
    data[3718] = 0.0 + k[173]*y[IDX_H2COII];
    data[3719] = 0.0 + k[1641]*y[IDX_CH2I] + k[1667]*y[IDX_CH3I] - k[1937]*y[IDX_H2COI];
    data[3720] = 0.0 - k[212]*y[IDX_H2COI] - k[1216]*y[IDX_H2COI] - k[1217]*y[IDX_H2COI]
        - k[1218]*y[IDX_H2COI];
    data[3721] = 0.0 - k[1041]*y[IDX_H2COI];
    data[3722] = 0.0 - k[119]*y[IDX_H2COI] - k[892]*y[IDX_H2COI] - k[893]*y[IDX_H2COI];
    data[3723] = 0.0 + k[1665]*y[IDX_CH3I] + k[1872]*y[IDX_C2H4I] + k[1874]*y[IDX_C2H5I]
        - k[1886]*y[IDX_H2COI];
    data[3724] = 0.0 - k[1141]*y[IDX_H2COI];
    data[3725] = 0.0 + k[1001]*y[IDX_H3COII];
    data[3726] = 0.0 + k[490]*y[IDX_CH3OH2II] + k[524]*y[IDX_H3COII] +
        k[2136]*y[IDX_H2COII];
    data[3727] = 0.0 - k[1747]*y[IDX_H2COI];
    data[3728] = 0.0 + k[2303] + k[2304] + k[2305] + k[2306];
    data[3729] = 0.0 + k[411];
    data[3730] = 0.0 + k[832]*y[IDX_CH5II];
    data[3731] = 0.0 + k[1663]*y[IDX_CH3I];
    data[3732] = 0.0 + k[1023]*y[IDX_H3II] + k[1024]*y[IDX_H3II];
    data[3733] = 0.0 + k[1655]*y[IDX_CH3I];
    data[3734] = 0.0 + k[789]*y[IDX_CH3II] + k[836]*y[IDX_CH5II];
    data[3735] = 0.0 + k[1646]*y[IDX_CH3I];
    data[3736] = 0.0 - k[812]*y[IDX_CH4I];
    data[3737] = 0.0 - k[808]*y[IDX_CH4I];
    data[3738] = 0.0 + k[776]*y[IDX_CH3II];
    data[3739] = 0.0 + k[75]*y[IDX_C2H2I] + k[76]*y[IDX_H2COI] + k[77]*y[IDX_H2SI] +
        k[78]*y[IDX_NH3I] + k[79]*y[IDX_O2I] + k[80]*y[IDX_OCSI] -
        k[795]*y[IDX_CH4I];
    data[3740] = 0.0 + k[834]*y[IDX_CH5II];
    data[3741] = 0.0 - k[803]*y[IDX_CH4I] - k[804]*y[IDX_CH4I];
    data[3742] = 0.0 - k[817]*y[IDX_CH4I];
    data[3743] = 0.0 + k[80]*y[IDX_CH4II];
    data[3744] = 0.0 - k[813]*y[IDX_CH4I];
    data[3745] = 0.0 + k[496]*y[IDX_EM] + k[689]*y[IDX_CI] + k[755]*y[IDX_CH2I] +
        k[823]*y[IDX_C2I] + k[824]*y[IDX_C2HI] + k[825]*y[IDX_CO2I] +
        k[826]*y[IDX_COI] + k[827]*y[IDX_H2COI] + k[828]*y[IDX_H2OI] +
        k[829]*y[IDX_H2SI] + k[830]*y[IDX_HCNI] + k[831]*y[IDX_HCOI] +
        k[832]*y[IDX_HClI] + k[833]*y[IDX_HNCI] + k[834]*y[IDX_MgI] +
        k[835]*y[IDX_SI] + k[836]*y[IDX_SiH4I] + k[840]*y[IDX_CHI] +
        k[1397]*y[IDX_NH2I] + k[1422]*y[IDX_NH3I] + k[1447]*y[IDX_NHI] +
        k[1538]*y[IDX_OHI];
    data[3746] = 0.0 - k[815]*y[IDX_CH4I] - k[816]*y[IDX_CH4I];
    data[3747] = 0.0 - k[157]*y[IDX_CH4I] - k[914]*y[IDX_CH4I] - k[915]*y[IDX_CH4I];
    data[3748] = 0.0 + k[75]*y[IDX_CH4II];
    data[3749] = 0.0 - k[811]*y[IDX_CH4I];
    data[3750] = 0.0 + k[833]*y[IDX_CH5II];
    data[3751] = 0.0 - k[81]*y[IDX_CH4I] - k[807]*y[IDX_CH4I];
    data[3752] = 0.0 - k[814]*y[IDX_CH4I];
    data[3753] = 0.0 + k[825]*y[IDX_CH5II];
    data[3754] = 0.0 + k[1397]*y[IDX_CH5II] + k[1656]*y[IDX_CH3I] - k[1833]*y[IDX_CH4I];
    data[3755] = 0.0 - k[236]*y[IDX_CH4I] - k[1293]*y[IDX_CH4I] - k[1294]*y[IDX_CH4I] -
        k[1295]*y[IDX_CH4I];
    data[3756] = 0.0 - k[805]*y[IDX_CH4I];
    data[3757] = 0.0 + k[824]*y[IDX_CH5II];
    data[3758] = 0.0 - k[309]*y[IDX_CH4I] - k[1468]*y[IDX_CH4I];
    data[3759] = 0.0 - k[809]*y[IDX_CH4I];
    data[3760] = 0.0 - k[818]*y[IDX_CH4I];
    data[3761] = 0.0 + k[77]*y[IDX_CH4II] + k[829]*y[IDX_CH5II] + k[1653]*y[IDX_CH3I];
    data[3762] = 0.0 - k[810]*y[IDX_CH4I];
    data[3763] = 0.0 - k[819]*y[IDX_CH4I] - k[820]*y[IDX_CH4I];
    data[3764] = 0.0 + k[776]*y[IDX_CH3OHI] + k[777]*y[IDX_H2COI] + k[789]*y[IDX_SiH4I];
    data[3765] = 0.0 - k[806]*y[IDX_CH4I];
    data[3766] = 0.0 + k[823]*y[IDX_CH5II];
    data[3767] = 0.0 + k[755]*y[IDX_CH5II] - k[1622]*y[IDX_CH4I];
    data[3768] = 0.0 + k[1646]*y[IDX_C2H3I] + k[1649]*y[IDX_CH3I] + k[1649]*y[IDX_CH3I] +
        k[1651]*y[IDX_H2COI] + k[1652]*y[IDX_H2OI] + k[1653]*y[IDX_H2SI] +
        k[1654]*y[IDX_HCOI] + k[1655]*y[IDX_HNOI] + k[1656]*y[IDX_NH2I] +
        k[1657]*y[IDX_NH3I] + k[1663]*y[IDX_O2HI] + k[1666]*y[IDX_OHI] +
        k[1724]*y[IDX_H2I];
    data[3769] = 0.0 - k[711]*y[IDX_CH4I];
    data[3770] = 0.0 + k[1447]*y[IDX_CH5II] - k[1839]*y[IDX_CH4I];
    data[3771] = 0.0 - k[821]*y[IDX_CH4I] - k[822]*y[IDX_CH4I];
    data[3772] = 0.0 - k[1670]*y[IDX_CH4I];
    data[3773] = 0.0 + k[76]*y[IDX_CH4II] + k[777]*y[IDX_CH3II] + k[827]*y[IDX_CH5II] +
        k[1651]*y[IDX_CH3I];
    data[3774] = 0.0 - k[81]*y[IDX_COII] - k[116]*y[IDX_HII] - k[157]*y[IDX_H2II] -
        k[210]*y[IDX_HeII] - k[236]*y[IDX_NII] - k[309]*y[IDX_OII] - k[390] -
        k[614]*y[IDX_CII] - k[711]*y[IDX_CHII] - k[795]*y[IDX_CH4II] -
        k[803]*y[IDX_C2II] - k[804]*y[IDX_C2II] - k[805]*y[IDX_C2HII] -
        k[806]*y[IDX_C2H2II] - k[807]*y[IDX_COII] - k[808]*y[IDX_CSII] -
        k[809]*y[IDX_H2COII] - k[810]*y[IDX_H2OII] - k[811]*y[IDX_HCNII] -
        k[812]*y[IDX_HCO2II] - k[813]*y[IDX_HNOII] - k[814]*y[IDX_HSII] -
        k[815]*y[IDX_N2II] - k[816]*y[IDX_N2II] - k[817]*y[IDX_N2HII] -
        k[818]*y[IDX_NH3II] - k[819]*y[IDX_OHII] - k[820]*y[IDX_OHII] -
        k[821]*y[IDX_SII] - k[822]*y[IDX_SII] - k[890]*y[IDX_HII] -
        k[914]*y[IDX_H2II] - k[915]*y[IDX_H2II] - k[1033]*y[IDX_H3II] -
        k[1202]*y[IDX_HeII] - k[1203]*y[IDX_HeII] - k[1204]*y[IDX_HeII] -
        k[1205]*y[IDX_HeII] - k[1293]*y[IDX_NII] - k[1294]*y[IDX_NII] -
        k[1295]*y[IDX_NII] - k[1468]*y[IDX_OII] - k[1622]*y[IDX_CH2I] -
        k[1670]*y[IDX_CNI] - k[1671]*y[IDX_O2I] - k[1672]*y[IDX_OHI] -
        k[1673]*y[IDX_SI] - k[1676]*y[IDX_CHI] - k[1742]*y[IDX_HI] -
        k[1833]*y[IDX_NH2I] - k[1839]*y[IDX_NHI] - k[1879]*y[IDX_OI] - k[1995] -
        k[1996] - k[1997] - k[1998] - k[2217];
    data[3775] = 0.0 + k[831]*y[IDX_CH5II] + k[1654]*y[IDX_CH3I];
    data[3776] = 0.0 + k[830]*y[IDX_CH5II];
    data[3777] = 0.0 + k[840]*y[IDX_CH5II] - k[1676]*y[IDX_CH4I];
    data[3778] = 0.0 + k[78]*y[IDX_CH4II] + k[1422]*y[IDX_CH5II] + k[1657]*y[IDX_CH3I];
    data[3779] = 0.0 + k[79]*y[IDX_CH4II] - k[1671]*y[IDX_CH4I];
    data[3780] = 0.0 - k[614]*y[IDX_CH4I];
    data[3781] = 0.0 + k[835]*y[IDX_CH5II] - k[1673]*y[IDX_CH4I];
    data[3782] = 0.0 + k[1538]*y[IDX_CH5II] + k[1666]*y[IDX_CH3I] - k[1672]*y[IDX_CH4I];
    data[3783] = 0.0 - k[210]*y[IDX_CH4I] - k[1202]*y[IDX_CH4I] - k[1203]*y[IDX_CH4I] -
        k[1204]*y[IDX_CH4I] - k[1205]*y[IDX_CH4I];
    data[3784] = 0.0 + k[1023]*y[IDX_C2H5OHI] + k[1024]*y[IDX_C2H5OHI] -
        k[1033]*y[IDX_CH4I];
    data[3785] = 0.0 - k[116]*y[IDX_CH4I] - k[890]*y[IDX_CH4I];
    data[3786] = 0.0 - k[1879]*y[IDX_CH4I];
    data[3787] = 0.0 + k[689]*y[IDX_CH5II];
    data[3788] = 0.0 + k[828]*y[IDX_CH5II] + k[1652]*y[IDX_CH3I];
    data[3789] = 0.0 + k[826]*y[IDX_CH5II];
    data[3790] = 0.0 + k[1724]*y[IDX_CH3I];
    data[3791] = 0.0 + k[496]*y[IDX_CH5II];
    data[3792] = 0.0 - k[1742]*y[IDX_CH4I];
    data[3793] = 0.0 + k[536]*y[IDX_EM];
    data[3794] = 0.0 + k[1691]*y[IDX_CHI] - k[1786]*y[IDX_HCOI];
    data[3795] = 0.0 - k[1782]*y[IDX_HCOI];
    data[3796] = 0.0 - k[206]*y[IDX_HCOI];
    data[3797] = 0.0 + k[1576]*y[IDX_O2I];
    data[3798] = 0.0 + k[613]*y[IDX_CII] + k[965]*y[IDX_H2COII];
    data[3799] = 0.0 + k[1561]*y[IDX_SOII] + k[1873]*y[IDX_OI];
    data[3800] = 0.0 + k[224]*y[IDX_HCOII];
    data[3801] = 0.0 + k[1558]*y[IDX_C2H2I] + k[1561]*y[IDX_C2H4I];
    data[3802] = 0.0 - k[33]*y[IDX_HCOI] - k[648]*y[IDX_HCOI];
    data[3803] = 0.0 - k[1162]*y[IDX_HCOI];
    data[3804] = 0.0 - k[96]*y[IDX_HCOI] - k[867]*y[IDX_HCOI];
    data[3805] = 0.0 - k[1160]*y[IDX_HCOI];
    data[3806] = 0.0 + k[752]*y[IDX_CH2II];
    data[3807] = 0.0 - k[203]*y[IDX_HCOI];
    data[3808] = 0.0 - k[1159]*y[IDX_HCOI];
    data[3809] = 0.0 - k[831]*y[IDX_HCOI];
    data[3810] = 0.0 - k[254]*y[IDX_HCOI] - k[1317]*y[IDX_HCOI];
    data[3811] = 0.0 - k[165]*y[IDX_HCOI] - k[925]*y[IDX_HCOI];
    data[3812] = 0.0 + k[525]*y[IDX_EM];
    data[3813] = 0.0 + k[1558]*y[IDX_SOII];
    data[3814] = 0.0 - k[1114]*y[IDX_HCOI] - k[1115]*y[IDX_HCOI];
    data[3815] = 0.0 + k[1165]*y[IDX_H2COII];
    data[3816] = 0.0 - k[103]*y[IDX_HCOI] + k[871]*y[IDX_H2COI];
    data[3817] = 0.0 + k[1352]*y[IDX_NHII] + k[1677]*y[IDX_CHI];
    data[3818] = 0.0 + k[1352]*y[IDX_CO2I] - k[1361]*y[IDX_HCOI];
    data[3819] = 0.0 + k[1399]*y[IDX_H2COII];
    data[3820] = 0.0 - k[204]*y[IDX_HCOI] - k[1161]*y[IDX_HCOI];
    data[3821] = 0.0 - k[243]*y[IDX_HCOI] - k[1303]*y[IDX_HCOI];
    data[3822] = 0.0 - k[267]*y[IDX_HCOI] + k[1377]*y[IDX_H2COI] - k[1386]*y[IDX_HCOI];
    data[3823] = 0.0 - k[663]*y[IDX_HCOI];
    data[3824] = 0.0 + k[678]*y[IDX_H2COII] + k[1581]*y[IDX_O2I];
    data[3825] = 0.0 - k[314]*y[IDX_HCOI] - k[1476]*y[IDX_HCOI];
    data[3826] = 0.0 - k[202]*y[IDX_HCOI] + k[505]*y[IDX_EM] + k[651]*y[IDX_C2I] +
        k[678]*y[IDX_C2HI] + k[757]*y[IDX_CH2I] + k[842]*y[IDX_CHI] +
        k[965]*y[IDX_CH3OHI] + k[966]*y[IDX_H2COI] + k[968]*y[IDX_SI] +
        k[998]*y[IDX_H2OI] + k[1121]*y[IDX_HCNI] - k[1158]*y[IDX_HCOI] +
        k[1165]*y[IDX_HNCI] + k[1399]*y[IDX_NH2I] + k[1424]*y[IDX_NH3I];
    data[3827] = 0.0 - k[747]*y[IDX_HCOI] + k[752]*y[IDX_OCSI];
    data[3828] = 0.0 - k[278]*y[IDX_HCOI] + k[1413]*y[IDX_H2COI] - k[1416]*y[IDX_HCOI];
    data[3829] = 0.0 - k[180]*y[IDX_HCOI] - k[984]*y[IDX_HCOI] - k[985]*y[IDX_HCOI];
    data[3830] = 0.0 - k[334]*y[IDX_HCOI] - k[1526]*y[IDX_HCOI] - k[1527]*y[IDX_HCOI];
    data[3831] = 0.0 - k[72]*y[IDX_HCOI] - k[779]*y[IDX_HCOI];
    data[3832] = 0.0 - k[44]*y[IDX_HCOI];
    data[3833] = 0.0 + k[651]*y[IDX_H2COII];
    data[3834] = 0.0 + k[757]*y[IDX_H2COII] + k[1624]*y[IDX_H2COI] - k[1625]*y[IDX_HCOI]
        + k[1636]*y[IDX_O2I] + k[1639]*y[IDX_OI];
    data[3835] = 0.0 + k[1651]*y[IDX_H2COI] - k[1654]*y[IDX_HCOI] + k[1661]*y[IDX_O2I];
    data[3836] = 0.0 - k[55]*y[IDX_HCOI] - k[726]*y[IDX_HCOI] + k[734]*y[IDX_O2I];
    data[3837] = 0.0 - k[205]*y[IDX_HCOI] - k[1163]*y[IDX_HCOI];
    data[3838] = 0.0 + k[1704]*y[IDX_H2COI] - k[1706]*y[IDX_HCOI];
    data[3839] = 0.0 + k[871]*y[IDX_COII] + k[966]*y[IDX_H2COII] + k[1377]*y[IDX_NH2II] +
        k[1413]*y[IDX_NH3II] + k[1624]*y[IDX_CH2I] + k[1651]*y[IDX_CH3I] +
        k[1678]*y[IDX_CHI] + k[1704]*y[IDX_CNI] + k[1747]*y[IDX_HI] +
        k[1886]*y[IDX_OI] + k[1937]*y[IDX_OHI];
    data[3840] = 0.0 - k[18]*y[IDX_CII] - k[33]*y[IDX_C2II] - k[44]*y[IDX_C2H2II] -
        k[55]*y[IDX_CHII] - k[72]*y[IDX_CH3II] - k[96]*y[IDX_CNII] -
        k[103]*y[IDX_COII] - k[125]*y[IDX_HII] - k[165]*y[IDX_H2II] -
        k[180]*y[IDX_H2OII] - k[202]*y[IDX_H2COII] - k[203]*y[IDX_H2SII] -
        k[204]*y[IDX_O2II] - k[205]*y[IDX_SII] - k[206]*y[IDX_SiOII] -
        k[243]*y[IDX_NII] - k[254]*y[IDX_N2II] - k[267]*y[IDX_NH2II] -
        k[278]*y[IDX_NH3II] - k[314]*y[IDX_OII] - k[334]*y[IDX_OHII] - k[409] -
        k[410] - k[626]*y[IDX_CII] - k[648]*y[IDX_C2II] - k[663]*y[IDX_C2HII] -
        k[726]*y[IDX_CHII] - k[747]*y[IDX_CH2II] - k[779]*y[IDX_CH3II] -
        k[831]*y[IDX_CH5II] - k[867]*y[IDX_CNII] - k[897]*y[IDX_HII] -
        k[898]*y[IDX_HII] - k[925]*y[IDX_H2II] - k[984]*y[IDX_H2OII] -
        k[985]*y[IDX_H2OII] - k[1046]*y[IDX_H3II] - k[1114]*y[IDX_HCNII] -
        k[1115]*y[IDX_HCNII] - k[1144]*y[IDX_HCOII] - k[1158]*y[IDX_H2COII] -
        k[1159]*y[IDX_HNOII] - k[1160]*y[IDX_N2HII] - k[1161]*y[IDX_O2II] -
        k[1162]*y[IDX_O2HII] - k[1163]*y[IDX_SII] - k[1235]*y[IDX_HeII] -
        k[1236]*y[IDX_HeII] - k[1237]*y[IDX_HeII] - k[1303]*y[IDX_NII] -
        k[1317]*y[IDX_N2II] - k[1361]*y[IDX_NHII] - k[1386]*y[IDX_NH2II] -
        k[1416]*y[IDX_NH3II] - k[1476]*y[IDX_OII] - k[1526]*y[IDX_OHII] -
        k[1527]*y[IDX_OHII] - k[1594]*y[IDX_CI] - k[1625]*y[IDX_CH2I] -
        k[1654]*y[IDX_CH3I] - k[1679]*y[IDX_CHI] - k[1706]*y[IDX_CNI] -
        k[1751]*y[IDX_HI] - k[1752]*y[IDX_HI] - k[1780]*y[IDX_HCOI] -
        k[1780]*y[IDX_HCOI] - k[1780]*y[IDX_HCOI] - k[1780]*y[IDX_HCOI] -
        k[1781]*y[IDX_HCOI] - k[1781]*y[IDX_HCOI] - k[1781]*y[IDX_HCOI] -
        k[1781]*y[IDX_HCOI] - k[1782]*y[IDX_HNOI] - k[1783]*y[IDX_NOI] -
        k[1784]*y[IDX_O2I] - k[1785]*y[IDX_O2I] - k[1786]*y[IDX_O2HI] -
        k[1811]*y[IDX_NI] - k[1812]*y[IDX_NI] - k[1813]*y[IDX_NI] -
        k[1892]*y[IDX_OI] - k[1893]*y[IDX_OI] - k[1941]*y[IDX_OHI] -
        k[1951]*y[IDX_SI] - k[1952]*y[IDX_SI] - k[2030] - k[2031] - k[2218];
    data[3841] = 0.0 + k[1121]*y[IDX_H2COII];
    data[3842] = 0.0 + k[1685]*y[IDX_CHI] - k[1783]*y[IDX_HCOI];
    data[3843] = 0.0 + k[842]*y[IDX_H2COII] + k[1677]*y[IDX_CO2I] + k[1678]*y[IDX_H2COI]
        - k[1679]*y[IDX_HCOI] + k[1685]*y[IDX_NOI] + k[1690]*y[IDX_O2I] +
        k[1691]*y[IDX_O2HI] + k[1696]*y[IDX_OHI];
    data[3844] = 0.0 + k[1424]*y[IDX_H2COII];
    data[3845] = 0.0 + k[734]*y[IDX_CHII] + k[1576]*y[IDX_C2H3I] + k[1581]*y[IDX_C2HI] +
        k[1636]*y[IDX_CH2I] + k[1661]*y[IDX_CH3I] + k[1690]*y[IDX_CHI] -
        k[1784]*y[IDX_HCOI] - k[1785]*y[IDX_HCOI];
    data[3846] = 0.0 - k[18]*y[IDX_HCOI] + k[613]*y[IDX_CH3OHI] - k[626]*y[IDX_HCOI];
    data[3847] = 0.0 + k[968]*y[IDX_H2COII] - k[1951]*y[IDX_HCOI] - k[1952]*y[IDX_HCOI];
    data[3848] = 0.0 + k[1696]*y[IDX_CHI] + k[1937]*y[IDX_H2COI] - k[1941]*y[IDX_HCOI];
    data[3849] = 0.0 - k[1811]*y[IDX_HCOI] - k[1812]*y[IDX_HCOI] - k[1813]*y[IDX_HCOI];
    data[3850] = 0.0 - k[1235]*y[IDX_HCOI] - k[1236]*y[IDX_HCOI] - k[1237]*y[IDX_HCOI];
    data[3851] = 0.0 - k[1046]*y[IDX_HCOI];
    data[3852] = 0.0 - k[125]*y[IDX_HCOI] - k[897]*y[IDX_HCOI] - k[898]*y[IDX_HCOI];
    data[3853] = 0.0 + k[1639]*y[IDX_CH2I] + k[1873]*y[IDX_C2H4I] + k[1886]*y[IDX_H2COI]
        - k[1892]*y[IDX_HCOI] - k[1893]*y[IDX_HCOI];
    data[3854] = 0.0 - k[1594]*y[IDX_HCOI];
    data[3855] = 0.0 + k[224]*y[IDX_MgI] - k[1144]*y[IDX_HCOI];
    data[3856] = 0.0 + k[998]*y[IDX_H2COII];
    data[3857] = 0.0 + k[505]*y[IDX_H2COII] + k[525]*y[IDX_H3COII] +
        k[536]*y[IDX_H5C2O2II];
    data[3858] = 0.0 + k[1747]*y[IDX_H2COI] - k[1751]*y[IDX_HCOI] - k[1752]*y[IDX_HCOI];
    data[3859] = 0.0 + k[2331] + k[2332] + k[2333] + k[2334];
    data[3860] = 0.0 + k[398] + k[1746]*y[IDX_HI] + k[1810]*y[IDX_NI] + k[2010];
    data[3861] = 0.0 - k[1118]*y[IDX_HCNI];
    data[3862] = 0.0 - k[1119]*y[IDX_HCNI];
    data[3863] = 0.0 + k[624]*y[IDX_CII];
    data[3864] = 0.0 - k[1127]*y[IDX_HCNI];
    data[3865] = 0.0 + k[1759]*y[IDX_HI] + k[1943]*y[IDX_OHI];
    data[3866] = 0.0 + k[1814]*y[IDX_NI];
    data[3867] = 0.0 - k[1120]*y[IDX_HCNI];
    data[3868] = 0.0 + k[1772]*y[IDX_HI];
    data[3869] = 0.0 + k[886]*y[IDX_HII] + k[1130]*y[IDX_HCNHII];
    data[3870] = 0.0 + k[1708]*y[IDX_CNI];
    data[3871] = 0.0 + k[1715]*y[IDX_CNI];
    data[3872] = 0.0 + k[993]*y[IDX_H2OI] + k[1018]*y[IDX_H2SI] + k[1421]*y[IDX_NH3I];
    data[3873] = 0.0 - k[1123]*y[IDX_HCNI];
    data[3874] = 0.0 + k[1702]*y[IDX_CNI] + k[1792]*y[IDX_NI];
    data[3875] = 0.0 - k[1129]*y[IDX_HCNI];
    data[3876] = 0.0 - k[95]*y[IDX_HCNI] + k[865]*y[IDX_H2COI] - k[866]*y[IDX_HCNI];
    data[3877] = 0.0 - k[1128]*y[IDX_HCNI];
    data[3878] = 0.0 - k[1125]*y[IDX_HCNI];
    data[3879] = 0.0 - k[830]*y[IDX_HCNI];
    data[3880] = 0.0 - k[201]*y[IDX_HCNI];
    data[3881] = 0.0 + k[540]*y[IDX_EM] + k[761]*y[IDX_CH2I] + k[847]*y[IDX_CHI] +
        k[1130]*y[IDX_CH3CNI] + k[1132]*y[IDX_H2COI] + k[1134]*y[IDX_H2SI] +
        k[1404]*y[IDX_NH2I] + k[1431]*y[IDX_NH3I];
    data[3882] = 0.0 - k[164]*y[IDX_HCNI];
    data[3883] = 0.0 - k[1122]*y[IDX_HCNI];
    data[3884] = 0.0 + k[46]*y[IDX_HCNII] + k[1574]*y[IDX_NOI];
    data[3885] = 0.0 + k[46]*y[IDX_C2H2I] + k[188]*y[IDX_H2OI] + k[194]*y[IDX_HI] +
        k[197]*y[IDX_NOI] + k[198]*y[IDX_O2I] + k[199]*y[IDX_SI] +
        k[287]*y[IDX_NH3I] - k[1113]*y[IDX_HCNI];
    data[3886] = 0.0 + k[1]*y[IDX_HII] + k[1754]*y[IDX_HI];
    data[3887] = 0.0 - k[200]*y[IDX_HCNI];
    data[3888] = 0.0 - k[1126]*y[IDX_HCNI];
    data[3889] = 0.0 - k[1360]*y[IDX_HCNI];
    data[3890] = 0.0 + k[1404]*y[IDX_HCNHII] + k[1599]*y[IDX_CI];
    data[3891] = 0.0 - k[242]*y[IDX_HCNI];
    data[3892] = 0.0 - k[1385]*y[IDX_HCNI];
    data[3893] = 0.0 - k[661]*y[IDX_HCNI] - k[662]*y[IDX_HCNI];
    data[3894] = 0.0 - k[1578]*y[IDX_HCNI];
    data[3895] = 0.0 - k[1474]*y[IDX_HCNI] - k[1475]*y[IDX_HCNI];
    data[3896] = 0.0 - k[1121]*y[IDX_HCNI];
    data[3897] = 0.0 + k[1018]*y[IDX_C2NII] + k[1134]*y[IDX_HCNHII];
    data[3898] = 0.0 - k[983]*y[IDX_HCNI];
    data[3899] = 0.0 - k[1525]*y[IDX_HCNI];
    data[3900] = 0.0 - k[2108]*y[IDX_HCNI];
    data[3901] = 0.0 + k[1627]*y[IDX_CH2I] + k[1681]*y[IDX_CHI];
    data[3902] = 0.0 - k[668]*y[IDX_HCNI] + k[1330]*y[IDX_NI];
    data[3903] = 0.0 - k[1571]*y[IDX_HCNI];
    data[3904] = 0.0 + k[761]*y[IDX_HCNHII] + k[1623]*y[IDX_CNI] + k[1627]*y[IDX_N2I] +
        k[1630]*y[IDX_NOI] + k[1801]*y[IDX_NI];
    data[3905] = 0.0 + k[1650]*y[IDX_CNI] + k[1659]*y[IDX_NOI] + k[1805]*y[IDX_NI] +
        k[1806]*y[IDX_NI];
    data[3906] = 0.0 - k[723]*y[IDX_HCNI] - k[724]*y[IDX_HCNI] - k[725]*y[IDX_HCNI];
    data[3907] = 0.0 + k[1840]*y[IDX_CNI];
    data[3908] = 0.0 + k[1623]*y[IDX_CH2I] + k[1650]*y[IDX_CH3I] + k[1670]*y[IDX_CH4I] +
        k[1702]*y[IDX_C2H4I] + k[1704]*y[IDX_H2COI] - k[1705]*y[IDX_HCNI] +
        k[1706]*y[IDX_HCOI] + k[1708]*y[IDX_HNOI] + k[1715]*y[IDX_SiH4I] +
        k[1726]*y[IDX_H2I] + k[1838]*y[IDX_NH3I] + k[1840]*y[IDX_NHI] +
        k[1932]*y[IDX_OHI];
    data[3909] = 0.0 + k[865]*y[IDX_CNII] + k[1132]*y[IDX_HCNHII] + k[1704]*y[IDX_CNI];
    data[3910] = 0.0 + k[1670]*y[IDX_CNI];
    data[3911] = 0.0 + k[1706]*y[IDX_CNI] + k[1812]*y[IDX_NI];
    data[3912] = 0.0 - k[95]*y[IDX_CNII] - k[124]*y[IDX_HII] - k[164]*y[IDX_H2II] -
        k[200]*y[IDX_COII] - k[201]*y[IDX_N2II] - k[242]*y[IDX_NII] - k[408] -
        k[661]*y[IDX_C2HII] - k[662]*y[IDX_C2HII] - k[668]*y[IDX_C2H2II] -
        k[723]*y[IDX_CHII] - k[724]*y[IDX_CHII] - k[725]*y[IDX_CHII] -
        k[830]*y[IDX_CH5II] - k[866]*y[IDX_CNII] - k[983]*y[IDX_H2OII] -
        k[1045]*y[IDX_H3II] - k[1088]*y[IDX_H3OII] - k[1113]*y[IDX_HCNII] -
        k[1118]*y[IDX_C2N2II] - k[1119]*y[IDX_C3II] - k[1120]*y[IDX_CH3OH2II] -
        k[1121]*y[IDX_H2COII] - k[1122]*y[IDX_H3COII] - k[1123]*y[IDX_H3SII] -
        k[1124]*y[IDX_HCOII] - k[1125]*y[IDX_HNOII] - k[1126]*y[IDX_HSII] -
        k[1127]*y[IDX_HSiSII] - k[1128]*y[IDX_N2HII] - k[1129]*y[IDX_O2HII] -
        k[1231]*y[IDX_HeII] - k[1232]*y[IDX_HeII] - k[1233]*y[IDX_HeII] -
        k[1234]*y[IDX_HeII] - k[1360]*y[IDX_NHII] - k[1385]*y[IDX_NH2II] -
        k[1474]*y[IDX_OII] - k[1475]*y[IDX_OII] - k[1525]*y[IDX_OHII] -
        k[1571]*y[IDX_C2I] - k[1578]*y[IDX_C2HI] - k[1705]*y[IDX_CNI] -
        k[1750]*y[IDX_HI] - k[1889]*y[IDX_OI] - k[1890]*y[IDX_OI] -
        k[1891]*y[IDX_OI] - k[1939]*y[IDX_OHI] - k[1940]*y[IDX_OHI] - k[2028] -
        k[2108]*y[IDX_CH3II] - k[2223];
    data[3913] = 0.0 + k[197]*y[IDX_HCNII] + k[1574]*y[IDX_C2H2I] + k[1630]*y[IDX_CH2I] +
        k[1659]*y[IDX_CH3I] + k[1684]*y[IDX_CHI];
    data[3914] = 0.0 + k[847]*y[IDX_HCNHII] + k[1681]*y[IDX_N2I] + k[1684]*y[IDX_NOI];
    data[3915] = 0.0 + k[287]*y[IDX_HCNII] + k[1421]*y[IDX_C2NII] + k[1431]*y[IDX_HCNHII]
        + k[1838]*y[IDX_CNI];
    data[3916] = 0.0 - k[1088]*y[IDX_HCNI];
    data[3917] = 0.0 + k[198]*y[IDX_HCNII];
    data[3918] = 0.0 + k[624]*y[IDX_HC3NI];
    data[3919] = 0.0 + k[199]*y[IDX_HCNII];
    data[3920] = 0.0 + k[1932]*y[IDX_CNI] - k[1939]*y[IDX_HCNI] - k[1940]*y[IDX_HCNI] +
        k[1943]*y[IDX_NCCNI];
    data[3921] = 0.0 + k[1330]*y[IDX_C2H2II] + k[1792]*y[IDX_C2H4I] + k[1801]*y[IDX_CH2I]
        + k[1805]*y[IDX_CH3I] + k[1806]*y[IDX_CH3I] + k[1810]*y[IDX_H2CNI] +
        k[1812]*y[IDX_HCOI] + k[1814]*y[IDX_HCSI];
    data[3922] = 0.0 - k[1231]*y[IDX_HCNI] - k[1232]*y[IDX_HCNI] - k[1233]*y[IDX_HCNI] -
        k[1234]*y[IDX_HCNI];
    data[3923] = 0.0 - k[1045]*y[IDX_HCNI];
    data[3924] = 0.0 + k[1]*y[IDX_HNCI] - k[124]*y[IDX_HCNI] + k[886]*y[IDX_CH3CNI];
    data[3925] = 0.0 - k[1889]*y[IDX_HCNI] - k[1890]*y[IDX_HCNI] - k[1891]*y[IDX_HCNI];
    data[3926] = 0.0 + k[1599]*y[IDX_NH2I];
    data[3927] = 0.0 - k[1124]*y[IDX_HCNI];
    data[3928] = 0.0 + k[188]*y[IDX_HCNII] + k[993]*y[IDX_C2NII];
    data[3929] = 0.0 + k[1726]*y[IDX_CNI];
    data[3930] = 0.0 + k[540]*y[IDX_HCNHII];
    data[3931] = 0.0 + k[194]*y[IDX_HCNII] + k[1746]*y[IDX_H2CNI] - k[1750]*y[IDX_HCNI] +
        k[1754]*y[IDX_HNCI] + k[1759]*y[IDX_NCCNI] + k[1772]*y[IDX_OCNI];
    data[3932] = 0.0 + k[2363] + k[2364] + k[2365] + k[2366];
    data[3933] = 0.0 + k[511]*y[IDX_EM];
    data[3934] = 0.0 + k[431] + k[1628]*y[IDX_CH2I] + k[1709]*y[IDX_CNI] +
        k[1717]*y[IDX_COI] + k[1763]*y[IDX_HI] + k[1821]*y[IDX_NI] +
        k[1821]*y[IDX_NI] + k[1846]*y[IDX_NHI] + k[1905]*y[IDX_OI] + k[2056];
    data[3935] = 0.0 - k[304]*y[IDX_NOI];
    data[3936] = 0.0 - k[1860]*y[IDX_NOI] + k[1863]*y[IDX_O2I] + k[1910]*y[IDX_OI];
    data[3937] = 0.0 + k[1907]*y[IDX_OI];
    data[3938] = 0.0 + k[416] + k[1246]*y[IDX_HeII] + k[1626]*y[IDX_CH2I] +
        k[1655]*y[IDX_CH3I] + k[1680]*y[IDX_CHI] + k[1708]*y[IDX_CNI] +
        k[1756]*y[IDX_HI] + k[1782]*y[IDX_HCOI] + k[1815]*y[IDX_NI] +
        k[1897]*y[IDX_OI] + k[1942]*y[IDX_OHI] + k[2038];
    data[3939] = 0.0 - k[305]*y[IDX_NOI] + k[1344]*y[IDX_NI];
    data[3940] = 0.0 + k[1292]*y[IDX_NII];
    data[3941] = 0.0 + k[1831]*y[IDX_NI];
    data[3942] = 0.0 + k[227]*y[IDX_NOII];
    data[3943] = 0.0 - k[34]*y[IDX_NOI];
    data[3944] = 0.0 - k[1462]*y[IDX_NOI];
    data[3945] = 0.0 - k[97]*y[IDX_NOI];
    data[3946] = 0.0 + k[1312]*y[IDX_NII];
    data[3947] = 0.0 - k[299]*y[IDX_NOI];
    data[3948] = 0.0 - k[300]*y[IDX_NOI] + k[549]*y[IDX_EM] + k[654]*y[IDX_C2I] +
        k[681]*y[IDX_C2HI] + k[696]*y[IDX_CI] + k[764]*y[IDX_CH2I] +
        k[813]*y[IDX_CH4I] + k[850]*y[IDX_CHI] + k[869]*y[IDX_CNI] +
        k[876]*y[IDX_COI] + k[971]*y[IDX_H2COI] + k[1005]*y[IDX_H2OI] +
        k[1125]*y[IDX_HCNI] + k[1159]*y[IDX_HCOI] + k[1169]*y[IDX_HNCI] +
        k[1173]*y[IDX_CO2I] + k[1174]*y[IDX_SI] + k[1319]*y[IDX_N2I] +
        k[1407]*y[IDX_NH2I] + k[1436]*y[IDX_NH3I] + k[1453]*y[IDX_NHI] +
        k[1544]*y[IDX_OHI];
    data[3949] = 0.0 - k[255]*y[IDX_NOI];
    data[3950] = 0.0 - k[169]*y[IDX_NOI] - k[930]*y[IDX_NOI];
    data[3951] = 0.0 + k[352]*y[IDX_NOII] - k[1958]*y[IDX_NOI];
    data[3952] = 0.0 - k[1574]*y[IDX_NOI];
    data[3953] = 0.0 - k[197]*y[IDX_NOI];
    data[3954] = 0.0 + k[1169]*y[IDX_HNOII];
    data[3955] = 0.0 - k[104]*y[IDX_NOI];
    data[3956] = 0.0 - k[301]*y[IDX_NOI];
    data[3957] = 0.0 + k[1173]*y[IDX_HNOII] + k[1296]*y[IDX_NII] + k[1808]*y[IDX_NI];
    data[3958] = 0.0 - k[263]*y[IDX_NOI] - k[1367]*y[IDX_NOI];
    data[3959] = 0.0 + k[1407]*y[IDX_HNOII] - k[1834]*y[IDX_NOI] - k[1835]*y[IDX_NOI];
    data[3960] = 0.0 - k[302]*y[IDX_NOI];
    data[3961] = 0.0 - k[248]*y[IDX_NOI] + k[1292]*y[IDX_CH3OHI] + k[1296]*y[IDX_CO2I] -
        k[1309]*y[IDX_NOI] + k[1311]*y[IDX_O2I] + k[1312]*y[IDX_OCSI];
    data[3962] = 0.0 - k[269]*y[IDX_NOI];
    data[3963] = 0.0 - k[40]*y[IDX_NOI];
    data[3964] = 0.0 + k[681]*y[IDX_HNOII];
    data[3965] = 0.0 + k[227]*y[IDX_MgI] + k[352]*y[IDX_SiI];
    data[3966] = 0.0 - k[298]*y[IDX_NOI];
    data[3967] = 0.0 - k[61]*y[IDX_NOI];
    data[3968] = 0.0 - k[280]*y[IDX_NOI];
    data[3969] = 0.0 - k[182]*y[IDX_NOI];
    data[3970] = 0.0 - k[336]*y[IDX_NOI] - k[1531]*y[IDX_NOI];
    data[3971] = 0.0 - k[74]*y[IDX_NOI];
    data[3972] = 0.0 + k[1319]*y[IDX_HNOII] + k[1901]*y[IDX_OI];
    data[3973] = 0.0 - k[45]*y[IDX_NOI];
    data[3974] = 0.0 + k[654]*y[IDX_HNOII];
    data[3975] = 0.0 + k[764]*y[IDX_HNOII] + k[1626]*y[IDX_HNOI] + k[1628]*y[IDX_NO2I] -
        k[1629]*y[IDX_NOI] - k[1630]*y[IDX_NOI] - k[1631]*y[IDX_NOI];
    data[3976] = 0.0 + k[1655]*y[IDX_HNOI] - k[1659]*y[IDX_NOI];
    data[3977] = 0.0 - k[58]*y[IDX_NOI];
    data[3978] = 0.0 + k[1453]*y[IDX_HNOII] + k[1846]*y[IDX_NO2I] - k[1847]*y[IDX_NOI] -
        k[1848]*y[IDX_NOI] + k[1850]*y[IDX_O2I] + k[1851]*y[IDX_OI];
    data[3979] = 0.0 - k[303]*y[IDX_NOI];
    data[3980] = 0.0 + k[869]*y[IDX_HNOII] + k[1708]*y[IDX_HNOI] + k[1709]*y[IDX_NO2I] -
        k[1710]*y[IDX_NOI] - k[1711]*y[IDX_NOI] + k[1712]*y[IDX_O2I] +
        k[1881]*y[IDX_OI];
    data[3981] = 0.0 + k[971]*y[IDX_HNOII];
    data[3982] = 0.0 + k[813]*y[IDX_HNOII];
    data[3983] = 0.0 + k[1159]*y[IDX_HNOII] + k[1782]*y[IDX_HNOI] - k[1783]*y[IDX_NOI];
    data[3984] = 0.0 + k[1125]*y[IDX_HNOII];
    data[3985] = 0.0 - k[22]*y[IDX_CII] - k[34]*y[IDX_C2II] - k[40]*y[IDX_C2HII] -
        k[45]*y[IDX_C2H2II] - k[58]*y[IDX_CHII] - k[61]*y[IDX_CH2II] -
        k[74]*y[IDX_CH3II] - k[97]*y[IDX_CNII] - k[104]*y[IDX_COII] -
        k[133]*y[IDX_HII] - k[169]*y[IDX_H2II] - k[182]*y[IDX_H2OII] -
        k[197]*y[IDX_HCNII] - k[248]*y[IDX_NII] - k[255]*y[IDX_N2II] -
        k[263]*y[IDX_NHII] - k[269]*y[IDX_NH2II] - k[280]*y[IDX_NH3II] -
        k[298]*y[IDX_H2COII] - k[299]*y[IDX_H2SII] - k[300]*y[IDX_HNOII] -
        k[301]*y[IDX_HSII] - k[302]*y[IDX_O2II] - k[303]*y[IDX_SII] -
        k[304]*y[IDX_S2II] - k[305]*y[IDX_SiOII] - k[336]*y[IDX_OHII] - k[432] -
        k[433] - k[930]*y[IDX_H2II] - k[1060]*y[IDX_H3II] - k[1257]*y[IDX_HeII]
        - k[1258]*y[IDX_HeII] - k[1309]*y[IDX_NII] - k[1367]*y[IDX_NHII] -
        k[1462]*y[IDX_O2HII] - k[1531]*y[IDX_OHII] - k[1574]*y[IDX_C2H2I] -
        k[1604]*y[IDX_CI] - k[1605]*y[IDX_CI] - k[1629]*y[IDX_CH2I] -
        k[1630]*y[IDX_CH2I] - k[1631]*y[IDX_CH2I] - k[1659]*y[IDX_CH3I] -
        k[1684]*y[IDX_CHI] - k[1685]*y[IDX_CHI] - k[1686]*y[IDX_CHI] -
        k[1710]*y[IDX_CNI] - k[1711]*y[IDX_CNI] - k[1764]*y[IDX_HI] -
        k[1765]*y[IDX_HI] - k[1783]*y[IDX_HCOI] - k[1823]*y[IDX_NI] -
        k[1834]*y[IDX_NH2I] - k[1835]*y[IDX_NH2I] - k[1847]*y[IDX_NHI] -
        k[1848]*y[IDX_NHI] - k[1858]*y[IDX_NOI] - k[1858]*y[IDX_NOI] -
        k[1858]*y[IDX_NOI] - k[1858]*y[IDX_NOI] - k[1859]*y[IDX_O2I] -
        k[1860]*y[IDX_OCNI] - k[1861]*y[IDX_SI] - k[1862]*y[IDX_SI] -
        k[1906]*y[IDX_OI] - k[1945]*y[IDX_OHI] - k[1958]*y[IDX_SiI] - k[2057] -
        k[2058] - k[2212];
    data[3986] = 0.0 + k[850]*y[IDX_HNOII] + k[1680]*y[IDX_HNOI] - k[1684]*y[IDX_NOI] -
        k[1685]*y[IDX_NOI] - k[1686]*y[IDX_NOI];
    data[3987] = 0.0 + k[1436]*y[IDX_HNOII];
    data[3988] = 0.0 + k[1311]*y[IDX_NII] + k[1712]*y[IDX_CNI] + k[1825]*y[IDX_NI] +
        k[1850]*y[IDX_NHI] - k[1859]*y[IDX_NOI] + k[1863]*y[IDX_OCNI];
    data[3989] = 0.0 - k[22]*y[IDX_NOI];
    data[3990] = 0.0 + k[1174]*y[IDX_HNOII] - k[1861]*y[IDX_NOI] - k[1862]*y[IDX_NOI];
    data[3991] = 0.0 + k[1544]*y[IDX_HNOII] + k[1827]*y[IDX_NI] + k[1942]*y[IDX_HNOI] -
        k[1945]*y[IDX_NOI];
    data[3992] = 0.0 + k[1344]*y[IDX_SiOII] + k[1808]*y[IDX_CO2I] + k[1815]*y[IDX_HNOI] +
        k[1821]*y[IDX_NO2I] + k[1821]*y[IDX_NO2I] - k[1823]*y[IDX_NOI] +
        k[1825]*y[IDX_O2I] + k[1827]*y[IDX_OHI] + k[1831]*y[IDX_SOI];
    data[3993] = 0.0 + k[1246]*y[IDX_HNOI] - k[1257]*y[IDX_NOI] - k[1258]*y[IDX_NOI];
    data[3994] = 0.0 - k[1060]*y[IDX_NOI];
    data[3995] = 0.0 - k[133]*y[IDX_NOI];
    data[3996] = 0.0 + k[1851]*y[IDX_NHI] + k[1881]*y[IDX_CNI] + k[1897]*y[IDX_HNOI] +
        k[1901]*y[IDX_N2I] + k[1905]*y[IDX_NO2I] - k[1906]*y[IDX_NOI] +
        k[1907]*y[IDX_NSI] + k[1910]*y[IDX_OCNI];
    data[3997] = 0.0 + k[696]*y[IDX_HNOII] - k[1604]*y[IDX_NOI] - k[1605]*y[IDX_NOI];
    data[3998] = 0.0 + k[1005]*y[IDX_HNOII];
    data[3999] = 0.0 + k[876]*y[IDX_HNOII] + k[1717]*y[IDX_NO2I];
    data[4000] = 0.0 + k[511]*y[IDX_H2NOII] + k[549]*y[IDX_HNOII];
    data[4001] = 0.0 + k[1756]*y[IDX_HNOI] + k[1763]*y[IDX_NO2I] - k[1764]*y[IDX_NOI] -
        k[1765]*y[IDX_NOI];
    data[4002] = 0.0 + k[1229]*y[IDX_HeII];
    data[4003] = 0.0 - k[1691]*y[IDX_CHI] - k[1692]*y[IDX_CHI];
    data[4004] = 0.0 - k[1680]*y[IDX_CHI];
    data[4005] = 0.0 - k[864]*y[IDX_CHI];
    data[4006] = 0.0 - k[863]*y[IDX_CHI];
    data[4007] = 0.0 + k[546]*y[IDX_EM];
    data[4008] = 0.0 + k[612]*y[IDX_CII];
    data[4009] = 0.0 - k[1699]*y[IDX_CHI] - k[1700]*y[IDX_CHI];
    data[4010] = 0.0 - k[1675]*y[IDX_CHI];
    data[4011] = 0.0 + k[56]*y[IDX_CHII];
    data[4012] = 0.0 - k[82]*y[IDX_CHI] - k[837]*y[IDX_CHI];
    data[4013] = 0.0 - k[859]*y[IDX_CHI];
    data[4014] = 0.0 - k[83]*y[IDX_CHI];
    data[4015] = 0.0 - k[853]*y[IDX_CHI];
    data[4016] = 0.0 - k[1695]*y[IDX_CHI];
    data[4017] = 0.0 - k[850]*y[IDX_CHI];
    data[4018] = 0.0 + k[497]*y[IDX_EM] - k[840]*y[IDX_CHI];
    data[4019] = 0.0 - k[88]*y[IDX_CHI];
    data[4020] = 0.0 + k[1596]*y[IDX_CI];
    data[4021] = 0.0 - k[847]*y[IDX_CHI] - k[848]*y[IDX_CHI];
    data[4022] = 0.0 - k[158]*y[IDX_CHI] - k[916]*y[IDX_CHI];
    data[4023] = 0.0 + k[522]*y[IDX_EM] - k[844]*y[IDX_CHI];
    data[4024] = 0.0 + k[60]*y[IDX_CHII];
    data[4025] = 0.0 + k[1180]*y[IDX_HeII] - k[1674]*y[IDX_CHI];
    data[4026] = 0.0 - k[846]*y[IDX_CHI];
    data[4027] = 0.0 - k[862]*y[IDX_CHI];
    data[4028] = 0.0 - k[84]*y[IDX_CHI] + k[756]*y[IDX_CH2I] - k[841]*y[IDX_CHI];
    data[4029] = 0.0 - k[851]*y[IDX_CHI];
    data[4030] = 0.0 - k[1677]*y[IDX_CHI];
    data[4031] = 0.0 - k[854]*y[IDX_CHI];
    data[4032] = 0.0 + k[1601]*y[IDX_CI];
    data[4033] = 0.0 - k[91]*y[IDX_CHI] - k[858]*y[IDX_CHI];
    data[4034] = 0.0 - k[87]*y[IDX_CHI] - k[852]*y[IDX_CHI];
    data[4035] = 0.0 - k[89]*y[IDX_CHI] - k[855]*y[IDX_CHI];
    data[4036] = 0.0 + k[460]*y[IDX_EM] - k[838]*y[IDX_CHI];
    data[4037] = 0.0 + k[1188]*y[IDX_HeII] + k[1465]*y[IDX_OII] + k[1875]*y[IDX_OI];
    data[4038] = 0.0 - k[90]*y[IDX_CHI] - k[857]*y[IDX_CHI] + k[1465]*y[IDX_C2HI] +
        k[1475]*y[IDX_HCNI];
    data[4039] = 0.0 - k[85]*y[IDX_CHI] - k[842]*y[IDX_CHI];
    data[4040] = 0.0 + k[480]*y[IDX_EM] + k[745]*y[IDX_H2SI] + k[748]*y[IDX_NH3I] +
        k[1980];
    data[4041] = 0.0 - k[856]*y[IDX_CHI];
    data[4042] = 0.0 + k[745]*y[IDX_CH2II];
    data[4043] = 0.0 - k[86]*y[IDX_CHI] - k[843]*y[IDX_CHI];
    data[4044] = 0.0 - k[92]*y[IDX_CHI] - k[860]*y[IDX_CHI];
    data[4045] = 0.0 + k[482]*y[IDX_EM] + k[483]*y[IDX_EM] - k[839]*y[IDX_CHI];
    data[4046] = 0.0 - k[1681]*y[IDX_CHI];
    data[4047] = 0.0 + k[463]*y[IDX_EM] + k[463]*y[IDX_EM] + k[1492]*y[IDX_OI];
    data[4048] = 0.0 + k[1736]*y[IDX_HI];
    data[4049] = 0.0 + k[382] + k[756]*y[IDX_COII] + k[1587]*y[IDX_CI] +
        k[1587]*y[IDX_CI] + k[1621]*y[IDX_CH2I] + k[1621]*y[IDX_CH2I] +
        k[1623]*y[IDX_CNI] + k[1640]*y[IDX_OI] + k[1642]*y[IDX_OHI] +
        k[1739]*y[IDX_HI] + k[1803]*y[IDX_NI] + k[1982];
    data[4050] = 0.0 + k[386] + k[1988];
    data[4051] = 0.0 + k[55]*y[IDX_HCOI] + k[56]*y[IDX_MgI] + k[57]*y[IDX_NH3I] +
        k[58]*y[IDX_NOI] + k[59]*y[IDX_SI] + k[60]*y[IDX_SiI] -
        k[712]*y[IDX_CHI];
    data[4052] = 0.0 + k[1603]*y[IDX_CI];
    data[4053] = 0.0 - k[861]*y[IDX_CHI];
    data[4054] = 0.0 + k[1623]*y[IDX_CH2I];
    data[4055] = 0.0 + k[618]*y[IDX_CII] - k[1678]*y[IDX_CHI];
    data[4056] = 0.0 - k[1676]*y[IDX_CHI] + k[1998];
    data[4057] = 0.0 + k[55]*y[IDX_CHII] + k[1594]*y[IDX_CI] - k[1679]*y[IDX_CHI];
    data[4058] = 0.0 + k[1232]*y[IDX_HeII] + k[1475]*y[IDX_OII];
    data[4059] = 0.0 + k[58]*y[IDX_CHII] - k[1684]*y[IDX_CHI] - k[1685]*y[IDX_CHI] -
        k[1686]*y[IDX_CHI];
    data[4060] = 0.0 - k[0]*y[IDX_OI] - k[2]*y[IDX_H2I] - k[9]*y[IDX_HI] -
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
    data[4061] = 0.0 + k[57]*y[IDX_CHII] + k[748]*y[IDX_CH2II];
    data[4062] = 0.0 - k[845]*y[IDX_CHI];
    data[4063] = 0.0 - k[1687]*y[IDX_CHI] - k[1688]*y[IDX_CHI] - k[1689]*y[IDX_CHI] -
        k[1690]*y[IDX_CHI];
    data[4064] = 0.0 - k[15]*y[IDX_CHI] + k[612]*y[IDX_CH3OHI] - k[615]*y[IDX_CHI] +
        k[618]*y[IDX_H2COI];
    data[4065] = 0.0 + k[59]*y[IDX_CHII] - k[1697]*y[IDX_CHI] - k[1698]*y[IDX_CHI];
    data[4066] = 0.0 + k[1612]*y[IDX_CI] + k[1642]*y[IDX_CH2I] - k[1696]*y[IDX_CHI];
    data[4067] = 0.0 - k[1682]*y[IDX_CHI] - k[1683]*y[IDX_CHI] + k[1803]*y[IDX_CH2I];
    data[4068] = 0.0 - k[211]*y[IDX_CHI] + k[1180]*y[IDX_C2H2I] + k[1188]*y[IDX_C2HI] -
        k[1206]*y[IDX_CHI] + k[1229]*y[IDX_HC3NI] + k[1232]*y[IDX_HCNI];
    data[4069] = 0.0 - k[1034]*y[IDX_CHI];
    data[4070] = 0.0 - k[117]*y[IDX_CHI];
    data[4071] = 0.0 - k[0]*y[IDX_CHI] + k[1492]*y[IDX_C2H2II] + k[1640]*y[IDX_CH2I] -
        k[1693]*y[IDX_CHI] - k[1694]*y[IDX_CHI] + k[1875]*y[IDX_C2HI];
    data[4072] = 0.0 + k[1587]*y[IDX_CH2I] + k[1587]*y[IDX_CH2I] - k[1589]*y[IDX_CHI] +
        k[1594]*y[IDX_HCOI] + k[1596]*y[IDX_HSI] + k[1601]*y[IDX_NH2I] +
        k[1603]*y[IDX_NHI] + k[1612]*y[IDX_OHI] + k[1722]*y[IDX_H2I] +
        k[2123]*y[IDX_HI];
    data[4073] = 0.0 - k[849]*y[IDX_CHI];
    data[4074] = 0.0 - k[2]*y[IDX_CHI] + k[1722]*y[IDX_CI] - k[1725]*y[IDX_CHI] -
        k[2115]*y[IDX_CHI];
    data[4075] = 0.0 + k[460]*y[IDX_C2HII] + k[463]*y[IDX_C2H2II] + k[463]*y[IDX_C2H2II]
        + k[480]*y[IDX_CH2II] + k[482]*y[IDX_CH3II] + k[483]*y[IDX_CH3II] +
        k[497]*y[IDX_CH5II] + k[522]*y[IDX_H3COII] + k[546]*y[IDX_HCSII];
    data[4076] = 0.0 - k[9]*y[IDX_CHI] + k[1736]*y[IDX_C2I] + k[1739]*y[IDX_CH2I] -
        k[1743]*y[IDX_CHI] + k[2123]*y[IDX_CI];
    data[4077] = 0.0 + k[2307] + k[2308] + k[2309] + k[2310];
    data[4078] = 0.0 - k[1420]*y[IDX_NH3I];
    data[4079] = 0.0 - k[1438]*y[IDX_NH3I];
    data[4080] = 0.0 - k[1439]*y[IDX_NH3I];
    data[4081] = 0.0 - k[792]*y[IDX_NH3I];
    data[4082] = 0.0 - k[291]*y[IDX_NH3I];
    data[4083] = 0.0 - k[1443]*y[IDX_NH3I];
    data[4084] = 0.0 - k[1442]*y[IDX_NH3I];
    data[4085] = 0.0 - k[1421]*y[IDX_NH3I];
    data[4086] = 0.0 - k[1434]*y[IDX_NH3I];
    data[4087] = 0.0 - k[1435]*y[IDX_NH3I];
    data[4088] = 0.0 - k[1429]*y[IDX_NH3I];
    data[4089] = 0.0 - k[78]*y[IDX_NH3I] - k[801]*y[IDX_NH3I];
    data[4090] = 0.0 + k[279]*y[IDX_NH3II];
    data[4091] = 0.0 - k[293]*y[IDX_NH3I];
    data[4092] = 0.0 - k[1441]*y[IDX_NH3I];
    data[4093] = 0.0 - k[1440]*y[IDX_NH3I];
    data[4094] = 0.0 - k[286]*y[IDX_NH3I] - k[1426]*y[IDX_NH3I];
    data[4095] = 0.0 - k[1436]*y[IDX_NH3I];
    data[4096] = 0.0 - k[1422]*y[IDX_NH3I];
    data[4097] = 0.0 - k[289]*y[IDX_NH3I];
    data[4098] = 0.0 - k[1431]*y[IDX_NH3I] - k[1432]*y[IDX_NH3I];
    data[4099] = 0.0 - k[167]*y[IDX_NH3I];
    data[4100] = 0.0 - k[1427]*y[IDX_NH3I];
    data[4101] = 0.0 + k[281]*y[IDX_NH3II];
    data[4102] = 0.0 - k[287]*y[IDX_NH3I] - k[1430]*y[IDX_NH3I];
    data[4103] = 0.0 - k[283]*y[IDX_NH3I] - k[1423]*y[IDX_NH3I];
    data[4104] = 0.0 - k[288]*y[IDX_NH3I] - k[1437]*y[IDX_NH3I];
    data[4105] = 0.0 - k[262]*y[IDX_NH3I] - k[1365]*y[IDX_NH3I];
    data[4106] = 0.0 + k[1729]*y[IDX_H2I] + k[1833]*y[IDX_CH4I] + k[1837]*y[IDX_OHI];
    data[4107] = 0.0 - k[290]*y[IDX_NH3I];
    data[4108] = 0.0 - k[246]*y[IDX_NH3I] - k[1306]*y[IDX_NH3I] - k[1307]*y[IDX_NH3I];
    data[4109] = 0.0 - k[268]*y[IDX_NH3I] + k[1382]*y[IDX_H2SI] - k[1389]*y[IDX_NH3I];
    data[4110] = 0.0 - k[1418]*y[IDX_NH3I];
    data[4111] = 0.0 + k[574]*y[IDX_EM];
    data[4112] = 0.0 - k[316]*y[IDX_NH3I];
    data[4113] = 0.0 - k[284]*y[IDX_NH3I] - k[1424]*y[IDX_NH3I];
    data[4114] = 0.0 - k[748]*y[IDX_NH3I];
    data[4115] = 0.0 + k[278]*y[IDX_HCOI] + k[279]*y[IDX_MgI] + k[280]*y[IDX_NOI] +
        k[281]*y[IDX_SiI] - k[1417]*y[IDX_NH3I];
    data[4116] = 0.0 + k[1382]*y[IDX_NH2II];
    data[4117] = 0.0 - k[285]*y[IDX_NH3I] - k[1425]*y[IDX_NH3I];
    data[4118] = 0.0 - k[335]*y[IDX_NH3I] - k[1530]*y[IDX_NH3I];
    data[4119] = 0.0 - k[781]*y[IDX_NH3I];
    data[4120] = 0.0 - k[282]*y[IDX_NH3I] - k[1419]*y[IDX_NH3I];
    data[4121] = 0.0 - k[1657]*y[IDX_NH3I];
    data[4122] = 0.0 - k[57]*y[IDX_NH3I] - k[730]*y[IDX_NH3I];
    data[4123] = 0.0 - k[1842]*y[IDX_NH3I];
    data[4124] = 0.0 - k[292]*y[IDX_NH3I];
    data[4125] = 0.0 - k[1838]*y[IDX_NH3I];
    data[4126] = 0.0 + k[1833]*y[IDX_NH2I];
    data[4127] = 0.0 + k[278]*y[IDX_NH3II];
    data[4128] = 0.0 + k[280]*y[IDX_NH3II];
    data[4129] = 0.0 - k[21]*y[IDX_CII] - k[57]*y[IDX_CHII] - k[78]*y[IDX_CH4II] -
        k[131]*y[IDX_HII] - k[167]*y[IDX_H2II] - k[216]*y[IDX_HeII] -
        k[246]*y[IDX_NII] - k[262]*y[IDX_NHII] - k[268]*y[IDX_NH2II] -
        k[282]*y[IDX_C2H2II] - k[283]*y[IDX_COII] - k[284]*y[IDX_H2COII] -
        k[285]*y[IDX_H2OII] - k[286]*y[IDX_H2SII] - k[287]*y[IDX_HCNII] -
        k[288]*y[IDX_HSII] - k[289]*y[IDX_N2II] - k[290]*y[IDX_O2II] -
        k[291]*y[IDX_OCSII] - k[292]*y[IDX_SII] - k[293]*y[IDX_SOII] -
        k[316]*y[IDX_OII] - k[335]*y[IDX_OHII] - k[426] - k[427] - k[428] -
        k[630]*y[IDX_CII] - k[730]*y[IDX_CHII] - k[748]*y[IDX_CH2II] -
        k[781]*y[IDX_CH3II] - k[792]*y[IDX_CH3OH2II] - k[801]*y[IDX_CH4II] -
        k[1057]*y[IDX_H3II] - k[1254]*y[IDX_HeII] - k[1255]*y[IDX_HeII] -
        k[1306]*y[IDX_NII] - k[1307]*y[IDX_NII] - k[1365]*y[IDX_NHII] -
        k[1389]*y[IDX_NH2II] - k[1417]*y[IDX_NH3II] - k[1418]*y[IDX_C2HII] -
        k[1419]*y[IDX_C2H2II] - k[1420]*y[IDX_C2H5OH2II] - k[1421]*y[IDX_C2NII]
        - k[1422]*y[IDX_CH5II] - k[1423]*y[IDX_COII] - k[1424]*y[IDX_H2COII] -
        k[1425]*y[IDX_H2OII] - k[1426]*y[IDX_H2SII] - k[1427]*y[IDX_H3COII] -
        k[1428]*y[IDX_H3OII] - k[1429]*y[IDX_H3SII] - k[1430]*y[IDX_HCNII] -
        k[1431]*y[IDX_HCNHII] - k[1432]*y[IDX_HCNHII] - k[1433]*y[IDX_HCOII] -
        k[1434]*y[IDX_HCO2II] - k[1435]*y[IDX_HCSII] - k[1436]*y[IDX_HNOII] -
        k[1437]*y[IDX_HSII] - k[1438]*y[IDX_HSO2II] - k[1439]*y[IDX_HSiSII] -
        k[1440]*y[IDX_N2HII] - k[1441]*y[IDX_O2HII] - k[1442]*y[IDX_SiHII] -
        k[1443]*y[IDX_SiOHII] - k[1530]*y[IDX_OHII] - k[1657]*y[IDX_CH3I] -
        k[1761]*y[IDX_HI] - k[1838]*y[IDX_CNI] - k[1842]*y[IDX_NHI] -
        k[1904]*y[IDX_OI] - k[1944]*y[IDX_OHI] - k[2051] - k[2052] - k[2053] -
        k[2225];
    data[4130] = 0.0 - k[1428]*y[IDX_NH3I];
    data[4131] = 0.0 - k[21]*y[IDX_NH3I] - k[630]*y[IDX_NH3I];
    data[4132] = 0.0 + k[1837]*y[IDX_NH2I] - k[1944]*y[IDX_NH3I];
    data[4133] = 0.0 - k[216]*y[IDX_NH3I] - k[1254]*y[IDX_NH3I] - k[1255]*y[IDX_NH3I];
    data[4134] = 0.0 - k[1057]*y[IDX_NH3I];
    data[4135] = 0.0 - k[131]*y[IDX_NH3I];
    data[4136] = 0.0 - k[1904]*y[IDX_NH3I];
    data[4137] = 0.0 - k[1433]*y[IDX_NH3I];
    data[4138] = 0.0 + k[1729]*y[IDX_NH2I];
    data[4139] = 0.0 + k[574]*y[IDX_NH4II];
    data[4140] = 0.0 - k[1761]*y[IDX_NH3I];
    data[4141] = 0.0 + k[1016]*y[IDX_H2OI];
    data[4142] = 0.0 + k[999]*y[IDX_H2OI];
    data[4143] = 0.0 + k[1015]*y[IDX_H2OI];
    data[4144] = 0.0 + k[1008]*y[IDX_H2OI];
    data[4145] = 0.0 - k[1089]*y[IDX_H3OII];
    data[4146] = 0.0 + k[1006]*y[IDX_H2OI];
    data[4147] = 0.0 - k[1091]*y[IDX_H3OII];
    data[4148] = 0.0 - k[1082]*y[IDX_H3OII];
    data[4149] = 0.0 - k[1081]*y[IDX_H3OII];
    data[4150] = 0.0 - k[1094]*y[IDX_H3OII];
    data[4151] = 0.0 - k[1092]*y[IDX_H3OII];
    data[4152] = 0.0 - k[1083]*y[IDX_H3OII];
    data[4153] = 0.0 - k[1095]*y[IDX_H3OII];
    data[4154] = 0.0 + k[1014]*y[IDX_H2OI];
    data[4155] = 0.0 + k[1004]*y[IDX_H2OI];
    data[4156] = 0.0 - k[1096]*y[IDX_H3OII];
    data[4157] = 0.0 - k[1084]*y[IDX_H3OII];
    data[4158] = 0.0 - k[2121]*y[IDX_H3OII];
    data[4159] = 0.0 + k[799]*y[IDX_H2OI];
    data[4160] = 0.0 - k[1085]*y[IDX_H3OII];
    data[4161] = 0.0 + k[1012]*y[IDX_H2OI];
    data[4162] = 0.0 + k[1011]*y[IDX_H2OI];
    data[4163] = 0.0 + k[1000]*y[IDX_H2OI];
    data[4164] = 0.0 + k[1005]*y[IDX_H2OI];
    data[4165] = 0.0 + k[828]*y[IDX_H2OI] + k[1495]*y[IDX_OI];
    data[4166] = 0.0 + k[922]*y[IDX_H2OI];
    data[4167] = 0.0 + k[1001]*y[IDX_H2OI];
    data[4168] = 0.0 - k[1093]*y[IDX_H3OII];
    data[4169] = 0.0 + k[1002]*y[IDX_H2OI];
    data[4170] = 0.0 - k[1090]*y[IDX_H3OII];
    data[4171] = 0.0 + k[1007]*y[IDX_H2OI];
    data[4172] = 0.0 + k[1356]*y[IDX_H2OI];
    data[4173] = 0.0 - k[1402]*y[IDX_H3OII];
    data[4174] = 0.0 + k[1378]*y[IDX_H2OI];
    data[4175] = 0.0 + k[998]*y[IDX_H2OI];
    data[4176] = 0.0 + k[982]*y[IDX_H2OII] - k[1087]*y[IDX_H3OII];
    data[4177] = 0.0 + k[810]*y[IDX_CH4I] + k[945]*y[IDX_H2I] + k[980]*y[IDX_H2OI] +
        k[982]*y[IDX_H2SI] + k[984]*y[IDX_HCOI] + k[1450]*y[IDX_NHI] +
        k[1540]*y[IDX_OHI];
    data[4178] = 0.0 + k[820]*y[IDX_CH4I] + k[1523]*y[IDX_H2OI];
    data[4179] = 0.0 + k[991]*y[IDX_H2OI];
    data[4180] = 0.0 - k[1080]*y[IDX_H3OII];
    data[4181] = 0.0 - k[759]*y[IDX_H3OII];
    data[4182] = 0.0 + k[719]*y[IDX_H2OI];
    data[4183] = 0.0 + k[1450]*y[IDX_H2OII];
    data[4184] = 0.0 - k[1086]*y[IDX_H3OII];
    data[4185] = 0.0 + k[810]*y[IDX_H2OII] + k[820]*y[IDX_OHII];
    data[4186] = 0.0 + k[984]*y[IDX_H2OII];
    data[4187] = 0.0 - k[1088]*y[IDX_H3OII];
    data[4188] = 0.0 - k[845]*y[IDX_H3OII];
    data[4189] = 0.0 - k[1428]*y[IDX_H3OII];
    data[4190] = 0.0 - k[528]*y[IDX_EM] - k[529]*y[IDX_EM] - k[530]*y[IDX_EM] -
        k[531]*y[IDX_EM] - k[692]*y[IDX_CI] - k[759]*y[IDX_CH2I] -
        k[845]*y[IDX_CHI] - k[1080]*y[IDX_C2I] - k[1081]*y[IDX_C2H5OHI] -
        k[1082]*y[IDX_CH3CCHI] - k[1083]*y[IDX_CH3CNI] - k[1084]*y[IDX_CH3OHI] -
        k[1085]*y[IDX_CSI] - k[1086]*y[IDX_H2COI] - k[1087]*y[IDX_H2SI] -
        k[1088]*y[IDX_HCNI] - k[1089]*y[IDX_HCOOCH3I] - k[1090]*y[IDX_HNCI] -
        k[1091]*y[IDX_HS2I] - k[1092]*y[IDX_S2I] - k[1093]*y[IDX_SiI] -
        k[1094]*y[IDX_SiH2I] - k[1095]*y[IDX_SiHI] - k[1096]*y[IDX_SiOI] -
        k[1402]*y[IDX_NH2I] - k[1428]*y[IDX_NH3I] - k[2121]*y[IDX_C2H4I] -
        k[2246];
    data[4191] = 0.0 + k[1540]*y[IDX_H2OII];
    data[4192] = 0.0 + k[1043]*y[IDX_H2OI];
    data[4193] = 0.0 + k[1495]*y[IDX_CH5II];
    data[4194] = 0.0 - k[692]*y[IDX_H3OII];
    data[4195] = 0.0 + k[1003]*y[IDX_H2OI];
    data[4196] = 0.0 + k[719]*y[IDX_CHII] + k[799]*y[IDX_CH4II] + k[828]*y[IDX_CH5II] +
        k[922]*y[IDX_H2II] + k[980]*y[IDX_H2OII] + k[991]*y[IDX_C2H2II] +
        k[998]*y[IDX_H2COII] + k[999]*y[IDX_H2ClII] + k[1000]*y[IDX_H2SII] +
        k[1001]*y[IDX_H3COII] + k[1002]*y[IDX_HCNII] + k[1003]*y[IDX_HCOII] +
        k[1004]*y[IDX_HCO2II] + k[1005]*y[IDX_HNOII] + k[1006]*y[IDX_HOCSII] +
        k[1007]*y[IDX_HSII] + k[1008]*y[IDX_HSO2II] + k[1011]*y[IDX_N2HII] +
        k[1012]*y[IDX_O2HII] + k[1014]*y[IDX_SiHII] + k[1015]*y[IDX_SiH4II] +
        k[1016]*y[IDX_SiH5II] + k[1043]*y[IDX_H3II] + k[1356]*y[IDX_NHII] +
        k[1378]*y[IDX_NH2II] + k[1523]*y[IDX_OHII];
    data[4197] = 0.0 + k[945]*y[IDX_H2OII];
    data[4198] = 0.0 - k[528]*y[IDX_H3OII] - k[529]*y[IDX_H3OII] - k[530]*y[IDX_H3OII] -
        k[531]*y[IDX_H3OII];
    data[4199] = 0.0 + k[2375] + k[2376] + k[2377] + k[2378];
    data[4200] = 0.0 - k[324]*y[IDX_O2I];
    data[4201] = 0.0 - k[325]*y[IDX_O2I];
    data[4202] = 0.0 - k[1487]*y[IDX_O2I] - k[1488]*y[IDX_O2I];
    data[4203] = 0.0 + k[437] + k[1663]*y[IDX_CH3I] + k[1692]*y[IDX_CHI] +
        k[1770]*y[IDX_HI] + k[1786]*y[IDX_HCOI] + k[1826]*y[IDX_NI] +
        k[1909]*y[IDX_OI] + k[1946]*y[IDX_OHI] + k[2063];
    data[4204] = 0.0 + k[1478]*y[IDX_OII] + k[1822]*y[IDX_NI] + k[1905]*y[IDX_OI];
    data[4205] = 0.0 - k[1863]*y[IDX_O2I] - k[1864]*y[IDX_O2I] + k[1911]*y[IDX_OI];
    data[4206] = 0.0 - k[1567]*y[IDX_O2I];
    data[4207] = 0.0 + k[1270]*y[IDX_HeII] + k[1481]*y[IDX_OII] + k[1916]*y[IDX_OI];
    data[4208] = 0.0 + k[1898]*y[IDX_OI];
    data[4209] = 0.0 + k[1516]*y[IDX_OI];
    data[4210] = 0.0 - k[1576]*y[IDX_O2I] - k[1577]*y[IDX_O2I];
    data[4211] = 0.0 + k[1500]*y[IDX_OI];
    data[4212] = 0.0 - k[1485]*y[IDX_O2I];
    data[4213] = 0.0 + k[1483]*y[IDX_O2II];
    data[4214] = 0.0 - k[1866]*y[IDX_O2I] + k[1917]*y[IDX_OI];
    data[4215] = 0.0 - k[79]*y[IDX_O2I];
    data[4216] = 0.0 + k[228]*y[IDX_O2II];
    data[4217] = 0.0 - k[649]*y[IDX_O2I];
    data[4218] = 0.0 + k[578]*y[IDX_EM] + k[657]*y[IDX_C2I] + k[683]*y[IDX_C2HI] +
        k[701]*y[IDX_CI] + k[770]*y[IDX_CH2I] + k[859]*y[IDX_CHI] +
        k[870]*y[IDX_CNI] + k[878]*y[IDX_COI] + k[959]*y[IDX_H2I] +
        k[973]*y[IDX_H2COI] + k[1012]*y[IDX_H2OI] + k[1129]*y[IDX_HCNI] +
        k[1162]*y[IDX_HCOI] + k[1172]*y[IDX_HNCI] + k[1320]*y[IDX_N2I] +
        k[1410]*y[IDX_NH2I] + k[1441]*y[IDX_NH3I] + k[1459]*y[IDX_NHI] +
        k[1462]*y[IDX_NOI] + k[1489]*y[IDX_CO2I] + k[1510]*y[IDX_OI] +
        k[1547]*y[IDX_OHI] + k[1554]*y[IDX_SI];
    data[4219] = 0.0 - k[98]*y[IDX_O2I] - k[868]*y[IDX_O2I];
    data[4220] = 0.0 - k[256]*y[IDX_O2I];
    data[4221] = 0.0 - k[170]*y[IDX_O2I] - k[931]*y[IDX_O2I];
    data[4222] = 0.0 + k[353]*y[IDX_O2II] - k[1959]*y[IDX_O2I];
    data[4223] = 0.0 + k[321]*y[IDX_O2II];
    data[4224] = 0.0 - k[198]*y[IDX_O2I];
    data[4225] = 0.0 + k[1172]*y[IDX_O2HII];
    data[4226] = 0.0 - k[105]*y[IDX_O2I];
    data[4227] = 0.0 + k[1212]*y[IDX_HeII] + k[1489]*y[IDX_O2HII] + k[1882]*y[IDX_OI];
    data[4228] = 0.0 - k[264]*y[IDX_O2I] - k[1368]*y[IDX_O2I] - k[1369]*y[IDX_O2I];
    data[4229] = 0.0 + k[276]*y[IDX_O2II] + k[1410]*y[IDX_O2HII];
    data[4230] = 0.0 + k[39]*y[IDX_C2I] + k[54]*y[IDX_CI] + k[70]*y[IDX_CH2I] +
        k[91]*y[IDX_CHI] + k[174]*y[IDX_H2COI] + k[204]*y[IDX_HCOI] +
        k[228]*y[IDX_MgI] + k[276]*y[IDX_NH2I] + k[290]*y[IDX_NH3I] +
        k[302]*y[IDX_NOI] + k[321]*y[IDX_C2H2I] + k[322]*y[IDX_H2SI] +
        k[323]*y[IDX_SI] + k[353]*y[IDX_SiI] + k[972]*y[IDX_H2COI] +
        k[1483]*y[IDX_CH3OHI];
    data[4231] = 0.0 - k[249]*y[IDX_O2I] - k[1310]*y[IDX_O2I] - k[1311]*y[IDX_O2I];
    data[4232] = 0.0 - k[1390]*y[IDX_O2I] - k[1391]*y[IDX_O2I];
    data[4233] = 0.0 + k[683]*y[IDX_O2HII] - k[1581]*y[IDX_O2I];
    data[4234] = 0.0 - k[317]*y[IDX_O2I] + k[1478]*y[IDX_NO2I] + k[1481]*y[IDX_SO2I];
    data[4235] = 0.0 - k[967]*y[IDX_O2I];
    data[4236] = 0.0 - k[749]*y[IDX_O2I];
    data[4237] = 0.0 + k[322]*y[IDX_O2II];
    data[4238] = 0.0 - k[183]*y[IDX_O2I];
    data[4239] = 0.0 - k[337]*y[IDX_O2I];
    data[4240] = 0.0 - k[782]*y[IDX_O2I];
    data[4241] = 0.0 + k[1320]*y[IDX_O2HII];
    data[4242] = 0.0 + k[39]*y[IDX_O2II] + k[657]*y[IDX_O2HII] - k[1572]*y[IDX_O2I];
    data[4243] = 0.0 + k[70]*y[IDX_O2II] + k[770]*y[IDX_O2HII] - k[1632]*y[IDX_O2I] -
        k[1633]*y[IDX_O2I] - k[1634]*y[IDX_O2I] - k[1635]*y[IDX_O2I] -
        k[1636]*y[IDX_O2I];
    data[4244] = 0.0 - k[1660]*y[IDX_O2I] - k[1661]*y[IDX_O2I] - k[1662]*y[IDX_O2I] +
        k[1663]*y[IDX_O2HI];
    data[4245] = 0.0 - k[732]*y[IDX_O2I] - k[733]*y[IDX_O2I] - k[734]*y[IDX_O2I];
    data[4246] = 0.0 + k[1459]*y[IDX_O2HII] - k[1849]*y[IDX_O2I] - k[1850]*y[IDX_O2I];
    data[4247] = 0.0 - k[1486]*y[IDX_O2I];
    data[4248] = 0.0 + k[870]*y[IDX_O2HII] - k[1712]*y[IDX_O2I] - k[1713]*y[IDX_O2I];
    data[4249] = 0.0 + k[174]*y[IDX_O2II] + k[972]*y[IDX_O2II] + k[973]*y[IDX_O2HII];
    data[4250] = 0.0 - k[1671]*y[IDX_O2I];
    data[4251] = 0.0 + k[204]*y[IDX_O2II] + k[1162]*y[IDX_O2HII] - k[1784]*y[IDX_O2I] -
        k[1785]*y[IDX_O2I] + k[1786]*y[IDX_O2HI];
    data[4252] = 0.0 + k[1129]*y[IDX_O2HII];
    data[4253] = 0.0 + k[302]*y[IDX_O2II] + k[1462]*y[IDX_O2HII] + k[1858]*y[IDX_NOI] +
        k[1858]*y[IDX_NOI] - k[1859]*y[IDX_O2I] + k[1906]*y[IDX_OI];
    data[4254] = 0.0 + k[91]*y[IDX_O2II] + k[859]*y[IDX_O2HII] - k[1687]*y[IDX_O2I] -
        k[1688]*y[IDX_O2I] - k[1689]*y[IDX_O2I] - k[1690]*y[IDX_O2I] +
        k[1692]*y[IDX_O2HI];
    data[4255] = 0.0 + k[290]*y[IDX_O2II] + k[1441]*y[IDX_O2HII];
    data[4256] = 0.0 - k[6]*y[IDX_H2I] - k[12]*y[IDX_HI] - k[79]*y[IDX_CH4II] -
        k[98]*y[IDX_CNII] - k[105]*y[IDX_COII] - k[135]*y[IDX_HII] -
        k[170]*y[IDX_H2II] - k[183]*y[IDX_H2OII] - k[198]*y[IDX_HCNII] -
        k[217]*y[IDX_HeII] - k[249]*y[IDX_NII] - k[256]*y[IDX_N2II] -
        k[264]*y[IDX_NHII] - k[317]*y[IDX_OII] - k[324]*y[IDX_ClII] -
        k[325]*y[IDX_SO2II] - k[337]*y[IDX_OHII] - k[435] - k[436] -
        k[633]*y[IDX_CII] - k[634]*y[IDX_CII] - k[649]*y[IDX_C2II] -
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
    data[4257] = 0.0 - k[633]*y[IDX_O2I] - k[634]*y[IDX_O2I];
    data[4258] = 0.0 + k[323]*y[IDX_O2II] + k[1554]*y[IDX_O2HII] - k[1865]*y[IDX_O2I];
    data[4259] = 0.0 + k[1547]*y[IDX_O2HII] + k[1914]*y[IDX_OI] + k[1946]*y[IDX_O2HI];
    data[4260] = 0.0 + k[1822]*y[IDX_NO2I] - k[1825]*y[IDX_O2I] + k[1826]*y[IDX_O2HI];
    data[4261] = 0.0 - k[217]*y[IDX_O2I] + k[1212]*y[IDX_CO2I] - k[1261]*y[IDX_O2I] +
        k[1270]*y[IDX_SO2I];
    data[4262] = 0.0 - k[1062]*y[IDX_O2I];
    data[4263] = 0.0 - k[135]*y[IDX_O2I];
    data[4264] = 0.0 + k[1500]*y[IDX_HCO2II] + k[1510]*y[IDX_O2HII] +
        k[1516]*y[IDX_SiOII] + k[1882]*y[IDX_CO2I] + k[1898]*y[IDX_HNOI] +
        k[1905]*y[IDX_NO2I] + k[1906]*y[IDX_NOI] + k[1909]*y[IDX_O2HI] +
        k[1911]*y[IDX_OCNI] + k[1914]*y[IDX_OHI] + k[1916]*y[IDX_SO2I] +
        k[1917]*y[IDX_SOI] + k[2128]*y[IDX_OI] + k[2128]*y[IDX_OI];
    data[4265] = 0.0 + k[54]*y[IDX_O2II] + k[701]*y[IDX_O2HII] - k[1608]*y[IDX_O2I];
    data[4266] = 0.0 + k[1012]*y[IDX_O2HII];
    data[4267] = 0.0 + k[878]*y[IDX_O2HII] - k[1718]*y[IDX_O2I];
    data[4268] = 0.0 - k[6]*y[IDX_O2I] + k[959]*y[IDX_O2HII] - k[1731]*y[IDX_O2I] -
        k[1732]*y[IDX_O2I];
    data[4269] = 0.0 + k[578]*y[IDX_O2HII];
    data[4270] = 0.0 - k[12]*y[IDX_O2I] - k[1768]*y[IDX_O2I] + k[1770]*y[IDX_O2HI];
    data[4271] = 0.0 - k[28]*y[IDX_CII];
    data[4272] = 0.0 - k[611]*y[IDX_CII];
    data[4273] = 0.0 - k[27]*y[IDX_CII];
    data[4274] = 0.0 - k[619]*y[IDX_CII];
    data[4275] = 0.0 - k[623]*y[IDX_CII] - k[624]*y[IDX_CII] - k[625]*y[IDX_CII];
    data[4276] = 0.0 - k[606]*y[IDX_CII];
    data[4277] = 0.0 + k[1189]*y[IDX_HeII];
    data[4278] = 0.0 - k[20]*y[IDX_CII];
    data[4279] = 0.0 - k[29]*y[IDX_CII] - k[642]*y[IDX_CII] + k[1277]*y[IDX_HeII];
    data[4280] = 0.0 - k[30]*y[IDX_CII] - k[643]*y[IDX_CII];
    data[4281] = 0.0 - k[32]*y[IDX_CII] - k[646]*y[IDX_CII];
    data[4282] = 0.0 - k[635]*y[IDX_CII];
    data[4283] = 0.0 - k[31]*y[IDX_CII];
    data[4284] = 0.0 - k[23]*y[IDX_CII] - k[632]*y[IDX_CII];
    data[4285] = 0.0 - k[638]*y[IDX_CII];
    data[4286] = 0.0 - k[644]*y[IDX_CII];
    data[4287] = 0.0 - k[645]*y[IDX_CII];
    data[4288] = 0.0 - k[612]*y[IDX_CII] - k[613]*y[IDX_CII];
    data[4289] = 0.0 - k[25]*y[IDX_CII] - k[639]*y[IDX_CII] - k[640]*y[IDX_CII] -
        k[641]*y[IDX_CII];
    data[4290] = 0.0 + k[1215]*y[IDX_HeII];
    data[4291] = 0.0 - k[19]*y[IDX_CII];
    data[4292] = 0.0 + k[50]*y[IDX_CI] + k[1325]*y[IDX_NI] + k[1960];
    data[4293] = 0.0 + k[51]*y[IDX_CI];
    data[4294] = 0.0 - k[24]*y[IDX_CII] - k[636]*y[IDX_CII];
    data[4295] = 0.0 + k[53]*y[IDX_CI];
    data[4296] = 0.0 - k[628]*y[IDX_CII];
    data[4297] = 0.0 - k[26]*y[IDX_CII];
    data[4298] = 0.0 - k[627]*y[IDX_CII] + k[1243]*y[IDX_HeII];
    data[4299] = 0.0 + k[52]*y[IDX_CI] + k[2002];
    data[4300] = 0.0 - k[616]*y[IDX_CII] + k[1212]*y[IDX_HeII];
    data[4301] = 0.0 - k[629]*y[IDX_CII];
    data[4302] = 0.0 + k[54]*y[IDX_CI];
    data[4303] = 0.0 - k[607]*y[IDX_CII] + k[1188]*y[IDX_HeII];
    data[4304] = 0.0 + k[1978];
    data[4305] = 0.0 - k[17]*y[IDX_CII] - k[622]*y[IDX_CII];
    data[4306] = 0.0 + k[1177]*y[IDX_HeII];
    data[4307] = 0.0 - k[14]*y[IDX_CII] - k[608]*y[IDX_CII] + k[1192]*y[IDX_HeII];
    data[4308] = 0.0 - k[609]*y[IDX_CII] - k[610]*y[IDX_CII];
    data[4309] = 0.0 + k[380] + k[1098]*y[IDX_HI];
    data[4310] = 0.0 - k[631]*y[IDX_CII];
    data[4311] = 0.0 + k[1208]*y[IDX_HeII];
    data[4312] = 0.0 - k[16]*y[IDX_CII] - k[617]*y[IDX_CII] - k[618]*y[IDX_CII];
    data[4313] = 0.0 - k[614]*y[IDX_CII];
    data[4314] = 0.0 - k[18]*y[IDX_CII] - k[626]*y[IDX_CII];
    data[4315] = 0.0 + k[1233]*y[IDX_HeII];
    data[4316] = 0.0 - k[22]*y[IDX_CII];
    data[4317] = 0.0 - k[15]*y[IDX_CII] - k[615]*y[IDX_CII] + k[1206]*y[IDX_HeII];
    data[4318] = 0.0 - k[21]*y[IDX_CII] - k[630]*y[IDX_CII];
    data[4319] = 0.0 - k[633]*y[IDX_CII] - k[634]*y[IDX_CII];
    data[4320] = 0.0 - k[14]*y[IDX_CH2I] - k[15]*y[IDX_CHI] - k[16]*y[IDX_H2COI] -
        k[17]*y[IDX_H2SI] - k[18]*y[IDX_HCOI] - k[19]*y[IDX_MgI] -
        k[20]*y[IDX_NCCNI] - k[21]*y[IDX_NH3I] - k[22]*y[IDX_NOI] -
        k[23]*y[IDX_NSI] - k[24]*y[IDX_OCSI] - k[25]*y[IDX_SOI] -
        k[26]*y[IDX_SiI] - k[27]*y[IDX_SiC2I] - k[28]*y[IDX_SiC3I] -
        k[29]*y[IDX_SiCI] - k[30]*y[IDX_SiH2I] - k[31]*y[IDX_SiH3I] -
        k[32]*y[IDX_SiSI] - k[345]*y[IDX_SI] - k[606]*y[IDX_C2H5OHI] -
        k[607]*y[IDX_C2HI] - k[608]*y[IDX_CH2I] - k[609]*y[IDX_CH3I] -
        k[610]*y[IDX_CH3I] - k[611]*y[IDX_CH3CCHI] - k[612]*y[IDX_CH3OHI] -
        k[613]*y[IDX_CH3OHI] - k[614]*y[IDX_CH4I] - k[615]*y[IDX_CHI] -
        k[616]*y[IDX_CO2I] - k[617]*y[IDX_H2COI] - k[618]*y[IDX_H2COI] -
        k[619]*y[IDX_H2CSI] - k[620]*y[IDX_H2OI] - k[621]*y[IDX_H2OI] -
        k[622]*y[IDX_H2SI] - k[623]*y[IDX_HC3NI] - k[624]*y[IDX_HC3NI] -
        k[625]*y[IDX_HC3NI] - k[626]*y[IDX_HCOI] - k[627]*y[IDX_HNCI] -
        k[628]*y[IDX_HSI] - k[629]*y[IDX_NH2I] - k[630]*y[IDX_NH3I] -
        k[631]*y[IDX_NHI] - k[632]*y[IDX_NSI] - k[633]*y[IDX_O2I] -
        k[634]*y[IDX_O2I] - k[635]*y[IDX_OCNI] - k[636]*y[IDX_OCSI] -
        k[637]*y[IDX_OHI] - k[638]*y[IDX_SO2I] - k[639]*y[IDX_SOI] -
        k[640]*y[IDX_SOI] - k[641]*y[IDX_SOI] - k[642]*y[IDX_SiCI] -
        k[643]*y[IDX_SiH2I] - k[644]*y[IDX_SiHI] - k[645]*y[IDX_SiOI] -
        k[646]*y[IDX_SiSI] - k[934]*y[IDX_H2I] - k[2096]*y[IDX_CI] -
        k[2097]*y[IDX_NI] - k[2098]*y[IDX_OI] - k[2099]*y[IDX_SI] -
        k[2112]*y[IDX_H2I] - k[2122]*y[IDX_HI] - k[2132]*y[IDX_EM] - k[2221];
    data[4321] = 0.0 - k[345]*y[IDX_CII] - k[2099]*y[IDX_CII];
    data[4322] = 0.0 - k[637]*y[IDX_CII];
    data[4323] = 0.0 + k[1325]*y[IDX_C2II] - k[2097]*y[IDX_CII];
    data[4324] = 0.0 + k[209]*y[IDX_CI] + k[1177]*y[IDX_C2I] + k[1188]*y[IDX_C2HI] +
        k[1189]*y[IDX_C2NI] + k[1192]*y[IDX_CH2I] + k[1206]*y[IDX_CHI] +
        k[1208]*y[IDX_CNI] + k[1212]*y[IDX_CO2I] + k[1213]*y[IDX_COI] +
        k[1215]*y[IDX_CSI] + k[1233]*y[IDX_HCNI] + k[1243]*y[IDX_HNCI] +
        k[1277]*y[IDX_SiCI];
    data[4325] = 0.0 - k[2098]*y[IDX_CII];
    data[4326] = 0.0 + k[50]*y[IDX_C2II] + k[51]*y[IDX_CNII] + k[52]*y[IDX_COII] +
        k[53]*y[IDX_N2II] + k[54]*y[IDX_O2II] + k[209]*y[IDX_HeII] + k[356] +
        k[379] + k[1976] - k[2096]*y[IDX_CII];
    data[4327] = 0.0 - k[620]*y[IDX_CII] - k[621]*y[IDX_CII];
    data[4328] = 0.0 + k[1213]*y[IDX_HeII];
    data[4329] = 0.0 - k[934]*y[IDX_CII] - k[2112]*y[IDX_CII];
    data[4330] = 0.0 - k[2132]*y[IDX_CII];
    data[4331] = 0.0 + k[1098]*y[IDX_CHII] - k[2122]*y[IDX_CII];
    data[4332] = 0.0 + k[417] + k[2042];
    data[4333] = 0.0 + k[576]*y[IDX_EM] + k[1509]*y[IDX_OI];
    data[4334] = 0.0 + k[585]*y[IDX_EM];
    data[4335] = 0.0 + k[1221]*y[IDX_HeII];
    data[4336] = 0.0 + k[605]*y[IDX_EM];
    data[4337] = 0.0 + k[555]*y[IDX_EM];
    data[4338] = 0.0 + k[583]*y[IDX_EM] + k[583]*y[IDX_EM];
    data[4339] = 0.0 + k[343]*y[IDX_SII];
    data[4340] = 0.0 + k[1814]*y[IDX_NI];
    data[4341] = 0.0 + k[344]*y[IDX_SII] + k[457] + k[646]*y[IDX_CII] +
        k[1288]*y[IDX_HeII] + k[2095];
    data[4342] = 0.0 + k[443] + k[443] + k[1269]*y[IDX_HeII] + k[1613]*y[IDX_CI] +
        k[1777]*y[IDX_HI] + k[1829]*y[IDX_NI] + k[1915]*y[IDX_OI] + k[2072] +
        k[2072];
    data[4343] = 0.0 + k[434] + k[1260]*y[IDX_HeII] + k[1607]*y[IDX_CI] +
        k[1767]*y[IDX_HI] + k[1824]*y[IDX_NI] + k[1907]*y[IDX_OI] + k[2059];
    data[4344] = 0.0 + k[581]*y[IDX_EM];
    data[4345] = 0.0 - k[1568]*y[IDX_SI];
    data[4346] = 0.0 - k[1954]*y[IDX_SI];
    data[4347] = 0.0 + k[355]*y[IDX_SII];
    data[4348] = 0.0 - k[1555]*y[IDX_SI];
    data[4349] = 0.0 + k[500]*y[IDX_EM] + k[1496]*y[IDX_OI];
    data[4350] = 0.0 + k[546]*y[IDX_EM] + k[1502]*y[IDX_OI];
    data[4351] = 0.0 + k[535]*y[IDX_EM] - k[1553]*y[IDX_SI];
    data[4352] = 0.0 + k[446] + k[641]*y[IDX_CII] + k[1273]*y[IDX_HeII] +
        k[1616]*y[IDX_CI] + k[1779]*y[IDX_HI] + k[1831]*y[IDX_NI] +
        k[1917]*y[IDX_OI] - k[1955]*y[IDX_SI] + k[2075];
    data[4353] = 0.0 + k[396] + k[1215]*y[IDX_HeII] + k[1592]*y[IDX_CI] +
        k[1809]*y[IDX_NI] + k[1883]*y[IDX_OI] + k[2007];
    data[4354] = 0.0 + k[229]*y[IDX_SII];
    data[4355] = 0.0 + k[584]*y[IDX_EM];
    data[4356] = 0.0 - k[35]*y[IDX_SI] - k[650]*y[IDX_SI];
    data[4357] = 0.0 - k[1554]*y[IDX_SI];
    data[4358] = 0.0 - k[99]*y[IDX_SI];
    data[4359] = 0.0 - k[1324]*y[IDX_SI];
    data[4360] = 0.0 + k[441] + k[1267]*y[IDX_HeII] + k[1912]*y[IDX_OI] + k[2067];
    data[4361] = 0.0 - k[346]*y[IDX_SI] + k[516]*y[IDX_EM];
    data[4362] = 0.0 - k[1174]*y[IDX_SI];
    data[4363] = 0.0 - k[835]*y[IDX_SI];
    data[4364] = 0.0 - k[258]*y[IDX_SI];
    data[4365] = 0.0 + k[418] + k[1596]*y[IDX_CI] + k[1758]*y[IDX_HI] +
        k[1789]*y[IDX_HSI] + k[1789]*y[IDX_HSI] + k[1817]*y[IDX_NI] +
        k[1899]*y[IDX_OI] - k[1953]*y[IDX_SI] + k[2043];
    data[4366] = 0.0 + k[354]*y[IDX_SII];
    data[4367] = 0.0 - k[199]*y[IDX_SI] - k[1117]*y[IDX_SI];
    data[4368] = 0.0 + k[1170]*y[IDX_HSII];
    data[4369] = 0.0 - k[106]*y[IDX_SI];
    data[4370] = 0.0 - k[347]*y[IDX_SI] + k[554]*y[IDX_EM] + k[851]*y[IDX_CHI] +
        k[1007]*y[IDX_H2OI] + k[1126]*y[IDX_HCNI] + k[1170]*y[IDX_HNCI] +
        k[1175]*y[IDX_H2SI] + k[1437]*y[IDX_NH3I] + k[2040];
    data[4371] = 0.0 - k[265]*y[IDX_SI] - k[1372]*y[IDX_SI] - k[1373]*y[IDX_SI];
    data[4372] = 0.0 - k[323]*y[IDX_SI] - k[1484]*y[IDX_SI];
    data[4373] = 0.0 - k[270]*y[IDX_SI] + k[1384]*y[IDX_H2SI] - k[1392]*y[IDX_SI] -
        k[1393]*y[IDX_SI];
    data[4374] = 0.0 - k[41]*y[IDX_SI];
    data[4375] = 0.0 - k[173]*y[IDX_SI] - k[968]*y[IDX_SI];
    data[4376] = 0.0 - k[753]*y[IDX_SI];
    data[4377] = 0.0 + k[404] + k[1175]*y[IDX_HSII] + k[1384]*y[IDX_NH2II] + k[2022];
    data[4378] = 0.0 - k[185]*y[IDX_SI] - k[987]*y[IDX_SI] - k[988]*y[IDX_SI];
    data[4379] = 0.0 - k[338]*y[IDX_SI] - k[1533]*y[IDX_SI] - k[1534]*y[IDX_SI];
    data[4380] = 0.0 - k[787]*y[IDX_SI];
    data[4381] = 0.0 - k[1573]*y[IDX_SI];
    data[4382] = 0.0 - k[1644]*y[IDX_SI] - k[1645]*y[IDX_SI];
    data[4383] = 0.0 - k[1669]*y[IDX_SI];
    data[4384] = 0.0 - k[59]*y[IDX_SI] - k[739]*y[IDX_SI] - k[740]*y[IDX_SI];
    data[4385] = 0.0 - k[1856]*y[IDX_SI] - k[1857]*y[IDX_SI];
    data[4386] = 0.0 + k[205]*y[IDX_HCOI] + k[229]*y[IDX_MgI] + k[292]*y[IDX_NH3I] +
        k[303]*y[IDX_NOI] + k[343]*y[IDX_SiCI] + k[344]*y[IDX_SiSI] +
        k[354]*y[IDX_SiI] + k[355]*y[IDX_SiHI] + k[2143]*y[IDX_EM];
    data[4387] = 0.0 - k[1714]*y[IDX_SI];
    data[4388] = 0.0 - k[1673]*y[IDX_SI];
    data[4389] = 0.0 + k[205]*y[IDX_SII] - k[1951]*y[IDX_SI] - k[1952]*y[IDX_SI];
    data[4390] = 0.0 + k[1126]*y[IDX_HSII];
    data[4391] = 0.0 + k[303]*y[IDX_SII] - k[1861]*y[IDX_SI] - k[1862]*y[IDX_SI];
    data[4392] = 0.0 + k[851]*y[IDX_HSII] - k[1697]*y[IDX_SI] - k[1698]*y[IDX_SI];
    data[4393] = 0.0 + k[292]*y[IDX_SII] + k[1437]*y[IDX_HSII];
    data[4394] = 0.0 - k[1865]*y[IDX_SI];
    data[4395] = 0.0 - k[345]*y[IDX_SI] + k[641]*y[IDX_SOI] + k[646]*y[IDX_SiSI] -
        k[2099]*y[IDX_SI];
    data[4396] = 0.0 - k[35]*y[IDX_C2II] - k[41]*y[IDX_C2HII] - k[59]*y[IDX_CHII] -
        k[99]*y[IDX_CNII] - k[106]*y[IDX_COII] - k[140]*y[IDX_HII] -
        k[173]*y[IDX_H2COII] - k[185]*y[IDX_H2OII] - k[199]*y[IDX_HCNII] -
        k[258]*y[IDX_N2II] - k[265]*y[IDX_NHII] - k[270]*y[IDX_NH2II] -
        k[323]*y[IDX_O2II] - k[338]*y[IDX_OHII] - k[345]*y[IDX_CII] -
        k[346]*y[IDX_H2SII] - k[347]*y[IDX_HSII] - k[444] - k[650]*y[IDX_C2II] -
        k[739]*y[IDX_CHII] - k[740]*y[IDX_CHII] - k[753]*y[IDX_CH2II] -
        k[787]*y[IDX_CH3II] - k[835]*y[IDX_CH5II] - k[968]*y[IDX_H2COII] -
        k[987]*y[IDX_H2OII] - k[988]*y[IDX_H2OII] - k[1068]*y[IDX_H3II] -
        k[1117]*y[IDX_HCNII] - k[1151]*y[IDX_HCOII] - k[1174]*y[IDX_HNOII] -
        k[1324]*y[IDX_N2HII] - k[1372]*y[IDX_NHII] - k[1373]*y[IDX_NHII] -
        k[1392]*y[IDX_NH2II] - k[1393]*y[IDX_NH2II] - k[1484]*y[IDX_O2II] -
        k[1533]*y[IDX_OHII] - k[1534]*y[IDX_OHII] - k[1553]*y[IDX_H3SII] -
        k[1554]*y[IDX_O2HII] - k[1555]*y[IDX_SiOII] - k[1568]*y[IDX_SiH2II] -
        k[1573]*y[IDX_C2I] - k[1644]*y[IDX_CH2I] - k[1645]*y[IDX_CH2I] -
        k[1669]*y[IDX_CH3I] - k[1673]*y[IDX_CH4I] - k[1697]*y[IDX_CHI] -
        k[1698]*y[IDX_CHI] - k[1714]*y[IDX_CNI] - k[1735]*y[IDX_H2I] -
        k[1856]*y[IDX_NHI] - k[1857]*y[IDX_NHI] - k[1861]*y[IDX_NOI] -
        k[1862]*y[IDX_NOI] - k[1865]*y[IDX_O2I] - k[1948]*y[IDX_OHI] -
        k[1951]*y[IDX_HCOI] - k[1952]*y[IDX_HCOI] - k[1953]*y[IDX_HSI] -
        k[1954]*y[IDX_SO2I] - k[1955]*y[IDX_SOI] - k[2073] - k[2099]*y[IDX_CII]
        - k[2106]*y[IDX_CI] - k[2261];
    data[4397] = 0.0 - k[1948]*y[IDX_SI];
    data[4398] = 0.0 + k[1809]*y[IDX_CSI] + k[1814]*y[IDX_HCSI] + k[1817]*y[IDX_HSI] +
        k[1824]*y[IDX_NSI] + k[1829]*y[IDX_S2I] + k[1831]*y[IDX_SOI];
    data[4399] = 0.0 + k[1215]*y[IDX_CSI] + k[1221]*y[IDX_H2CSI] + k[1260]*y[IDX_NSI] +
        k[1267]*y[IDX_OCSI] + k[1269]*y[IDX_S2I] + k[1273]*y[IDX_SOI] +
        k[1288]*y[IDX_SiSI];
    data[4400] = 0.0 - k[1068]*y[IDX_SI];
    data[4401] = 0.0 - k[140]*y[IDX_SI];
    data[4402] = 0.0 + k[1496]*y[IDX_CSII] + k[1502]*y[IDX_HCSII] + k[1509]*y[IDX_NSII] +
        k[1883]*y[IDX_CSI] + k[1899]*y[IDX_HSI] + k[1907]*y[IDX_NSI] +
        k[1912]*y[IDX_OCSI] + k[1915]*y[IDX_S2I] + k[1917]*y[IDX_SOI];
    data[4403] = 0.0 + k[1592]*y[IDX_CSI] + k[1596]*y[IDX_HSI] + k[1607]*y[IDX_NSI] +
        k[1613]*y[IDX_S2I] + k[1616]*y[IDX_SOI] - k[2106]*y[IDX_SI];
    data[4404] = 0.0 - k[1151]*y[IDX_SI];
    data[4405] = 0.0 + k[1007]*y[IDX_HSII];
    data[4406] = 0.0 - k[1735]*y[IDX_SI];
    data[4407] = 0.0 + k[500]*y[IDX_CSII] + k[516]*y[IDX_H2SII] + k[535]*y[IDX_H3SII] +
        k[546]*y[IDX_HCSII] + k[554]*y[IDX_HSII] + k[555]*y[IDX_HS2II] +
        k[576]*y[IDX_NSII] + k[581]*y[IDX_OCSII] + k[583]*y[IDX_S2II] +
        k[583]*y[IDX_S2II] + k[584]*y[IDX_SOII] + k[585]*y[IDX_SO2II] +
        k[605]*y[IDX_SiSII] + k[2143]*y[IDX_SII];
    data[4408] = 0.0 + k[1758]*y[IDX_HSI] + k[1767]*y[IDX_NSI] + k[1777]*y[IDX_S2I] +
        k[1779]*y[IDX_SOI];
    data[4409] = 0.0 + k[466]*y[IDX_EM];
    data[4410] = 0.0 + k[560]*y[IDX_EM];
    data[4411] = 0.0 + k[552]*y[IDX_EM];
    data[4412] = 0.0 + k[1107]*y[IDX_HI];
    data[4413] = 0.0 + k[1691]*y[IDX_CHI] + k[1719]*y[IDX_COI] + k[1771]*y[IDX_HI] +
        k[1771]*y[IDX_HI] + k[1909]*y[IDX_OI] - k[1946]*y[IDX_OHI] + k[2064];
    data[4414] = 0.0 + k[903]*y[IDX_HII] + k[1059]*y[IDX_H3II] + k[1763]*y[IDX_HI];
    data[4415] = 0.0 - k[1931]*y[IDX_OHI];
    data[4416] = 0.0 + k[372] + k[1969];
    data[4417] = 0.0 - k[1943]*y[IDX_OHI];
    data[4418] = 0.0 + k[488]*y[IDX_EM];
    data[4419] = 0.0 + k[1774]*y[IDX_HI];
    data[4420] = 0.0 + k[1567]*y[IDX_O2I];
    data[4421] = 0.0 + k[989]*y[IDX_H2OII];
    data[4422] = 0.0 + k[1757]*y[IDX_HI] + k[1897]*y[IDX_OI] - k[1942]*y[IDX_OHI];
    data[4423] = 0.0 + k[603]*y[IDX_EM];
    data[4424] = 0.0 + k[1925]*y[IDX_OI];
    data[4425] = 0.0 - k[1930]*y[IDX_OHI];
    data[4426] = 0.0 + k[992]*y[IDX_H2OI];
    data[4427] = 0.0 + k[545]*y[IDX_EM];
    data[4428] = 0.0 + k[389] + k[1201]*y[IDX_HeII] + k[1467]*y[IDX_OII] + k[1992];
    data[4429] = 0.0 + k[1779]*y[IDX_HI] - k[1949]*y[IDX_OHI];
    data[4430] = 0.0 + k[1870]*y[IDX_OI];
    data[4431] = 0.0 + k[1493]*y[IDX_OI];
    data[4432] = 0.0 - k[1935]*y[IDX_OHI] - k[1936]*y[IDX_OHI];
    data[4433] = 0.0 - k[339]*y[IDX_OHI] + k[990]*y[IDX_H2OI];
    data[4434] = 0.0 - k[1547]*y[IDX_OHI];
    data[4435] = 0.0 - k[340]*y[IDX_OHI] + k[995]*y[IDX_H2OI];
    data[4436] = 0.0 - k[1545]*y[IDX_OHI];
    data[4437] = 0.0 + k[1498]*y[IDX_OI];
    data[4438] = 0.0 - k[1544]*y[IDX_OHI];
    data[4439] = 0.0 - k[1538]*y[IDX_OHI];
    data[4440] = 0.0 - k[342]*y[IDX_OHI] + k[1010]*y[IDX_H2OI];
    data[4441] = 0.0 + k[1899]*y[IDX_OI];
    data[4442] = 0.0 - k[171]*y[IDX_OHI] - k[933]*y[IDX_OHI];
    data[4443] = 0.0 + k[521]*y[IDX_EM];
    data[4444] = 0.0 - k[1950]*y[IDX_OHI];
    data[4445] = 0.0 - k[1927]*y[IDX_OHI] - k[1928]*y[IDX_OHI] - k[1929]*y[IDX_OHI];
    data[4446] = 0.0 - k[1541]*y[IDX_OHI];
    data[4447] = 0.0 + k[986]*y[IDX_H2OII];
    data[4448] = 0.0 - k[1549]*y[IDX_OHI];
    data[4449] = 0.0 - k[341]*y[IDX_OHI] + k[997]*y[IDX_H2OI] - k[1539]*y[IDX_OHI];
    data[4450] = 0.0 + k[1503]*y[IDX_OI];
    data[4451] = 0.0 + k[1744]*y[IDX_HI];
    data[4452] = 0.0 + k[1359]*y[IDX_H2OI] + k[1368]*y[IDX_O2I] - k[1371]*y[IDX_OHI];
    data[4453] = 0.0 + k[277]*y[IDX_OHII] + k[1400]*y[IDX_H2OII] + k[1835]*y[IDX_NOI] -
        k[1836]*y[IDX_OHI] - k[1837]*y[IDX_OHI] + k[1903]*y[IDX_OI];
    data[4454] = 0.0 - k[251]*y[IDX_OHI];
    data[4455] = 0.0 + k[1379]*y[IDX_H2OI] + k[1391]*y[IDX_O2I];
    data[4456] = 0.0 + k[330]*y[IDX_OHII] + k[977]*y[IDX_H2OII];
    data[4457] = 0.0 - k[319]*y[IDX_OHI] + k[1467]*y[IDX_CH3OHI] + k[1468]*y[IDX_CH4I] +
        k[1471]*y[IDX_H2COI] + k[1472]*y[IDX_H2SI] - k[1480]*y[IDX_OHI];
    data[4458] = 0.0 + k[749]*y[IDX_O2I];
    data[4459] = 0.0 + k[1414]*y[IDX_H2OI] - k[1546]*y[IDX_OHI];
    data[4460] = 0.0 + k[333]*y[IDX_OHII] + k[981]*y[IDX_H2OII] + k[1472]*y[IDX_OII] +
        k[1888]*y[IDX_OI] - k[1938]*y[IDX_OHI];
    data[4461] = 0.0 + k[514]*y[IDX_EM] + k[690]*y[IDX_CI] + k[758]*y[IDX_CH2I] +
        k[843]*y[IDX_CHI] + k[976]*y[IDX_C2I] + k[977]*y[IDX_C2HI] +
        k[978]*y[IDX_COI] + k[979]*y[IDX_H2COI] + k[980]*y[IDX_H2OI] +
        k[981]*y[IDX_H2SI] + k[983]*y[IDX_HCNI] + k[985]*y[IDX_HCOI] +
        k[986]*y[IDX_HNCI] + k[987]*y[IDX_SI] + k[989]*y[IDX_SO2I] +
        k[1400]*y[IDX_NH2I] + k[1425]*y[IDX_NH3I] - k[1540]*y[IDX_OHI];
    data[4462] = 0.0 + k[71]*y[IDX_CH2I] + k[92]*y[IDX_CHI] + k[277]*y[IDX_NH2I] +
        k[329]*y[IDX_C2I] + k[330]*y[IDX_C2HI] + k[331]*y[IDX_H2COI] +
        k[332]*y[IDX_H2OI] + k[333]*y[IDX_H2SI] + k[334]*y[IDX_HCOI] +
        k[335]*y[IDX_NH3I] + k[336]*y[IDX_NOI] + k[337]*y[IDX_O2I] +
        k[338]*y[IDX_SI] - k[1532]*y[IDX_OHI];
    data[4463] = 0.0 - k[786]*y[IDX_OHI];
    data[4464] = 0.0 + k[329]*y[IDX_OHII] + k[976]*y[IDX_H2OII];
    data[4465] = 0.0 + k[71]*y[IDX_OHII] + k[758]*y[IDX_H2OII] + k[1630]*y[IDX_NOI] +
        k[1636]*y[IDX_O2I] + k[1640]*y[IDX_OI] - k[1641]*y[IDX_OHI] -
        k[1642]*y[IDX_OHI] - k[1643]*y[IDX_OHI];
    data[4466] = 0.0 + k[1652]*y[IDX_H2OI] + k[1660]*y[IDX_O2I] - k[1666]*y[IDX_OHI] -
        k[1667]*y[IDX_OHI] - k[1668]*y[IDX_OHI];
    data[4467] = 0.0 + k[732]*y[IDX_O2I] - k[738]*y[IDX_OHI];
    data[4468] = 0.0 + k[1841]*y[IDX_H2OI] + k[1848]*y[IDX_NOI] + k[1850]*y[IDX_O2I] +
        k[1852]*y[IDX_OI] - k[1853]*y[IDX_OHI] - k[1854]*y[IDX_OHI] -
        k[1855]*y[IDX_OHI];
    data[4469] = 0.0 - k[1548]*y[IDX_OHI];
    data[4470] = 0.0 - k[1932]*y[IDX_OHI] - k[1933]*y[IDX_OHI];
    data[4471] = 0.0 + k[331]*y[IDX_OHII] + k[979]*y[IDX_H2OII] + k[1471]*y[IDX_OII] +
        k[1886]*y[IDX_OI] - k[1937]*y[IDX_OHI];
    data[4472] = 0.0 + k[1468]*y[IDX_OII] - k[1672]*y[IDX_OHI] + k[1879]*y[IDX_OI];
    data[4473] = 0.0 + k[334]*y[IDX_OHII] + k[985]*y[IDX_H2OII] + k[1784]*y[IDX_O2I] +
        k[1893]*y[IDX_OI] - k[1941]*y[IDX_OHI];
    data[4474] = 0.0 + k[983]*y[IDX_H2OII] + k[1889]*y[IDX_OI] - k[1939]*y[IDX_OHI] -
        k[1940]*y[IDX_OHI];
    data[4475] = 0.0 + k[336]*y[IDX_OHII] + k[1630]*y[IDX_CH2I] + k[1765]*y[IDX_HI] +
        k[1835]*y[IDX_NH2I] + k[1848]*y[IDX_NHI] - k[1945]*y[IDX_OHI];
    data[4476] = 0.0 + k[92]*y[IDX_OHII] + k[843]*y[IDX_H2OII] + k[1689]*y[IDX_O2I] +
        k[1691]*y[IDX_O2HI] + k[1694]*y[IDX_OI] - k[1696]*y[IDX_OHI];
    data[4477] = 0.0 + k[335]*y[IDX_OHII] + k[1425]*y[IDX_H2OII] + k[1904]*y[IDX_OI] -
        k[1944]*y[IDX_OHI];
    data[4478] = 0.0 + k[530]*y[IDX_EM] + k[531]*y[IDX_EM];
    data[4479] = 0.0 + k[337]*y[IDX_OHII] + k[732]*y[IDX_CHII] + k[749]*y[IDX_CH2II] +
        k[1368]*y[IDX_NHII] + k[1391]*y[IDX_NH2II] + k[1567]*y[IDX_SiH2II] +
        k[1636]*y[IDX_CH2I] + k[1660]*y[IDX_CH3I] + k[1689]*y[IDX_CHI] +
        k[1732]*y[IDX_H2I] + k[1732]*y[IDX_H2I] + k[1768]*y[IDX_HI] +
        k[1784]*y[IDX_HCOI] + k[1850]*y[IDX_NHI];
    data[4480] = 0.0 - k[637]*y[IDX_OHI];
    data[4481] = 0.0 + k[338]*y[IDX_OHII] + k[987]*y[IDX_H2OII] - k[1948]*y[IDX_OHI];
    data[4482] = 0.0 - k[7]*y[IDX_H2I] - k[13]*y[IDX_HI] - k[138]*y[IDX_HII] -
        k[171]*y[IDX_H2II] - k[251]*y[IDX_NII] - k[319]*y[IDX_OII] -
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
    data[4483] = 0.0 - k[1827]*y[IDX_OHI] - k[1828]*y[IDX_OHI];
    data[4484] = 0.0 + k[1201]*y[IDX_CH3OHI] + k[1223]*y[IDX_H2OI] - k[1268]*y[IDX_OHI];
    data[4485] = 0.0 + k[1059]*y[IDX_NO2I] - k[1066]*y[IDX_OHI];
    data[4486] = 0.0 - k[138]*y[IDX_OHI] + k[903]*y[IDX_NO2I];
    data[4487] = 0.0 + k[1493]*y[IDX_CH4II] + k[1498]*y[IDX_H2SII] + k[1503]*y[IDX_HSII]
        + k[1640]*y[IDX_CH2I] + k[1694]*y[IDX_CHI] + k[1733]*y[IDX_H2I] +
        k[1852]*y[IDX_NHI] + k[1870]*y[IDX_C2H4I] + k[1879]*y[IDX_CH4I] +
        k[1886]*y[IDX_H2COI] + k[1887]*y[IDX_H2OI] + k[1887]*y[IDX_H2OI] +
        k[1888]*y[IDX_H2SI] + k[1889]*y[IDX_HCNI] + k[1893]*y[IDX_HCOI] +
        k[1897]*y[IDX_HNOI] + k[1899]*y[IDX_HSI] + k[1903]*y[IDX_NH2I] +
        k[1904]*y[IDX_NH3I] + k[1909]*y[IDX_O2HI] - k[1914]*y[IDX_OHI] +
        k[1925]*y[IDX_SiH4I] + k[2124]*y[IDX_HI];
    data[4488] = 0.0 + k[690]*y[IDX_H2OII] - k[1611]*y[IDX_OHI] - k[1612]*y[IDX_OHI];
    data[4489] = 0.0 - k[1542]*y[IDX_OHI] - k[1543]*y[IDX_OHI];
    data[4490] = 0.0 + k[4]*y[IDX_H2I] + k[11]*y[IDX_HI] + k[332]*y[IDX_OHII] + k[401] +
        k[980]*y[IDX_H2OII] + k[990]*y[IDX_C2II] + k[992]*y[IDX_C2NII] +
        k[995]*y[IDX_CNII] + k[997]*y[IDX_COII] + k[1010]*y[IDX_N2II] +
        k[1223]*y[IDX_HeII] + k[1359]*y[IDX_NHII] + k[1379]*y[IDX_NH2II] +
        k[1414]*y[IDX_NH3II] + k[1652]*y[IDX_CH3I] + k[1748]*y[IDX_HI] +
        k[1841]*y[IDX_NHI] + k[1887]*y[IDX_OI] + k[1887]*y[IDX_OI] + k[2018];
    data[4491] = 0.0 + k[978]*y[IDX_H2OII] + k[1719]*y[IDX_O2HI] + k[1745]*y[IDX_HI] -
        k[1934]*y[IDX_OHI];
    data[4492] = 0.0 + k[4]*y[IDX_H2OI] - k[7]*y[IDX_OHI] + k[1732]*y[IDX_O2I] +
        k[1732]*y[IDX_O2I] + k[1733]*y[IDX_OI] - k[1734]*y[IDX_OHI];
    data[4493] = 0.0 + k[466]*y[IDX_C2H5OH2II] + k[488]*y[IDX_CH3OH2II] +
        k[514]*y[IDX_H2OII] + k[521]*y[IDX_H3COII] + k[530]*y[IDX_H3OII] +
        k[531]*y[IDX_H3OII] + k[545]*y[IDX_HCO2II] + k[552]*y[IDX_HOCSII] +
        k[560]*y[IDX_HSO2II] + k[603]*y[IDX_SiOHII];
    data[4494] = 0.0 + k[11]*y[IDX_H2OI] - k[13]*y[IDX_OHI] + k[1107]*y[IDX_SO2II] +
        k[1744]*y[IDX_CO2I] + k[1745]*y[IDX_COI] + k[1748]*y[IDX_H2OI] +
        k[1757]*y[IDX_HNOI] + k[1763]*y[IDX_NO2I] + k[1765]*y[IDX_NOI] +
        k[1768]*y[IDX_O2I] + k[1771]*y[IDX_O2HI] + k[1771]*y[IDX_O2HI] +
        k[1774]*y[IDX_OCNI] - k[1776]*y[IDX_OHI] + k[1779]*y[IDX_SOI] +
        k[2124]*y[IDX_OI] - k[2125]*y[IDX_OHI];
    data[4495] = 0.0 - k[1800]*y[IDX_NI];
    data[4496] = 0.0 - k[1797]*y[IDX_NI];
    data[4497] = 0.0 - k[1799]*y[IDX_NI];
    data[4498] = 0.0 - k[1810]*y[IDX_NI];
    data[4499] = 0.0 + k[470]*y[IDX_EM];
    data[4500] = 0.0 - k[1798]*y[IDX_NI];
    data[4501] = 0.0 + k[576]*y[IDX_EM];
    data[4502] = 0.0 - k[1826]*y[IDX_NI];
    data[4503] = 0.0 - k[1820]*y[IDX_NI] - k[1821]*y[IDX_NI] - k[1822]*y[IDX_NI];
    data[4504] = 0.0 - k[1793]*y[IDX_NI] - k[1794]*y[IDX_NI];
    data[4505] = 0.0 - k[1342]*y[IDX_NI];
    data[4506] = 0.0 + k[375] - k[1796]*y[IDX_NI] + k[1972];
    data[4507] = 0.0 + k[1305]*y[IDX_NII] - k[1818]*y[IDX_NI];
    data[4508] = 0.0 - k[1832]*y[IDX_NI];
    data[4509] = 0.0 - k[1814]*y[IDX_NI];
    data[4510] = 0.0 - k[1829]*y[IDX_NI];
    data[4511] = 0.0 + k[434] + k[632]*y[IDX_CII] + k[1259]*y[IDX_HeII] +
        k[1606]*y[IDX_CI] + k[1766]*y[IDX_HI] - k[1824]*y[IDX_NI] +
        k[1908]*y[IDX_OI] + k[2059];
    data[4512] = 0.0 - k[1815]*y[IDX_NI];
    data[4513] = 0.0 - k[1343]*y[IDX_NI] - k[1344]*y[IDX_NI];
    data[4514] = 0.0 - k[1791]*y[IDX_NI];
    data[4515] = 0.0 + k[468]*y[IDX_EM];
    data[4516] = 0.0 - k[1830]*y[IDX_NI] - k[1831]*y[IDX_NI];
    data[4517] = 0.0 - k[1792]*y[IDX_NI];
    data[4518] = 0.0 - k[1809]*y[IDX_NI];
    data[4519] = 0.0 + k[244]*y[IDX_NII];
    data[4520] = 0.0 - k[1341]*y[IDX_NI];
    data[4521] = 0.0 - k[1325]*y[IDX_NI] + k[1444]*y[IDX_NHI];
    data[4522] = 0.0 + k[498]*y[IDX_EM] - k[1332]*y[IDX_NI];
    data[4523] = 0.0 + k[566]*y[IDX_EM];
    data[4524] = 0.0 + k[250]*y[IDX_NII] + k[1313]*y[IDX_NII];
    data[4525] = 0.0 - k[1335]*y[IDX_NI];
    data[4526] = 0.0 - k[259]*y[IDX_NI] + k[564]*y[IDX_EM] + k[564]*y[IDX_EM] +
        k[1505]*y[IDX_OI];
    data[4527] = 0.0 - k[1816]*y[IDX_NI] - k[1817]*y[IDX_NI];
    data[4528] = 0.0 - k[928]*y[IDX_NI];
    data[4529] = 0.0 + k[1958]*y[IDX_NOI];
    data[4530] = 0.0 + k[1243]*y[IDX_HeII] + k[1362]*y[IDX_NHII];
    data[4531] = 0.0 + k[1448]*y[IDX_NHI];
    data[4532] = 0.0 - k[1336]*y[IDX_NI];
    data[4533] = 0.0 + k[1350]*y[IDX_NHII] - k[1808]*y[IDX_NI];
    data[4534] = 0.0 + k[567]*y[IDX_EM] + k[699]*y[IDX_CI] + k[766]*y[IDX_CH2I] +
        k[854]*y[IDX_CHI] + k[954]*y[IDX_H2I] - k[1337]*y[IDX_NI] +
        k[1345]*y[IDX_C2I] + k[1348]*y[IDX_C2HI] + k[1349]*y[IDX_CNI] +
        k[1350]*y[IDX_CO2I] + k[1353]*y[IDX_COI] + k[1354]*y[IDX_H2COI] +
        k[1356]*y[IDX_H2OI] + k[1360]*y[IDX_HCNI] + k[1361]*y[IDX_HCOI] +
        k[1362]*y[IDX_HNCI] + k[1363]*y[IDX_N2I] + k[1364]*y[IDX_NH2I] +
        k[1365]*y[IDX_NH3I] + k[1366]*y[IDX_NHI] + k[1369]*y[IDX_O2I] +
        k[1370]*y[IDX_OI] + k[1371]*y[IDX_OHI] + k[1372]*y[IDX_SI] + k[2048];
    data[4535] = 0.0 + k[245]*y[IDX_NII] + k[1364]*y[IDX_NHII];
    data[4536] = 0.0 - k[1339]*y[IDX_NI];
    data[4537] = 0.0 + k[87]*y[IDX_CHI] + k[233]*y[IDX_C2I] + k[234]*y[IDX_C2HI] +
        k[235]*y[IDX_CH2I] + k[236]*y[IDX_CH4I] + k[237]*y[IDX_CNI] +
        k[238]*y[IDX_COI] + k[239]*y[IDX_H2COI] + k[240]*y[IDX_H2OI] +
        k[241]*y[IDX_H2SI] + k[242]*y[IDX_HCNI] + k[243]*y[IDX_HCOI] +
        k[244]*y[IDX_MgI] + k[245]*y[IDX_NH2I] + k[246]*y[IDX_NH3I] +
        k[247]*y[IDX_NHI] + k[248]*y[IDX_NOI] + k[249]*y[IDX_O2I] +
        k[250]*y[IDX_OCSI] + k[251]*y[IDX_OHI] + k[1293]*y[IDX_CH4I] +
        k[1305]*y[IDX_NCCNI] + k[1313]*y[IDX_OCSI] - k[2127]*y[IDX_NI] +
        k[2141]*y[IDX_EM];
    data[4538] = 0.0 + k[568]*y[IDX_EM] - k[1338]*y[IDX_NI] + k[1455]*y[IDX_NHI];
    data[4539] = 0.0 - k[1326]*y[IDX_NI] - k[1327]*y[IDX_NI];
    data[4540] = 0.0 + k[234]*y[IDX_NII] + k[1348]*y[IDX_NHII] - k[1795]*y[IDX_NI];
    data[4541] = 0.0 + k[1474]*y[IDX_HCNI] + k[1477]*y[IDX_N2I];
    data[4542] = 0.0 + k[575]*y[IDX_EM];
    data[4543] = 0.0 + k[1449]*y[IDX_NHI];
    data[4544] = 0.0 - k[1331]*y[IDX_NI];
    data[4545] = 0.0 + k[1456]*y[IDX_NHI];
    data[4546] = 0.0 + k[241]*y[IDX_NII];
    data[4547] = 0.0 - k[1333]*y[IDX_NI] - k[1334]*y[IDX_NI] + k[1450]*y[IDX_NHI];
    data[4548] = 0.0 - k[1340]*y[IDX_NI];
    data[4549] = 0.0 + k[421] + k[421] + k[1250]*y[IDX_HeII] + k[1363]*y[IDX_NHII] +
        k[1477]*y[IDX_OII] + k[1597]*y[IDX_CI] + k[1681]*y[IDX_CHI] +
        k[1901]*y[IDX_OI] + k[2045] + k[2045];
    data[4550] = 0.0 - k[1328]*y[IDX_NI] - k[1329]*y[IDX_NI] - k[1330]*y[IDX_NI];
    data[4551] = 0.0 + k[233]*y[IDX_NII] + k[1345]*y[IDX_NHII] - k[1790]*y[IDX_NI];
    data[4552] = 0.0 + k[235]*y[IDX_NII] + k[766]*y[IDX_NHII] + k[1629]*y[IDX_NOI] -
        k[1801]*y[IDX_NI] - k[1802]*y[IDX_NI] - k[1803]*y[IDX_NI];
    data[4553] = 0.0 - k[1804]*y[IDX_NI] - k[1805]*y[IDX_NI] - k[1806]*y[IDX_NI];
    data[4554] = 0.0 - k[728]*y[IDX_NI];
    data[4555] = 0.0 + k[247]*y[IDX_NII] + k[429] + k[1366]*y[IDX_NHII] +
        k[1444]*y[IDX_C2II] + k[1448]*y[IDX_COII] + k[1449]*y[IDX_H2COII] +
        k[1450]*y[IDX_H2OII] + k[1455]*y[IDX_NH2II] + k[1456]*y[IDX_NH3II] +
        k[1603]*y[IDX_CI] + k[1762]*y[IDX_HI] - k[1819]*y[IDX_NI] +
        k[1840]*y[IDX_CNI] + k[1845]*y[IDX_NHI] + k[1845]*y[IDX_NHI] +
        k[1852]*y[IDX_OI] + k[1853]*y[IDX_OHI] + k[1856]*y[IDX_SI] + k[2054];
    data[4556] = 0.0 + k[237]*y[IDX_NII] + k[392] + k[1208]*y[IDX_HeII] +
        k[1349]*y[IDX_NHII] + k[1590]*y[IDX_CI] + k[1711]*y[IDX_NOI] -
        k[1807]*y[IDX_NI] + k[1840]*y[IDX_NHI] + k[1880]*y[IDX_OI] + k[2001];
    data[4557] = 0.0 + k[239]*y[IDX_NII] + k[1354]*y[IDX_NHII];
    data[4558] = 0.0 + k[236]*y[IDX_NII] + k[1293]*y[IDX_NII];
    data[4559] = 0.0 + k[243]*y[IDX_NII] + k[1361]*y[IDX_NHII] - k[1811]*y[IDX_NI] -
        k[1812]*y[IDX_NI] - k[1813]*y[IDX_NI];
    data[4560] = 0.0 + k[242]*y[IDX_NII] + k[1233]*y[IDX_HeII] + k[1234]*y[IDX_HeII] +
        k[1360]*y[IDX_NHII] + k[1474]*y[IDX_OII];
    data[4561] = 0.0 + k[248]*y[IDX_NII] + k[433] + k[1257]*y[IDX_HeII] +
        k[1605]*y[IDX_CI] + k[1629]*y[IDX_CH2I] + k[1685]*y[IDX_CHI] +
        k[1711]*y[IDX_CNI] + k[1765]*y[IDX_HI] - k[1823]*y[IDX_NI] +
        k[1862]*y[IDX_SI] + k[1906]*y[IDX_OI] + k[1958]*y[IDX_SiI] + k[2058];
    data[4562] = 0.0 + k[87]*y[IDX_NII] + k[854]*y[IDX_NHII] + k[1681]*y[IDX_N2I] -
        k[1682]*y[IDX_NI] - k[1683]*y[IDX_NI] + k[1685]*y[IDX_NOI];
    data[4563] = 0.0 + k[246]*y[IDX_NII] + k[1365]*y[IDX_NHII];
    data[4564] = 0.0 + k[249]*y[IDX_NII] + k[1369]*y[IDX_NHII] - k[1825]*y[IDX_NI];
    data[4565] = 0.0 + k[632]*y[IDX_NSI] - k[2097]*y[IDX_NI];
    data[4566] = 0.0 + k[1372]*y[IDX_NHII] + k[1856]*y[IDX_NHI] + k[1862]*y[IDX_NOI];
    data[4567] = 0.0 + k[251]*y[IDX_NII] + k[1371]*y[IDX_NHII] - k[1827]*y[IDX_NI] -
        k[1828]*y[IDX_NI] + k[1853]*y[IDX_NHI];
    data[4568] = 0.0 - k[259]*y[IDX_N2II] - k[364] - k[422] - k[728]*y[IDX_CHII] -
        k[928]*y[IDX_H2II] - k[1325]*y[IDX_C2II] - k[1326]*y[IDX_C2HII] -
        k[1327]*y[IDX_C2HII] - k[1328]*y[IDX_C2H2II] - k[1329]*y[IDX_C2H2II] -
        k[1330]*y[IDX_C2H2II] - k[1331]*y[IDX_CH2II] - k[1332]*y[IDX_CNII] -
        k[1333]*y[IDX_H2OII] - k[1334]*y[IDX_H2OII] - k[1335]*y[IDX_H2SII] -
        k[1336]*y[IDX_HSII] - k[1337]*y[IDX_NHII] - k[1338]*y[IDX_NH2II] -
        k[1339]*y[IDX_O2II] - k[1340]*y[IDX_OHII] - k[1341]*y[IDX_SOII] -
        k[1342]*y[IDX_SiCII] - k[1343]*y[IDX_SiOII] - k[1344]*y[IDX_SiOII] -
        k[1682]*y[IDX_CHI] - k[1683]*y[IDX_CHI] - k[1728]*y[IDX_H2I] -
        k[1790]*y[IDX_C2I] - k[1791]*y[IDX_C2H3I] - k[1792]*y[IDX_C2H4I] -
        k[1793]*y[IDX_C2H5I] - k[1794]*y[IDX_C2H5I] - k[1795]*y[IDX_C2HI] -
        k[1796]*y[IDX_C2NI] - k[1797]*y[IDX_C3H2I] - k[1798]*y[IDX_C3NI] -
        k[1799]*y[IDX_C4HI] - k[1800]*y[IDX_C4NI] - k[1801]*y[IDX_CH2I] -
        k[1802]*y[IDX_CH2I] - k[1803]*y[IDX_CH2I] - k[1804]*y[IDX_CH3I] -
        k[1805]*y[IDX_CH3I] - k[1806]*y[IDX_CH3I] - k[1807]*y[IDX_CNI] -
        k[1808]*y[IDX_CO2I] - k[1809]*y[IDX_CSI] - k[1810]*y[IDX_H2CNI] -
        k[1811]*y[IDX_HCOI] - k[1812]*y[IDX_HCOI] - k[1813]*y[IDX_HCOI] -
        k[1814]*y[IDX_HCSI] - k[1815]*y[IDX_HNOI] - k[1816]*y[IDX_HSI] -
        k[1817]*y[IDX_HSI] - k[1818]*y[IDX_NCCNI] - k[1819]*y[IDX_NHI] -
        k[1820]*y[IDX_NO2I] - k[1821]*y[IDX_NO2I] - k[1822]*y[IDX_NO2I] -
        k[1823]*y[IDX_NOI] - k[1824]*y[IDX_NSI] - k[1825]*y[IDX_O2I] -
        k[1826]*y[IDX_O2HI] - k[1827]*y[IDX_OHI] - k[1828]*y[IDX_OHI] -
        k[1829]*y[IDX_S2I] - k[1830]*y[IDX_SOI] - k[1831]*y[IDX_SOI] -
        k[1832]*y[IDX_SiCI] - k[2097]*y[IDX_CII] - k[2102]*y[IDX_CI] -
        k[2127]*y[IDX_NII] - k[2251];
    data[4569] = 0.0 + k[1208]*y[IDX_CNI] + k[1233]*y[IDX_HCNI] + k[1234]*y[IDX_HCNI] +
        k[1243]*y[IDX_HNCI] + k[1250]*y[IDX_N2I] + k[1257]*y[IDX_NOI] +
        k[1259]*y[IDX_NSI];
    data[4570] = 0.0 + k[1370]*y[IDX_NHII] + k[1505]*y[IDX_N2II] + k[1852]*y[IDX_NHI] +
        k[1880]*y[IDX_CNI] + k[1901]*y[IDX_N2I] + k[1906]*y[IDX_NOI] +
        k[1908]*y[IDX_NSI];
    data[4571] = 0.0 + k[699]*y[IDX_NHII] + k[1590]*y[IDX_CNI] + k[1597]*y[IDX_N2I] +
        k[1603]*y[IDX_NHI] + k[1605]*y[IDX_NOI] + k[1606]*y[IDX_NSI] -
        k[2102]*y[IDX_NI];
    data[4572] = 0.0 + k[240]*y[IDX_NII] + k[1356]*y[IDX_NHII];
    data[4573] = 0.0 + k[238]*y[IDX_NII] + k[1353]*y[IDX_NHII];
    data[4574] = 0.0 + k[954]*y[IDX_NHII] - k[1728]*y[IDX_NI];
    data[4575] = 0.0 + k[468]*y[IDX_C2NII] + k[470]*y[IDX_C2N2II] + k[498]*y[IDX_CNII] +
        k[564]*y[IDX_N2II] + k[564]*y[IDX_N2II] + k[566]*y[IDX_N2HII] +
        k[567]*y[IDX_NHII] + k[568]*y[IDX_NH2II] + k[575]*y[IDX_NOII] +
        k[576]*y[IDX_NSII] + k[2141]*y[IDX_NII];
    data[4576] = 0.0 + k[1762]*y[IDX_NHI] + k[1765]*y[IDX_NOI] + k[1766]*y[IDX_NSI];
    data[4577] = 0.0 - k[1224]*y[IDX_HeII] - k[1225]*y[IDX_HeII];
    data[4578] = 0.0 - k[1228]*y[IDX_HeII];
    data[4579] = 0.0 - k[1191]*y[IDX_HeII];
    data[4580] = 0.0 - k[1275]*y[IDX_HeII];
    data[4581] = 0.0 - k[1194]*y[IDX_HeII] - k[1195]*y[IDX_HeII];
    data[4582] = 0.0 - k[1190]*y[IDX_HeII];
    data[4583] = 0.0 - k[1238]*y[IDX_HeII];
    data[4584] = 0.0 - k[1241]*y[IDX_HeII];
    data[4585] = 0.0 - k[1247]*y[IDX_HeII] - k[1248]*y[IDX_HeII];
    data[4586] = 0.0 - k[1197]*y[IDX_HeII];
    data[4587] = 0.0 - k[1274]*y[IDX_HeII];
    data[4588] = 0.0 - k[1219]*y[IDX_HeII] - k[1220]*y[IDX_HeII] - k[1221]*y[IDX_HeII];
    data[4589] = 0.0 - k[1229]*y[IDX_HeII] - k[1230]*y[IDX_HeII];
    data[4590] = 0.0 - k[1189]*y[IDX_HeII];
    data[4591] = 0.0 - k[1251]*y[IDX_HeII];
    data[4592] = 0.0 - k[1276]*y[IDX_HeII] - k[1277]*y[IDX_HeII];
    data[4593] = 0.0 - k[1239]*y[IDX_HeII] - k[1240]*y[IDX_HeII];
    data[4594] = 0.0 - k[1278]*y[IDX_HeII] - k[1279]*y[IDX_HeII];
    data[4595] = 0.0 - k[1287]*y[IDX_HeII] - k[1288]*y[IDX_HeII];
    data[4596] = 0.0 - k[1262]*y[IDX_HeII] - k[1263]*y[IDX_HeII];
    data[4597] = 0.0 - k[1269]*y[IDX_HeII];
    data[4598] = 0.0 - k[1280]*y[IDX_HeII] - k[1281]*y[IDX_HeII];
    data[4599] = 0.0 - k[1198]*y[IDX_HeII] - k[1199]*y[IDX_HeII];
    data[4600] = 0.0 - k[1259]*y[IDX_HeII] - k[1260]*y[IDX_HeII];
    data[4601] = 0.0 - k[218]*y[IDX_HeII] - k[1270]*y[IDX_HeII] - k[1271]*y[IDX_HeII];
    data[4602] = 0.0 - k[1245]*y[IDX_HeII] - k[1246]*y[IDX_HeII];
    data[4603] = 0.0 - k[1284]*y[IDX_HeII];
    data[4604] = 0.0 - k[1282]*y[IDX_HeII] - k[1283]*y[IDX_HeII];
    data[4605] = 0.0 - k[1181]*y[IDX_HeII] - k[1182]*y[IDX_HeII];
    data[4606] = 0.0 - k[1285]*y[IDX_HeII] - k[1286]*y[IDX_HeII];
    data[4607] = 0.0 - k[1200]*y[IDX_HeII] - k[1201]*y[IDX_HeII];
    data[4608] = 0.0 - k[1272]*y[IDX_HeII] - k[1273]*y[IDX_HeII];
    data[4609] = 0.0 - k[1183]*y[IDX_HeII] - k[1184]*y[IDX_HeII] - k[1185]*y[IDX_HeII];
    data[4610] = 0.0 - k[1214]*y[IDX_HeII] - k[1215]*y[IDX_HeII];
    data[4611] = 0.0 - k[1264]*y[IDX_HeII] - k[1265]*y[IDX_HeII] - k[1266]*y[IDX_HeII] -
        k[1267]*y[IDX_HeII];
    data[4612] = 0.0 - k[1249]*y[IDX_HeII];
    data[4613] = 0.0 - k[219]*y[IDX_HeII];
    data[4614] = 0.0 - k[208]*y[IDX_HeII] - k[1178]*y[IDX_HeII] - k[1179]*y[IDX_HeII] -
        k[1180]*y[IDX_HeII];
    data[4615] = 0.0 - k[1242]*y[IDX_HeII] - k[1243]*y[IDX_HeII] - k[1244]*y[IDX_HeII];
    data[4616] = 0.0 - k[1209]*y[IDX_HeII] - k[1210]*y[IDX_HeII] - k[1211]*y[IDX_HeII] -
        k[1212]*y[IDX_HeII];
    data[4617] = 0.0 - k[1252]*y[IDX_HeII] - k[1253]*y[IDX_HeII];
    data[4618] = 0.0 - k[1186]*y[IDX_HeII] - k[1187]*y[IDX_HeII] - k[1188]*y[IDX_HeII];
    data[4619] = 0.0 - k[214]*y[IDX_HeII] - k[1226]*y[IDX_HeII] - k[1227]*y[IDX_HeII];
    data[4620] = 0.0 - k[215]*y[IDX_HeII] - k[1250]*y[IDX_HeII];
    data[4621] = 0.0 - k[207]*y[IDX_HeII] - k[1177]*y[IDX_HeII];
    data[4622] = 0.0 - k[1192]*y[IDX_HeII] - k[1193]*y[IDX_HeII];
    data[4623] = 0.0 - k[1196]*y[IDX_HeII];
    data[4624] = 0.0 - k[1256]*y[IDX_HeII];
    data[4625] = 0.0 - k[1207]*y[IDX_HeII] - k[1208]*y[IDX_HeII];
    data[4626] = 0.0 - k[212]*y[IDX_HeII] - k[1216]*y[IDX_HeII] - k[1217]*y[IDX_HeII] -
        k[1218]*y[IDX_HeII];
    data[4627] = 0.0 - k[210]*y[IDX_HeII] - k[1202]*y[IDX_HeII] - k[1203]*y[IDX_HeII] -
        k[1204]*y[IDX_HeII] - k[1205]*y[IDX_HeII];
    data[4628] = 0.0 - k[1235]*y[IDX_HeII] - k[1236]*y[IDX_HeII] - k[1237]*y[IDX_HeII];
    data[4629] = 0.0 - k[1231]*y[IDX_HeII] - k[1232]*y[IDX_HeII] - k[1233]*y[IDX_HeII] -
        k[1234]*y[IDX_HeII];
    data[4630] = 0.0 - k[1257]*y[IDX_HeII] - k[1258]*y[IDX_HeII];
    data[4631] = 0.0 - k[211]*y[IDX_HeII] - k[1206]*y[IDX_HeII];
    data[4632] = 0.0 - k[216]*y[IDX_HeII] - k[1254]*y[IDX_HeII] - k[1255]*y[IDX_HeII];
    data[4633] = 0.0 - k[217]*y[IDX_HeII] - k[1261]*y[IDX_HeII];
    data[4634] = 0.0 - k[1268]*y[IDX_HeII];
    data[4635] = 0.0 - k[172]*y[IDX_H2I] - k[195]*y[IDX_HI] - k[207]*y[IDX_C2I] -
        k[208]*y[IDX_C2H2I] - k[209]*y[IDX_CI] - k[210]*y[IDX_CH4I] -
        k[211]*y[IDX_CHI] - k[212]*y[IDX_H2COI] - k[213]*y[IDX_H2OI] -
        k[214]*y[IDX_H2SI] - k[215]*y[IDX_N2I] - k[216]*y[IDX_NH3I] -
        k[217]*y[IDX_O2I] - k[218]*y[IDX_SO2I] - k[219]*y[IDX_SiI] -
        k[950]*y[IDX_H2I] - k[1177]*y[IDX_C2I] - k[1178]*y[IDX_C2H2I] -
        k[1179]*y[IDX_C2H2I] - k[1180]*y[IDX_C2H2I] - k[1181]*y[IDX_C2H3I] -
        k[1182]*y[IDX_C2H3I] - k[1183]*y[IDX_C2H4I] - k[1184]*y[IDX_C2H4I] -
        k[1185]*y[IDX_C2H4I] - k[1186]*y[IDX_C2HI] - k[1187]*y[IDX_C2HI] -
        k[1188]*y[IDX_C2HI] - k[1189]*y[IDX_C2NI] - k[1190]*y[IDX_C3NI] -
        k[1191]*y[IDX_C4HI] - k[1192]*y[IDX_CH2I] - k[1193]*y[IDX_CH2I] -
        k[1194]*y[IDX_CH2COI] - k[1195]*y[IDX_CH2COI] - k[1196]*y[IDX_CH3I] -
        k[1197]*y[IDX_CH3CCHI] - k[1198]*y[IDX_CH3CNI] - k[1199]*y[IDX_CH3CNI] -
        k[1200]*y[IDX_CH3OHI] - k[1201]*y[IDX_CH3OHI] - k[1202]*y[IDX_CH4I] -
        k[1203]*y[IDX_CH4I] - k[1204]*y[IDX_CH4I] - k[1205]*y[IDX_CH4I] -
        k[1206]*y[IDX_CHI] - k[1207]*y[IDX_CNI] - k[1208]*y[IDX_CNI] -
        k[1209]*y[IDX_CO2I] - k[1210]*y[IDX_CO2I] - k[1211]*y[IDX_CO2I] -
        k[1212]*y[IDX_CO2I] - k[1213]*y[IDX_COI] - k[1214]*y[IDX_CSI] -
        k[1215]*y[IDX_CSI] - k[1216]*y[IDX_H2COI] - k[1217]*y[IDX_H2COI] -
        k[1218]*y[IDX_H2COI] - k[1219]*y[IDX_H2CSI] - k[1220]*y[IDX_H2CSI] -
        k[1221]*y[IDX_H2CSI] - k[1222]*y[IDX_H2OI] - k[1223]*y[IDX_H2OI] -
        k[1224]*y[IDX_H2S2I] - k[1225]*y[IDX_H2S2I] - k[1226]*y[IDX_H2SI] -
        k[1227]*y[IDX_H2SI] - k[1228]*y[IDX_H2SiOI] - k[1229]*y[IDX_HC3NI] -
        k[1230]*y[IDX_HC3NI] - k[1231]*y[IDX_HCNI] - k[1232]*y[IDX_HCNI] -
        k[1233]*y[IDX_HCNI] - k[1234]*y[IDX_HCNI] - k[1235]*y[IDX_HCOI] -
        k[1236]*y[IDX_HCOI] - k[1237]*y[IDX_HCOI] - k[1238]*y[IDX_HCOOCH3I] -
        k[1239]*y[IDX_HCSI] - k[1240]*y[IDX_HCSI] - k[1241]*y[IDX_HClI] -
        k[1242]*y[IDX_HNCI] - k[1243]*y[IDX_HNCI] - k[1244]*y[IDX_HNCI] -
        k[1245]*y[IDX_HNOI] - k[1246]*y[IDX_HNOI] - k[1247]*y[IDX_HS2I] -
        k[1248]*y[IDX_HS2I] - k[1249]*y[IDX_HSI] - k[1250]*y[IDX_N2I] -
        k[1251]*y[IDX_NCCNI] - k[1252]*y[IDX_NH2I] - k[1253]*y[IDX_NH2I] -
        k[1254]*y[IDX_NH3I] - k[1255]*y[IDX_NH3I] - k[1256]*y[IDX_NHI] -
        k[1257]*y[IDX_NOI] - k[1258]*y[IDX_NOI] - k[1259]*y[IDX_NSI] -
        k[1260]*y[IDX_NSI] - k[1261]*y[IDX_O2I] - k[1262]*y[IDX_OCNI] -
        k[1263]*y[IDX_OCNI] - k[1264]*y[IDX_OCSI] - k[1265]*y[IDX_OCSI] -
        k[1266]*y[IDX_OCSI] - k[1267]*y[IDX_OCSI] - k[1268]*y[IDX_OHI] -
        k[1269]*y[IDX_S2I] - k[1270]*y[IDX_SO2I] - k[1271]*y[IDX_SO2I] -
        k[1272]*y[IDX_SOI] - k[1273]*y[IDX_SOI] - k[1274]*y[IDX_SiC2I] -
        k[1275]*y[IDX_SiC3I] - k[1276]*y[IDX_SiCI] - k[1277]*y[IDX_SiCI] -
        k[1278]*y[IDX_SiH2I] - k[1279]*y[IDX_SiH2I] - k[1280]*y[IDX_SiH3I] -
        k[1281]*y[IDX_SiH3I] - k[1282]*y[IDX_SiH4I] - k[1283]*y[IDX_SiH4I] -
        k[1284]*y[IDX_SiHI] - k[1285]*y[IDX_SiOI] - k[1286]*y[IDX_SiOI] -
        k[1287]*y[IDX_SiSI] - k[1288]*y[IDX_SiSI] - k[2139]*y[IDX_EM];
    data[4636] = 0.0 + k[363] + k[419];
    data[4637] = 0.0 - k[209]*y[IDX_HeII];
    data[4638] = 0.0 - k[213]*y[IDX_HeII] - k[1222]*y[IDX_HeII] - k[1223]*y[IDX_HeII];
    data[4639] = 0.0 - k[1213]*y[IDX_HeII];
    data[4640] = 0.0 - k[172]*y[IDX_HeII] - k[950]*y[IDX_HeII];
    data[4641] = 0.0 - k[2139]*y[IDX_HeII];
    data[4642] = 0.0 - k[195]*y[IDX_HeII];
    data[4643] = 0.0 + k[1224]*y[IDX_HeII] + k[1225]*y[IDX_HeII];
    data[4644] = 0.0 + k[1228]*y[IDX_HeII];
    data[4645] = 0.0 + k[563]*y[IDX_EM] + k[951]*y[IDX_H2I] + k[1106]*y[IDX_HI];
    data[4646] = 0.0 + k[1191]*y[IDX_HeII];
    data[4647] = 0.0 + k[1275]*y[IDX_HeII];
    data[4648] = 0.0 + k[1194]*y[IDX_HeII] + k[1195]*y[IDX_HeII];
    data[4649] = 0.0 + k[1190]*y[IDX_HeII];
    data[4650] = 0.0 + k[1238]*y[IDX_HeII];
    data[4651] = 0.0 + k[1241]*y[IDX_HeII];
    data[4652] = 0.0 + k[1247]*y[IDX_HeII] + k[1248]*y[IDX_HeII];
    data[4653] = 0.0 + k[1197]*y[IDX_HeII];
    data[4654] = 0.0 + k[1274]*y[IDX_HeII];
    data[4655] = 0.0 + k[1219]*y[IDX_HeII] + k[1220]*y[IDX_HeII] + k[1221]*y[IDX_HeII];
    data[4656] = 0.0 + k[1229]*y[IDX_HeII] + k[1230]*y[IDX_HeII];
    data[4657] = 0.0 + k[1189]*y[IDX_HeII];
    data[4658] = 0.0 + k[1251]*y[IDX_HeII];
    data[4659] = 0.0 + k[1276]*y[IDX_HeII] + k[1277]*y[IDX_HeII];
    data[4660] = 0.0 + k[1239]*y[IDX_HeII] + k[1240]*y[IDX_HeII];
    data[4661] = 0.0 + k[1278]*y[IDX_HeII] + k[1279]*y[IDX_HeII];
    data[4662] = 0.0 + k[1287]*y[IDX_HeII] + k[1288]*y[IDX_HeII];
    data[4663] = 0.0 + k[1262]*y[IDX_HeII] + k[1263]*y[IDX_HeII];
    data[4664] = 0.0 + k[1269]*y[IDX_HeII];
    data[4665] = 0.0 + k[1280]*y[IDX_HeII] + k[1281]*y[IDX_HeII];
    data[4666] = 0.0 + k[1198]*y[IDX_HeII] + k[1199]*y[IDX_HeII];
    data[4667] = 0.0 + k[1259]*y[IDX_HeII] + k[1260]*y[IDX_HeII];
    data[4668] = 0.0 + k[218]*y[IDX_HeII] + k[1270]*y[IDX_HeII] + k[1271]*y[IDX_HeII];
    data[4669] = 0.0 + k[1245]*y[IDX_HeII] + k[1246]*y[IDX_HeII];
    data[4670] = 0.0 + k[1284]*y[IDX_HeII];
    data[4671] = 0.0 + k[1282]*y[IDX_HeII] + k[1283]*y[IDX_HeII];
    data[4672] = 0.0 + k[1181]*y[IDX_HeII] + k[1182]*y[IDX_HeII];
    data[4673] = 0.0 + k[1285]*y[IDX_HeII] + k[1286]*y[IDX_HeII];
    data[4674] = 0.0 + k[1200]*y[IDX_HeII] + k[1201]*y[IDX_HeII];
    data[4675] = 0.0 + k[1272]*y[IDX_HeII] + k[1273]*y[IDX_HeII];
    data[4676] = 0.0 + k[1183]*y[IDX_HeII] + k[1184]*y[IDX_HeII] + k[1185]*y[IDX_HeII];
    data[4677] = 0.0 + k[1214]*y[IDX_HeII] + k[1215]*y[IDX_HeII];
    data[4678] = 0.0 + k[1264]*y[IDX_HeII] + k[1265]*y[IDX_HeII] + k[1266]*y[IDX_HeII] +
        k[1267]*y[IDX_HeII];
    data[4679] = 0.0 + k[1249]*y[IDX_HeII];
    data[4680] = 0.0 - k[926]*y[IDX_HeI];
    data[4681] = 0.0 + k[219]*y[IDX_HeII];
    data[4682] = 0.0 + k[208]*y[IDX_HeII] + k[1178]*y[IDX_HeII] + k[1179]*y[IDX_HeII] +
        k[1180]*y[IDX_HeII];
    data[4683] = 0.0 + k[1242]*y[IDX_HeII] + k[1243]*y[IDX_HeII] + k[1244]*y[IDX_HeII];
    data[4684] = 0.0 + k[1209]*y[IDX_HeII] + k[1210]*y[IDX_HeII] + k[1211]*y[IDX_HeII] +
        k[1212]*y[IDX_HeII];
    data[4685] = 0.0 + k[1252]*y[IDX_HeII] + k[1253]*y[IDX_HeII];
    data[4686] = 0.0 + k[1186]*y[IDX_HeII] + k[1187]*y[IDX_HeII] + k[1188]*y[IDX_HeII];
    data[4687] = 0.0 + k[214]*y[IDX_HeII] + k[1226]*y[IDX_HeII] + k[1227]*y[IDX_HeII];
    data[4688] = 0.0 + k[215]*y[IDX_HeII] + k[1250]*y[IDX_HeII];
    data[4689] = 0.0 + k[207]*y[IDX_HeII] + k[1177]*y[IDX_HeII];
    data[4690] = 0.0 + k[1192]*y[IDX_HeII] + k[1193]*y[IDX_HeII];
    data[4691] = 0.0 + k[1196]*y[IDX_HeII];
    data[4692] = 0.0 + k[1256]*y[IDX_HeII];
    data[4693] = 0.0 + k[1207]*y[IDX_HeII] + k[1208]*y[IDX_HeII];
    data[4694] = 0.0 + k[212]*y[IDX_HeII] + k[1216]*y[IDX_HeII] + k[1217]*y[IDX_HeII] +
        k[1218]*y[IDX_HeII];
    data[4695] = 0.0 + k[210]*y[IDX_HeII] + k[1202]*y[IDX_HeII] + k[1203]*y[IDX_HeII] +
        k[1204]*y[IDX_HeII] + k[1205]*y[IDX_HeII];
    data[4696] = 0.0 + k[1235]*y[IDX_HeII] + k[1237]*y[IDX_HeII];
    data[4697] = 0.0 + k[1231]*y[IDX_HeII] + k[1232]*y[IDX_HeII] + k[1233]*y[IDX_HeII] +
        k[1234]*y[IDX_HeII];
    data[4698] = 0.0 + k[1257]*y[IDX_HeII] + k[1258]*y[IDX_HeII];
    data[4699] = 0.0 + k[211]*y[IDX_HeII] + k[1206]*y[IDX_HeII];
    data[4700] = 0.0 + k[216]*y[IDX_HeII] + k[1254]*y[IDX_HeII] + k[1255]*y[IDX_HeII];
    data[4701] = 0.0 + k[217]*y[IDX_HeII] + k[1261]*y[IDX_HeII];
    data[4702] = 0.0 + k[1268]*y[IDX_HeII];
    data[4703] = 0.0 + k[172]*y[IDX_H2I] + k[195]*y[IDX_HI] + k[207]*y[IDX_C2I] +
        k[208]*y[IDX_C2H2I] + k[209]*y[IDX_CI] + k[210]*y[IDX_CH4I] +
        k[211]*y[IDX_CHI] + k[212]*y[IDX_H2COI] + k[213]*y[IDX_H2OI] +
        k[214]*y[IDX_H2SI] + k[215]*y[IDX_N2I] + k[216]*y[IDX_NH3I] +
        k[217]*y[IDX_O2I] + k[218]*y[IDX_SO2I] + k[219]*y[IDX_SiI] +
        k[950]*y[IDX_H2I] + k[1177]*y[IDX_C2I] + k[1178]*y[IDX_C2H2I] +
        k[1179]*y[IDX_C2H2I] + k[1180]*y[IDX_C2H2I] + k[1181]*y[IDX_C2H3I] +
        k[1182]*y[IDX_C2H3I] + k[1183]*y[IDX_C2H4I] + k[1184]*y[IDX_C2H4I] +
        k[1185]*y[IDX_C2H4I] + k[1186]*y[IDX_C2HI] + k[1187]*y[IDX_C2HI] +
        k[1188]*y[IDX_C2HI] + k[1189]*y[IDX_C2NI] + k[1190]*y[IDX_C3NI] +
        k[1191]*y[IDX_C4HI] + k[1192]*y[IDX_CH2I] + k[1193]*y[IDX_CH2I] +
        k[1194]*y[IDX_CH2COI] + k[1195]*y[IDX_CH2COI] + k[1196]*y[IDX_CH3I] +
        k[1197]*y[IDX_CH3CCHI] + k[1198]*y[IDX_CH3CNI] + k[1199]*y[IDX_CH3CNI] +
        k[1200]*y[IDX_CH3OHI] + k[1201]*y[IDX_CH3OHI] + k[1202]*y[IDX_CH4I] +
        k[1203]*y[IDX_CH4I] + k[1204]*y[IDX_CH4I] + k[1205]*y[IDX_CH4I] +
        k[1206]*y[IDX_CHI] + k[1207]*y[IDX_CNI] + k[1208]*y[IDX_CNI] +
        k[1209]*y[IDX_CO2I] + k[1210]*y[IDX_CO2I] + k[1211]*y[IDX_CO2I] +
        k[1212]*y[IDX_CO2I] + k[1213]*y[IDX_COI] + k[1214]*y[IDX_CSI] +
        k[1215]*y[IDX_CSI] + k[1216]*y[IDX_H2COI] + k[1217]*y[IDX_H2COI] +
        k[1218]*y[IDX_H2COI] + k[1219]*y[IDX_H2CSI] + k[1220]*y[IDX_H2CSI] +
        k[1221]*y[IDX_H2CSI] + k[1222]*y[IDX_H2OI] + k[1223]*y[IDX_H2OI] +
        k[1224]*y[IDX_H2S2I] + k[1225]*y[IDX_H2S2I] + k[1226]*y[IDX_H2SI] +
        k[1227]*y[IDX_H2SI] + k[1228]*y[IDX_H2SiOI] + k[1229]*y[IDX_HC3NI] +
        k[1230]*y[IDX_HC3NI] + k[1231]*y[IDX_HCNI] + k[1232]*y[IDX_HCNI] +
        k[1233]*y[IDX_HCNI] + k[1234]*y[IDX_HCNI] + k[1235]*y[IDX_HCOI] +
        k[1237]*y[IDX_HCOI] + k[1238]*y[IDX_HCOOCH3I] + k[1239]*y[IDX_HCSI] +
        k[1240]*y[IDX_HCSI] + k[1241]*y[IDX_HClI] + k[1242]*y[IDX_HNCI] +
        k[1243]*y[IDX_HNCI] + k[1244]*y[IDX_HNCI] + k[1245]*y[IDX_HNOI] +
        k[1246]*y[IDX_HNOI] + k[1247]*y[IDX_HS2I] + k[1248]*y[IDX_HS2I] +
        k[1249]*y[IDX_HSI] + k[1250]*y[IDX_N2I] + k[1251]*y[IDX_NCCNI] +
        k[1252]*y[IDX_NH2I] + k[1253]*y[IDX_NH2I] + k[1254]*y[IDX_NH3I] +
        k[1255]*y[IDX_NH3I] + k[1256]*y[IDX_NHI] + k[1257]*y[IDX_NOI] +
        k[1258]*y[IDX_NOI] + k[1259]*y[IDX_NSI] + k[1260]*y[IDX_NSI] +
        k[1261]*y[IDX_O2I] + k[1262]*y[IDX_OCNI] + k[1263]*y[IDX_OCNI] +
        k[1264]*y[IDX_OCSI] + k[1265]*y[IDX_OCSI] + k[1266]*y[IDX_OCSI] +
        k[1267]*y[IDX_OCSI] + k[1268]*y[IDX_OHI] + k[1269]*y[IDX_S2I] +
        k[1270]*y[IDX_SO2I] + k[1271]*y[IDX_SO2I] + k[1272]*y[IDX_SOI] +
        k[1273]*y[IDX_SOI] + k[1274]*y[IDX_SiC2I] + k[1275]*y[IDX_SiC3I] +
        k[1276]*y[IDX_SiCI] + k[1277]*y[IDX_SiCI] + k[1278]*y[IDX_SiH2I] +
        k[1279]*y[IDX_SiH2I] + k[1280]*y[IDX_SiH3I] + k[1281]*y[IDX_SiH3I] +
        k[1282]*y[IDX_SiH4I] + k[1283]*y[IDX_SiH4I] + k[1284]*y[IDX_SiHI] +
        k[1285]*y[IDX_SiOI] + k[1286]*y[IDX_SiOI] + k[1287]*y[IDX_SiSI] +
        k[1288]*y[IDX_SiSI] + k[2139]*y[IDX_EM];
    data[4704] = 0.0 - k[363] - k[419] - k[926]*y[IDX_H2II] - k[2111]*y[IDX_HII];
    data[4705] = 0.0 - k[2111]*y[IDX_HeI];
    data[4706] = 0.0 + k[209]*y[IDX_HeII];
    data[4707] = 0.0 + k[213]*y[IDX_HeII] + k[1222]*y[IDX_HeII] + k[1223]*y[IDX_HeII];
    data[4708] = 0.0 + k[1213]*y[IDX_HeII];
    data[4709] = 0.0 + k[172]*y[IDX_HeII] + k[950]*y[IDX_HeII] + k[951]*y[IDX_HeHII];
    data[4710] = 0.0 + k[563]*y[IDX_HeHII] + k[2139]*y[IDX_HeII];
    data[4711] = 0.0 + k[195]*y[IDX_HeII] + k[1106]*y[IDX_HeHII];
    data[4712] = 0.0 + k[951]*y[IDX_H2I];
    data[4713] = 0.0 - k[1040]*y[IDX_H3II];
    data[4714] = 0.0 - k[1047]*y[IDX_H3II];
    data[4715] = 0.0 - k[1049]*y[IDX_H3II];
    data[4716] = 0.0 - k[1052]*y[IDX_H3II];
    data[4717] = 0.0 - k[1042]*y[IDX_H3II];
    data[4718] = 0.0 - k[1059]*y[IDX_H3II];
    data[4719] = 0.0 - k[1023]*y[IDX_H3II] - k[1024]*y[IDX_H3II];
    data[4720] = 0.0 - k[1026]*y[IDX_H3II];
    data[4721] = 0.0 - k[1048]*y[IDX_H3II];
    data[4722] = 0.0 - k[1072]*y[IDX_H3II];
    data[4723] = 0.0 - k[1077]*y[IDX_H3II];
    data[4724] = 0.0 - k[1067]*y[IDX_H3II];
    data[4725] = 0.0 - k[1073]*y[IDX_H3II];
    data[4726] = 0.0 - k[1030]*y[IDX_H3II];
    data[4727] = 0.0 - k[1061]*y[IDX_H3II];
    data[4728] = 0.0 - k[1069]*y[IDX_H3II];
    data[4729] = 0.0 - k[1051]*y[IDX_H3II];
    data[4730] = 0.0 - k[1075]*y[IDX_H3II];
    data[4731] = 0.0 - k[1074]*y[IDX_H3II];
    data[4732] = 0.0 - k[1076]*y[IDX_H3II];
    data[4733] = 0.0 - k[1031]*y[IDX_H3II] - k[1032]*y[IDX_H3II];
    data[4734] = 0.0 - k[1070]*y[IDX_H3II];
    data[4735] = 0.0 - k[1039]*y[IDX_H3II];
    data[4736] = 0.0 - k[1054]*y[IDX_H3II];
    data[4737] = 0.0 + k[959]*y[IDX_H2I];
    data[4738] = 0.0 - k[1065]*y[IDX_H3II];
    data[4739] = 0.0 - k[1053]*y[IDX_H3II];
    data[4740] = 0.0 + k[920]*y[IDX_H2I] + k[925]*y[IDX_HCOI];
    data[4741] = 0.0 - k[1071]*y[IDX_H3II];
    data[4742] = 0.0 - k[1050]*y[IDX_H3II];
    data[4743] = 0.0 - k[1036]*y[IDX_H3II];
    data[4744] = 0.0 + k[954]*y[IDX_H2I];
    data[4745] = 0.0 - k[1056]*y[IDX_H3II];
    data[4746] = 0.0 - k[1025]*y[IDX_H3II];
    data[4747] = 0.0 - k[1044]*y[IDX_H3II];
    data[4748] = 0.0 - k[1055]*y[IDX_H3II];
    data[4749] = 0.0 - k[1022]*y[IDX_H3II];
    data[4750] = 0.0 - k[1028]*y[IDX_H3II];
    data[4751] = 0.0 - k[1029]*y[IDX_H3II];
    data[4752] = 0.0 - k[1058]*y[IDX_H3II];
    data[4753] = 0.0 - k[1035]*y[IDX_H3II];
    data[4754] = 0.0 - k[1041]*y[IDX_H3II];
    data[4755] = 0.0 - k[1033]*y[IDX_H3II];
    data[4756] = 0.0 + k[925]*y[IDX_H2II] - k[1046]*y[IDX_H3II];
    data[4757] = 0.0 - k[1045]*y[IDX_H3II];
    data[4758] = 0.0 - k[1060]*y[IDX_H3II];
    data[4759] = 0.0 - k[1034]*y[IDX_H3II];
    data[4760] = 0.0 - k[1057]*y[IDX_H3II];
    data[4761] = 0.0 - k[1062]*y[IDX_H3II];
    data[4762] = 0.0 - k[1068]*y[IDX_H3II];
    data[4763] = 0.0 - k[1066]*y[IDX_H3II];
    data[4764] = 0.0 - k[519]*y[IDX_EM] - k[520]*y[IDX_EM] - k[1022]*y[IDX_C2I] -
        k[1023]*y[IDX_C2H5OHI] - k[1024]*y[IDX_C2H5OHI] - k[1025]*y[IDX_C2HI] -
        k[1026]*y[IDX_C2NI] - k[1027]*y[IDX_CI] - k[1028]*y[IDX_CH2I] -
        k[1029]*y[IDX_CH3I] - k[1030]*y[IDX_CH3CNI] - k[1031]*y[IDX_CH3OHI] -
        k[1032]*y[IDX_CH3OHI] - k[1033]*y[IDX_CH4I] - k[1034]*y[IDX_CHI] -
        k[1035]*y[IDX_CNI] - k[1036]*y[IDX_CO2I] - k[1037]*y[IDX_COI] -
        k[1038]*y[IDX_COI] - k[1039]*y[IDX_CSI] - k[1040]*y[IDX_ClI] -
        k[1041]*y[IDX_H2COI] - k[1042]*y[IDX_H2CSI] - k[1043]*y[IDX_H2OI] -
        k[1044]*y[IDX_H2SI] - k[1045]*y[IDX_HCNI] - k[1046]*y[IDX_HCOI] -
        k[1047]*y[IDX_HCOOCH3I] - k[1048]*y[IDX_HCSI] - k[1049]*y[IDX_HClI] -
        k[1050]*y[IDX_HNCI] - k[1051]*y[IDX_HNOI] - k[1052]*y[IDX_HS2I] -
        k[1053]*y[IDX_HSI] - k[1054]*y[IDX_MgI] - k[1055]*y[IDX_N2I] -
        k[1056]*y[IDX_NH2I] - k[1057]*y[IDX_NH3I] - k[1058]*y[IDX_NHI] -
        k[1059]*y[IDX_NO2I] - k[1060]*y[IDX_NOI] - k[1061]*y[IDX_NSI] -
        k[1062]*y[IDX_O2I] - k[1063]*y[IDX_OI] - k[1064]*y[IDX_OI] -
        k[1065]*y[IDX_OCSI] - k[1066]*y[IDX_OHI] - k[1067]*y[IDX_S2I] -
        k[1068]*y[IDX_SI] - k[1069]*y[IDX_SO2I] - k[1070]*y[IDX_SOI] -
        k[1071]*y[IDX_SiI] - k[1072]*y[IDX_SiH2I] - k[1073]*y[IDX_SiH3I] -
        k[1074]*y[IDX_SiH4I] - k[1075]*y[IDX_SiHI] - k[1076]*y[IDX_SiOI] -
        k[1077]*y[IDX_SiSI] - k[2025] - k[2026];
    data[4765] = 0.0 - k[1063]*y[IDX_H3II] - k[1064]*y[IDX_H3II];
    data[4766] = 0.0 - k[1027]*y[IDX_H3II];
    data[4767] = 0.0 - k[1043]*y[IDX_H3II];
    data[4768] = 0.0 - k[1037]*y[IDX_H3II] - k[1038]*y[IDX_H3II];
    data[4769] = 0.0 + k[920]*y[IDX_H2II] + k[951]*y[IDX_HeHII] + k[954]*y[IDX_NHII] +
        k[959]*y[IDX_O2HII];
    data[4770] = 0.0 - k[519]*y[IDX_H3II] - k[520]*y[IDX_H3II];
    data[4771] = 0.0 - k[122]*y[IDX_HII];
    data[4772] = 0.0 - k[896]*y[IDX_HII];
    data[4773] = 0.0 - k[900]*y[IDX_HII];
    data[4774] = 0.0 - k[109]*y[IDX_HII];
    data[4775] = 0.0 + k[108]*y[IDX_HI];
    data[4776] = 0.0 - k[145]*y[IDX_HII];
    data[4777] = 0.0 - k[126]*y[IDX_HII];
    data[4778] = 0.0 - k[127]*y[IDX_HII];
    data[4779] = 0.0 - k[144]*y[IDX_HII];
    data[4780] = 0.0 - k[120]*y[IDX_HII];
    data[4781] = 0.0 - k[903]*y[IDX_HII];
    data[4782] = 0.0 - k[113]*y[IDX_HII];
    data[4783] = 0.0 - k[146]*y[IDX_HII];
    data[4784] = 0.0 - k[899]*y[IDX_HII] + k[1240]*y[IDX_HeII];
    data[4785] = 0.0 - k[147]*y[IDX_HII] - k[905]*y[IDX_HII];
    data[4786] = 0.0 - k[152]*y[IDX_HII];
    data[4787] = 0.0 - k[139]*y[IDX_HII];
    data[4788] = 0.0 - k[148]*y[IDX_HII] - k[906]*y[IDX_HII];
    data[4789] = 0.0 - k[886]*y[IDX_HII];
    data[4790] = 0.0 - k[134]*y[IDX_HII];
    data[4791] = 0.0 - k[141]*y[IDX_HII];
    data[4792] = 0.0 - k[901]*y[IDX_HII] + k[1246]*y[IDX_HeII];
    data[4793] = 0.0 - k[150]*y[IDX_HII] - k[908]*y[IDX_HII];
    data[4794] = 0.0 - k[149]*y[IDX_HII] - k[907]*y[IDX_HII];
    data[4795] = 0.0 - k[882]*y[IDX_HII];
    data[4796] = 0.0 - k[151]*y[IDX_HII];
    data[4797] = 0.0 - k[887]*y[IDX_HII] - k[888]*y[IDX_HII] - k[889]*y[IDX_HII];
    data[4798] = 0.0 - k[142]*y[IDX_HII];
    data[4799] = 0.0 - k[883]*y[IDX_HII];
    data[4800] = 0.0 - k[118]*y[IDX_HII];
    data[4801] = 0.0 - k[129]*y[IDX_HII];
    data[4802] = 0.0 + k[191]*y[IDX_HI];
    data[4803] = 0.0 - k[137]*y[IDX_HII] - k[904]*y[IDX_HII];
    data[4804] = 0.0 - k[128]*y[IDX_HII] - k[902]*y[IDX_HII];
    data[4805] = 0.0 + k[193]*y[IDX_HI] + k[2009];
    data[4806] = 0.0 - k[143]*y[IDX_HII];
    data[4807] = 0.0 - k[111]*y[IDX_HII];
    data[4808] = 0.0 + k[194]*y[IDX_HI];
    data[4809] = 0.0 - k[1]*y[IDX_HII] + k[1]*y[IDX_HII];
    data[4810] = 0.0 + k[192]*y[IDX_HI];
    data[4811] = 0.0 + k[2040];
    data[4812] = 0.0 - k[891]*y[IDX_HII];
    data[4813] = 0.0 + k[2048];
    data[4814] = 0.0 - k[130]*y[IDX_HII];
    data[4815] = 0.0 - k[112]*y[IDX_HII] - k[884]*y[IDX_HII];
    data[4816] = 0.0 + k[196]*y[IDX_HI];
    data[4817] = 0.0 + k[1980];
    data[4818] = 0.0 - k[123]*y[IDX_HII] - k[894]*y[IDX_HII] - k[895]*y[IDX_HII];
    data[4819] = 0.0 - k[110]*y[IDX_HII];
    data[4820] = 0.0 - k[114]*y[IDX_HII] - k[885]*y[IDX_HII];
    data[4821] = 0.0 - k[115]*y[IDX_HII];
    data[4822] = 0.0 + k[1977];
    data[4823] = 0.0 - k[132]*y[IDX_HII];
    data[4824] = 0.0 - k[119]*y[IDX_HII] - k[892]*y[IDX_HII] - k[893]*y[IDX_HII];
    data[4825] = 0.0 - k[116]*y[IDX_HII] - k[890]*y[IDX_HII] + k[1205]*y[IDX_HeII];
    data[4826] = 0.0 - k[125]*y[IDX_HII] - k[897]*y[IDX_HII] - k[898]*y[IDX_HII];
    data[4827] = 0.0 - k[124]*y[IDX_HII];
    data[4828] = 0.0 - k[133]*y[IDX_HII];
    data[4829] = 0.0 - k[117]*y[IDX_HII];
    data[4830] = 0.0 - k[131]*y[IDX_HII];
    data[4831] = 0.0 - k[135]*y[IDX_HII];
    data[4832] = 0.0 - k[140]*y[IDX_HII];
    data[4833] = 0.0 - k[138]*y[IDX_HII];
    data[4834] = 0.0 + k[195]*y[IDX_HI] + k[950]*y[IDX_H2I] + k[1205]*y[IDX_CH4I] +
        k[1223]*y[IDX_H2OI] + k[1240]*y[IDX_HCSI] + k[1246]*y[IDX_HNOI];
    data[4835] = 0.0 - k[2111]*y[IDX_HII];
    data[4836] = 0.0 + k[2026];
    data[4837] = 0.0 - k[1]*y[IDX_HNCI] + k[1]*y[IDX_HNCI] - k[109]*y[IDX_ClI] -
        k[110]*y[IDX_C2I] - k[111]*y[IDX_C2H2I] - k[112]*y[IDX_C2HI] -
        k[113]*y[IDX_C2NI] - k[114]*y[IDX_CH2I] - k[115]*y[IDX_CH3I] -
        k[116]*y[IDX_CH4I] - k[117]*y[IDX_CHI] - k[118]*y[IDX_CSI] -
        k[119]*y[IDX_H2COI] - k[120]*y[IDX_H2CSI] - k[121]*y[IDX_H2OI] -
        k[122]*y[IDX_H2S2I] - k[123]*y[IDX_H2SI] - k[124]*y[IDX_HCNI] -
        k[125]*y[IDX_HCOI] - k[126]*y[IDX_HClI] - k[127]*y[IDX_HS2I] -
        k[128]*y[IDX_HSI] - k[129]*y[IDX_MgI] - k[130]*y[IDX_NH2I] -
        k[131]*y[IDX_NH3I] - k[132]*y[IDX_NHI] - k[133]*y[IDX_NOI] -
        k[134]*y[IDX_NSI] - k[135]*y[IDX_O2I] - k[136]*y[IDX_OI] -
        k[137]*y[IDX_OCSI] - k[138]*y[IDX_OHI] - k[139]*y[IDX_S2I] -
        k[140]*y[IDX_SI] - k[141]*y[IDX_SO2I] - k[142]*y[IDX_SOI] -
        k[143]*y[IDX_SiI] - k[144]*y[IDX_SiC2I] - k[145]*y[IDX_SiC3I] -
        k[146]*y[IDX_SiCI] - k[147]*y[IDX_SiH2I] - k[148]*y[IDX_SiH3I] -
        k[149]*y[IDX_SiH4I] - k[150]*y[IDX_SiHI] - k[151]*y[IDX_SiOI] -
        k[152]*y[IDX_SiSI] - k[882]*y[IDX_C2H3I] - k[883]*y[IDX_C2H4I] -
        k[884]*y[IDX_C2HI] - k[885]*y[IDX_CH2I] - k[886]*y[IDX_CH3CNI] -
        k[887]*y[IDX_CH3OHI] - k[888]*y[IDX_CH3OHI] - k[889]*y[IDX_CH3OHI] -
        k[890]*y[IDX_CH4I] - k[891]*y[IDX_CO2I] - k[892]*y[IDX_H2COI] -
        k[893]*y[IDX_H2COI] - k[894]*y[IDX_H2SI] - k[895]*y[IDX_H2SI] -
        k[896]*y[IDX_H2SiOI] - k[897]*y[IDX_HCOI] - k[898]*y[IDX_HCOI] -
        k[899]*y[IDX_HCSI] - k[900]*y[IDX_HNCOI] - k[901]*y[IDX_HNOI] -
        k[902]*y[IDX_HSI] - k[903]*y[IDX_NO2I] - k[904]*y[IDX_OCSI] -
        k[905]*y[IDX_SiH2I] - k[906]*y[IDX_SiH3I] - k[907]*y[IDX_SiH4I] -
        k[908]*y[IDX_SiHI] - k[2110]*y[IDX_HI] - k[2111]*y[IDX_HeI] -
        k[2135]*y[IDX_EM];
    data[4838] = 0.0 - k[136]*y[IDX_HII];
    data[4839] = 0.0 - k[121]*y[IDX_HII] + k[1223]*y[IDX_HeII];
    data[4840] = 0.0 + k[359] + k[950]*y[IDX_HeII];
    data[4841] = 0.0 - k[2135]*y[IDX_HII];
    data[4842] = 0.0 + k[108]*y[IDX_ClII] + k[191]*y[IDX_CNII] + k[192]*y[IDX_COII] +
        k[193]*y[IDX_H2II] + k[194]*y[IDX_HCNII] + k[195]*y[IDX_HeII] +
        k[196]*y[IDX_OII] + k[362] + k[406] - k[2110]*y[IDX_HII];
    data[4843] = 0.0 - k[1878]*y[IDX_OI];
    data[4844] = 0.0 - k[1885]*y[IDX_OI];
    data[4845] = 0.0 - k[1919]*y[IDX_OI];
    data[4846] = 0.0 + k[559]*y[IDX_EM];
    data[4847] = 0.0 - k[1877]*y[IDX_OI];
    data[4848] = 0.0 - k[1509]*y[IDX_OI];
    data[4849] = 0.0 - k[1918]*y[IDX_OI];
    data[4850] = 0.0 + k[585]*y[IDX_EM] + k[585]*y[IDX_EM] + k[586]*y[IDX_EM];
    data[4851] = 0.0 + k[1769]*y[IDX_HI] - k[1909]*y[IDX_OI] + k[2064];
    data[4852] = 0.0 + k[431] + k[1820]*y[IDX_NI] + k[1820]*y[IDX_NI] - k[1905]*y[IDX_OI]
        + k[2056];
    data[4853] = 0.0 - k[1874]*y[IDX_OI];
    data[4854] = 0.0 - k[1512]*y[IDX_OI];
    data[4855] = 0.0 - k[1876]*y[IDX_OI];
    data[4856] = 0.0 - k[1920]*y[IDX_OI] - k[1921]*y[IDX_OI];
    data[4857] = 0.0 - k[1894]*y[IDX_OI] - k[1895]*y[IDX_OI];
    data[4858] = 0.0 - k[1922]*y[IDX_OI] - k[1923]*y[IDX_OI];
    data[4859] = 0.0 - k[1515]*y[IDX_OI];
    data[4860] = 0.0 + k[439] + k[1262]*y[IDX_HeII] + k[1772]*y[IDX_HI] -
        k[1910]*y[IDX_OI] - k[1911]*y[IDX_OI] + k[2065];
    data[4861] = 0.0 - k[1915]*y[IDX_OI];
    data[4862] = 0.0 - k[1924]*y[IDX_OI];
    data[4863] = 0.0 - k[1907]*y[IDX_OI] - k[1908]*y[IDX_OI];
    data[4864] = 0.0 + k[580]*y[IDX_EM];
    data[4865] = 0.0 - k[1514]*y[IDX_OI];
    data[4866] = 0.0 + k[320]*y[IDX_OII] + k[445] + k[1271]*y[IDX_HeII] -
        k[1916]*y[IDX_OI] + k[2074];
    data[4867] = 0.0 + k[1755]*y[IDX_HI] - k[1896]*y[IDX_OI] - k[1897]*y[IDX_OI] -
        k[1898]*y[IDX_OI];
    data[4868] = 0.0 + k[1536]*y[IDX_OHII] - k[1926]*y[IDX_OI];
    data[4869] = 0.0 + k[602]*y[IDX_EM] - k[1516]*y[IDX_OI] + k[2092];
    data[4870] = 0.0 - k[1513]*y[IDX_OI];
    data[4871] = 0.0 - k[1925]*y[IDX_OI];
    data[4872] = 0.0 - k[1869]*y[IDX_OI];
    data[4873] = 0.0 + k[544]*y[IDX_EM] - k[1500]*y[IDX_OI];
    data[4874] = 0.0 + k[1485]*y[IDX_O2I] - k[1496]*y[IDX_OI];
    data[4875] = 0.0 - k[1501]*y[IDX_OI] - k[1502]*y[IDX_OI];
    data[4876] = 0.0 + k[456] + k[1285]*y[IDX_HeII] + k[1537]*y[IDX_OHII] + k[2093];
    data[4877] = 0.0 + k[446] + k[639]*y[IDX_CII] + k[1272]*y[IDX_HeII] +
        k[1615]*y[IDX_CI] + k[1778]*y[IDX_HI] + k[1830]*y[IDX_NI] +
        k[1866]*y[IDX_O2I] - k[1917]*y[IDX_OI] + k[1955]*y[IDX_SI] + k[2075] -
        k[2129]*y[IDX_OI];
    data[4878] = 0.0 - k[1870]*y[IDX_OI] - k[1871]*y[IDX_OI] - k[1872]*y[IDX_OI] -
        k[1873]*y[IDX_OI];
    data[4879] = 0.0 - k[1493]*y[IDX_OI];
    data[4880] = 0.0 - k[1883]*y[IDX_OI] - k[1884]*y[IDX_OI];
    data[4881] = 0.0 + k[584]*y[IDX_EM] + k[1341]*y[IDX_NI];
    data[4882] = 0.0 - k[1490]*y[IDX_OI];
    data[4883] = 0.0 - k[1510]*y[IDX_OI];
    data[4884] = 0.0 - k[326]*y[IDX_OI];
    data[4885] = 0.0 - k[1506]*y[IDX_OI];
    data[4886] = 0.0 + k[318]*y[IDX_OII] + k[1264]*y[IDX_HeII] - k[1912]*y[IDX_OI] -
        k[1913]*y[IDX_OI];
    data[4887] = 0.0 - k[1498]*y[IDX_OI] - k[1499]*y[IDX_OI];
    data[4888] = 0.0 - k[1494]*y[IDX_OI] - k[1495]*y[IDX_OI];
    data[4889] = 0.0 - k[328]*y[IDX_OI] - k[1505]*y[IDX_OI];
    data[4890] = 0.0 - k[1899]*y[IDX_OI] - k[1900]*y[IDX_OI];
    data[4891] = 0.0 - k[932]*y[IDX_OI];
    data[4892] = 0.0 + k[1535]*y[IDX_OHII] + k[1959]*y[IDX_O2I] - k[2131]*y[IDX_OI];
    data[4893] = 0.0 + k[307]*y[IDX_OII] - k[1868]*y[IDX_OI];
    data[4894] = 0.0 + k[1528]*y[IDX_OHII];
    data[4895] = 0.0 - k[2130]*y[IDX_OI];
    data[4896] = 0.0 - k[327]*y[IDX_OI] + k[499]*y[IDX_EM] + k[1539]*y[IDX_OHI] + k[2002];
    data[4897] = 0.0 - k[1503]*y[IDX_OI] - k[1504]*y[IDX_OI];
    data[4898] = 0.0 + k[393] + k[891]*y[IDX_HII] + k[1209]*y[IDX_HeII] +
        k[1520]*y[IDX_OHII] - k[1882]*y[IDX_OI] + k[2003];
    data[4899] = 0.0 + k[1358]*y[IDX_H2OI] + k[1367]*y[IDX_NOI] - k[1370]*y[IDX_OI];
    data[4900] = 0.0 + k[315]*y[IDX_OII] + k[1411]*y[IDX_OHII] + k[1837]*y[IDX_OHI] -
        k[1902]*y[IDX_OI] - k[1903]*y[IDX_OI];
    data[4901] = 0.0 + k[577]*y[IDX_EM] + k[577]*y[IDX_EM] + k[700]*y[IDX_CI] +
        k[769]*y[IDX_CH2I] + k[858]*y[IDX_CHI] + k[1339]*y[IDX_NI] +
        k[1458]*y[IDX_NHI] + k[1484]*y[IDX_SI] + k[2060];
    data[4902] = 0.0 + k[1309]*y[IDX_NOI] + k[1310]*y[IDX_O2I];
    data[4903] = 0.0 + k[1380]*y[IDX_H2OI] + k[1390]*y[IDX_O2I] - k[1507]*y[IDX_OI];
    data[4904] = 0.0 - k[1491]*y[IDX_OI];
    data[4905] = 0.0 + k[308]*y[IDX_OII] + k[1518]*y[IDX_OHII] - k[1875]*y[IDX_OI];
    data[4906] = 0.0 + k[69]*y[IDX_CH2I] + k[90]*y[IDX_CHI] + k[196]*y[IDX_HI] +
        k[297]*y[IDX_NHI] + k[306]*y[IDX_C2I] + k[307]*y[IDX_C2H2I] +
        k[308]*y[IDX_C2HI] + k[309]*y[IDX_CH4I] + k[310]*y[IDX_COI] +
        k[311]*y[IDX_H2COI] + k[312]*y[IDX_H2OI] + k[313]*y[IDX_H2SI] +
        k[314]*y[IDX_HCOI] + k[315]*y[IDX_NH2I] + k[316]*y[IDX_NH3I] +
        k[317]*y[IDX_O2I] + k[318]*y[IDX_OCSI] + k[319]*y[IDX_OHI] +
        k[320]*y[IDX_SO2I] + k[2142]*y[IDX_EM];
    data[4907] = 0.0 + k[575]*y[IDX_EM];
    data[4908] = 0.0 + k[502]*y[IDX_EM];
    data[4909] = 0.0 - k[750]*y[IDX_OI];
    data[4910] = 0.0 - k[1508]*y[IDX_OI] + k[1546]*y[IDX_OHI];
    data[4911] = 0.0 + k[313]*y[IDX_OII] + k[1524]*y[IDX_OHII] - k[1888]*y[IDX_OI];
    data[4912] = 0.0 + k[512]*y[IDX_EM] + k[513]*y[IDX_EM] - k[1497]*y[IDX_OI] +
        k[1540]*y[IDX_OHI];
    data[4913] = 0.0 + k[582]*y[IDX_EM] + k[702]*y[IDX_CI] + k[771]*y[IDX_CH2I] +
        k[819]*y[IDX_CH4I] + k[860]*y[IDX_CHI] + k[1411]*y[IDX_NH2I] +
        k[1460]*y[IDX_NHI] - k[1511]*y[IDX_OI] + k[1517]*y[IDX_C2I] +
        k[1518]*y[IDX_C2HI] + k[1519]*y[IDX_CNI] + k[1520]*y[IDX_CO2I] +
        k[1521]*y[IDX_COI] + k[1522]*y[IDX_H2COI] + k[1523]*y[IDX_H2OI] +
        k[1524]*y[IDX_H2SI] + k[1525]*y[IDX_HCNI] + k[1527]*y[IDX_HCOI] +
        k[1528]*y[IDX_HNCI] + k[1529]*y[IDX_N2I] + k[1530]*y[IDX_NH3I] +
        k[1531]*y[IDX_NOI] + k[1532]*y[IDX_OHI] + k[1533]*y[IDX_SI] +
        k[1535]*y[IDX_SiI] + k[1536]*y[IDX_SiHI] + k[1537]*y[IDX_SiOI];
    data[4914] = 0.0 + k[782]*y[IDX_O2I] - k[783]*y[IDX_OI] - k[784]*y[IDX_OI];
    data[4915] = 0.0 + k[1529]*y[IDX_OHII] - k[1901]*y[IDX_OI];
    data[4916] = 0.0 - k[1492]*y[IDX_OI];
    data[4917] = 0.0 + k[306]*y[IDX_OII] + k[1517]*y[IDX_OHII] - k[1867]*y[IDX_OI];
    data[4918] = 0.0 + k[69]*y[IDX_OII] + k[769]*y[IDX_O2II] + k[771]*y[IDX_OHII] +
        k[1635]*y[IDX_O2I] - k[1637]*y[IDX_OI] - k[1638]*y[IDX_OI] -
        k[1639]*y[IDX_OI] - k[1640]*y[IDX_OI] + k[1643]*y[IDX_OHI];
    data[4919] = 0.0 - k[1664]*y[IDX_OI] - k[1665]*y[IDX_OI] + k[1666]*y[IDX_OHI];
    data[4920] = 0.0 + k[733]*y[IDX_O2I] - k[735]*y[IDX_OI];
    data[4921] = 0.0 + k[297]*y[IDX_OII] + k[1458]*y[IDX_O2II] + k[1460]*y[IDX_OHII] +
        k[1847]*y[IDX_NOI] + k[1849]*y[IDX_O2I] - k[1851]*y[IDX_OI] -
        k[1852]*y[IDX_OI] + k[1855]*y[IDX_OHI];
    data[4922] = 0.0 + k[1486]*y[IDX_O2I];
    data[4923] = 0.0 + k[1519]*y[IDX_OHII] + k[1713]*y[IDX_O2I] - k[1880]*y[IDX_OI] -
        k[1881]*y[IDX_OI] + k[1932]*y[IDX_OHI];
    data[4924] = 0.0 + k[311]*y[IDX_OII] + k[1218]*y[IDX_HeII] + k[1522]*y[IDX_OHII] -
        k[1886]*y[IDX_OI];
    data[4925] = 0.0 + k[309]*y[IDX_OII] + k[819]*y[IDX_OHII] - k[1879]*y[IDX_OI];
    data[4926] = 0.0 + k[314]*y[IDX_OII] + k[1237]*y[IDX_HeII] + k[1527]*y[IDX_OHII] +
        k[1752]*y[IDX_HI] + k[1812]*y[IDX_NI] - k[1892]*y[IDX_OI] -
        k[1893]*y[IDX_OI];
    data[4927] = 0.0 + k[1525]*y[IDX_OHII] - k[1889]*y[IDX_OI] - k[1890]*y[IDX_OI] -
        k[1891]*y[IDX_OI];
    data[4928] = 0.0 + k[433] + k[1258]*y[IDX_HeII] + k[1309]*y[IDX_NII] +
        k[1367]*y[IDX_NHII] + k[1531]*y[IDX_OHII] + k[1604]*y[IDX_CI] +
        k[1684]*y[IDX_CHI] + k[1764]*y[IDX_HI] + k[1823]*y[IDX_NI] +
        k[1847]*y[IDX_NHI] + k[1859]*y[IDX_O2I] + k[1861]*y[IDX_SI] -
        k[1906]*y[IDX_OI] + k[2058];
    data[4929] = 0.0 - k[0]*y[IDX_OI] + k[90]*y[IDX_OII] + k[858]*y[IDX_O2II] +
        k[860]*y[IDX_OHII] + k[1684]*y[IDX_NOI] + k[1688]*y[IDX_O2I] +
        k[1690]*y[IDX_O2I] - k[1693]*y[IDX_OI] - k[1694]*y[IDX_OI];
    data[4930] = 0.0 + k[316]*y[IDX_OII] + k[1530]*y[IDX_OHII] - k[1904]*y[IDX_OI];
    data[4931] = 0.0 + k[529]*y[IDX_EM];
    data[4932] = 0.0 + k[6]*y[IDX_H2I] + k[6]*y[IDX_H2I] + k[12]*y[IDX_HI] +
        k[12]*y[IDX_HI] + k[317]*y[IDX_OII] + k[436] + k[436] +
        k[633]*y[IDX_CII] + k[733]*y[IDX_CHII] + k[782]*y[IDX_CH3II] +
        k[1261]*y[IDX_HeII] + k[1310]*y[IDX_NII] + k[1390]*y[IDX_NH2II] +
        k[1485]*y[IDX_CSII] + k[1486]*y[IDX_SII] + k[1608]*y[IDX_CI] +
        k[1635]*y[IDX_CH2I] + k[1688]*y[IDX_CHI] + k[1690]*y[IDX_CHI] +
        k[1713]*y[IDX_CNI] + k[1718]*y[IDX_COI] + k[1768]*y[IDX_HI] +
        k[1825]*y[IDX_NI] + k[1849]*y[IDX_NHI] + k[1859]*y[IDX_NOI] +
        k[1865]*y[IDX_SI] + k[1866]*y[IDX_SOI] + k[1959]*y[IDX_SiI] + k[2062] +
        k[2062];
    data[4933] = 0.0 + k[633]*y[IDX_O2I] + k[639]*y[IDX_SOI] - k[2098]*y[IDX_OI];
    data[4934] = 0.0 + k[1484]*y[IDX_O2II] + k[1533]*y[IDX_OHII] + k[1861]*y[IDX_NOI] +
        k[1865]*y[IDX_O2I] + k[1955]*y[IDX_SOI];
    data[4935] = 0.0 + k[7]*y[IDX_H2I] + k[13]*y[IDX_HI] + k[319]*y[IDX_OII] + k[442] +
        k[1532]*y[IDX_OHII] + k[1539]*y[IDX_COII] + k[1540]*y[IDX_H2OII] +
        k[1546]*y[IDX_NH3II] + k[1612]*y[IDX_CI] + k[1643]*y[IDX_CH2I] +
        k[1666]*y[IDX_CH3I] + k[1776]*y[IDX_HI] + k[1828]*y[IDX_NI] +
        k[1837]*y[IDX_NH2I] + k[1855]*y[IDX_NHI] - k[1914]*y[IDX_OI] +
        k[1932]*y[IDX_CNI] + k[1947]*y[IDX_OHI] + k[1947]*y[IDX_OHI] + k[2069];
    data[4936] = 0.0 + k[1339]*y[IDX_O2II] + k[1341]*y[IDX_SOII] + k[1812]*y[IDX_HCOI] +
        k[1820]*y[IDX_NO2I] + k[1820]*y[IDX_NO2I] + k[1823]*y[IDX_NOI] +
        k[1825]*y[IDX_O2I] + k[1828]*y[IDX_OHI] + k[1830]*y[IDX_SOI];
    data[4937] = 0.0 + k[1209]*y[IDX_CO2I] + k[1213]*y[IDX_COI] + k[1218]*y[IDX_H2COI] +
        k[1237]*y[IDX_HCOI] + k[1258]*y[IDX_NOI] + k[1261]*y[IDX_O2I] +
        k[1262]*y[IDX_OCNI] + k[1264]*y[IDX_OCSI] + k[1271]*y[IDX_SO2I] +
        k[1272]*y[IDX_SOI] + k[1285]*y[IDX_SiOI];
    data[4938] = 0.0 - k[1063]*y[IDX_OI] - k[1064]*y[IDX_OI];
    data[4939] = 0.0 - k[136]*y[IDX_OI] + k[891]*y[IDX_CO2I];
    data[4940] = 0.0 - k[0]*y[IDX_CHI] - k[136]*y[IDX_HII] - k[326]*y[IDX_CNII] -
        k[327]*y[IDX_COII] - k[328]*y[IDX_N2II] - k[365] - k[438] -
        k[735]*y[IDX_CHII] - k[750]*y[IDX_CH2II] - k[783]*y[IDX_CH3II] -
        k[784]*y[IDX_CH3II] - k[932]*y[IDX_H2II] - k[1063]*y[IDX_H3II] -
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
    data[4941] = 0.0 + k[700]*y[IDX_O2II] + k[702]*y[IDX_OHII] + k[1591]*y[IDX_COI] +
        k[1604]*y[IDX_NOI] + k[1608]*y[IDX_O2I] + k[1612]*y[IDX_OHI] +
        k[1615]*y[IDX_SOI] - k[2104]*y[IDX_OI];
    data[4942] = 0.0 + k[312]*y[IDX_OII] + k[1358]*y[IDX_NHII] + k[1380]*y[IDX_NH2II] +
        k[1523]*y[IDX_OHII] - k[1887]*y[IDX_OI];
    data[4943] = 0.0 + k[310]*y[IDX_OII] + k[394] + k[1213]*y[IDX_HeII] +
        k[1521]*y[IDX_OHII] + k[1591]*y[IDX_CI] + k[1718]*y[IDX_O2I] + k[2004];
    data[4944] = 0.0 + k[6]*y[IDX_O2I] + k[6]*y[IDX_O2I] + k[7]*y[IDX_OHI] -
        k[1733]*y[IDX_OI];
    data[4945] = 0.0 + k[499]*y[IDX_COII] + k[502]*y[IDX_H2COII] + k[512]*y[IDX_H2OII] +
        k[513]*y[IDX_H2OII] + k[529]*y[IDX_H3OII] + k[544]*y[IDX_HCO2II] +
        k[559]*y[IDX_HSO2II] + k[575]*y[IDX_NOII] + k[577]*y[IDX_O2II] +
        k[577]*y[IDX_O2II] + k[580]*y[IDX_OCSII] + k[582]*y[IDX_OHII] +
        k[584]*y[IDX_SOII] + k[585]*y[IDX_SO2II] + k[585]*y[IDX_SO2II] +
        k[586]*y[IDX_SO2II] + k[602]*y[IDX_SiOII] + k[2142]*y[IDX_OII];
    data[4946] = 0.0 + k[12]*y[IDX_O2I] + k[12]*y[IDX_O2I] + k[13]*y[IDX_OHI] +
        k[196]*y[IDX_OII] + k[1752]*y[IDX_HCOI] + k[1755]*y[IDX_HNOI] +
        k[1764]*y[IDX_NOI] + k[1768]*y[IDX_O2I] + k[1769]*y[IDX_O2HI] +
        k[1772]*y[IDX_OCNI] + k[1776]*y[IDX_OHI] + k[1778]*y[IDX_SOI] -
        k[2124]*y[IDX_OI];
    data[4947] = 0.0 - k[1585]*y[IDX_CI];
    data[4948] = 0.0 - k[1788]*y[IDX_CI];
    data[4949] = 0.0 + k[590]*y[IDX_EM];
    data[4950] = 0.0 - k[1593]*y[IDX_CI];
    data[4951] = 0.0 + k[476]*y[IDX_EM];
    data[4952] = 0.0 + k[28]*y[IDX_CII] + k[450] + k[1275]*y[IDX_HeII] + k[2080];
    data[4953] = 0.0 + k[589]*y[IDX_EM];
    data[4954] = 0.0 + k[473]*y[IDX_EM] + k[2292];
    data[4955] = 0.0 + k[27]*y[IDX_CII] + k[449];
    data[4956] = 0.0 - k[1583]*y[IDX_CI];
    data[4957] = 0.0 + k[587]*y[IDX_EM] + k[1512]*y[IDX_OI];
    data[4958] = 0.0 + k[376] - k[1584]*y[IDX_CI] + k[1973];
    data[4959] = 0.0 - k[1598]*y[IDX_CI];
    data[4960] = 0.0 + k[29]*y[IDX_CII] + k[451] + k[1276]*y[IDX_HeII] +
        k[1921]*y[IDX_OI] + k[2081];
    data[4961] = 0.0 + k[30]*y[IDX_CII];
    data[4962] = 0.0 + k[32]*y[IDX_CII];
    data[4963] = 0.0 - k[1609]*y[IDX_CI];
    data[4964] = 0.0 - k[1613]*y[IDX_CI];
    data[4965] = 0.0 + k[31]*y[IDX_CII];
    data[4966] = 0.0 + k[23]*y[IDX_CII] - k[1606]*y[IDX_CI] - k[1607]*y[IDX_CI];
    data[4967] = 0.0 + k[579]*y[IDX_EM];
    data[4968] = 0.0 - k[1614]*y[IDX_CI];
    data[4969] = 0.0 - k[1617]*y[IDX_CI];
    data[4970] = 0.0 - k[704]*y[IDX_CI];
    data[4971] = 0.0 - k[703]*y[IDX_CI];
    data[4972] = 0.0 - k[1582]*y[IDX_CI];
    data[4973] = 0.0 + k[469]*y[IDX_EM];
    data[4974] = 0.0 - k[695]*y[IDX_CI];
    data[4975] = 0.0 + k[500]*y[IDX_EM] + k[2005];
    data[4976] = 0.0 + k[708]*y[IDX_CHII];
    data[4977] = 0.0 + k[25]*y[IDX_CII] - k[1615]*y[IDX_CI] - k[1616]*y[IDX_CI];
    data[4978] = 0.0 + k[396] + k[1214]*y[IDX_HeII] - k[1592]*y[IDX_CI] +
        k[1884]*y[IDX_OI] + k[2007];
    data[4979] = 0.0 + k[19]*y[IDX_CII];
    data[4980] = 0.0 - k[50]*y[IDX_CI] + k[458]*y[IDX_EM] + k[458]*y[IDX_EM] +
        k[647]*y[IDX_C2I] + k[650]*y[IDX_SI] + k[1490]*y[IDX_OI] + k[1960];
    data[4981] = 0.0 - k[701]*y[IDX_CI];
    data[4982] = 0.0 - k[51]*y[IDX_CI] + k[498]*y[IDX_EM] + k[1332]*y[IDX_NI];
    data[4983] = 0.0 - k[698]*y[IDX_CI];
    data[4984] = 0.0 + k[24]*y[IDX_CII] + k[737]*y[IDX_CHII] - k[1610]*y[IDX_CI];
    data[4985] = 0.0 - k[691]*y[IDX_CI];
    data[4986] = 0.0 - k[696]*y[IDX_CI];
    data[4987] = 0.0 - k[689]*y[IDX_CI];
    data[4988] = 0.0 - k[53]*y[IDX_CI];
    data[4989] = 0.0 - k[1595]*y[IDX_CI] - k[1596]*y[IDX_CI];
    data[4990] = 0.0 - k[912]*y[IDX_CI];
    data[4991] = 0.0 + k[26]*y[IDX_CII] + k[1957]*y[IDX_COI];
    data[4992] = 0.0 - k[693]*y[IDX_CI];
    data[4993] = 0.0 + k[727]*y[IDX_CHII] + k[1244]*y[IDX_HeII];
    data[4994] = 0.0 - k[52]*y[IDX_CI] + k[499]*y[IDX_EM] + k[841]*y[IDX_CHI];
    data[4995] = 0.0 - k[697]*y[IDX_CI];
    data[4996] = 0.0 + k[1211]*y[IDX_HeII];
    data[4997] = 0.0 - k[699]*y[IDX_CI] + k[1347]*y[IDX_C2I];
    data[4998] = 0.0 - k[1599]*y[IDX_CI] - k[1600]*y[IDX_CI] - k[1601]*y[IDX_CI];
    data[4999] = 0.0 - k[54]*y[IDX_CI] - k[700]*y[IDX_CI];
    data[5000] = 0.0 + k[1297]*y[IDX_COI];
    data[5001] = 0.0 + k[460]*y[IDX_EM] - k[685]*y[IDX_CI] + k[1491]*y[IDX_OI];
    data[5002] = 0.0 + k[1187]*y[IDX_HeII];
    data[5003] = 0.0 + k[1463]*y[IDX_C2I] + k[1469]*y[IDX_CNI] - k[2103]*y[IDX_CI];
    data[5004] = 0.0 + k[478]*y[IDX_EM] + k[479]*y[IDX_EM] - k[687]*y[IDX_CI];
    data[5005] = 0.0 + k[856]*y[IDX_CHI];
    data[5006] = 0.0 + k[17]*y[IDX_CII] + k[721]*y[IDX_CHII];
    data[5007] = 0.0 - k[690]*y[IDX_CI];
    data[5008] = 0.0 - k[702]*y[IDX_CI];
    data[5009] = 0.0 - k[688]*y[IDX_CI];
    data[5010] = 0.0 - k[1597]*y[IDX_CI];
    data[5011] = 0.0 + k[366] + k[366] + k[647]*y[IDX_C2II] + k[658]*y[IDX_SII] +
        k[1177]*y[IDX_HeII] + k[1347]*y[IDX_NHII] + k[1463]*y[IDX_OII] +
        k[1573]*y[IDX_SI] + k[1736]*y[IDX_HI] + k[1790]*y[IDX_NI] +
        k[1867]*y[IDX_OI] + k[1962] + k[1962];
    data[5012] = 0.0 + k[14]*y[IDX_CII] - k[1586]*y[IDX_CI] - k[1587]*y[IDX_CI];
    data[5013] = 0.0 - k[1588]*y[IDX_CI];
    data[5014] = 0.0 + k[477]*y[IDX_EM] - k[686]*y[IDX_CI] + k[708]*y[IDX_CH3OHI] +
        k[716]*y[IDX_H2COI] + k[719]*y[IDX_H2OI] + k[721]*y[IDX_H2SI] +
        k[725]*y[IDX_HCNI] + k[727]*y[IDX_HNCI] + k[730]*y[IDX_NH3I] +
        k[737]*y[IDX_OCSI] + k[740]*y[IDX_SI] + k[1977];
    data[5015] = 0.0 - k[1602]*y[IDX_CI] - k[1603]*y[IDX_CI];
    data[5016] = 0.0 + k[658]*y[IDX_C2I] - k[2105]*y[IDX_CI];
    data[5017] = 0.0 + k[392] + k[1207]*y[IDX_HeII] + k[1469]*y[IDX_OII] -
        k[1590]*y[IDX_CI] + k[1714]*y[IDX_SI] + k[1807]*y[IDX_NI] +
        k[1881]*y[IDX_OI] + k[2001];
    data[5018] = 0.0 + k[16]*y[IDX_CII] + k[716]*y[IDX_CHII];
    data[5019] = 0.0 + k[18]*y[IDX_CII] - k[1594]*y[IDX_CI];
    data[5020] = 0.0 + k[725]*y[IDX_CHII];
    data[5021] = 0.0 + k[22]*y[IDX_CII] - k[1604]*y[IDX_CI] - k[1605]*y[IDX_CI];
    data[5022] = 0.0 + k[2]*y[IDX_H2I] + k[9]*y[IDX_HI] + k[15]*y[IDX_CII] + k[391] +
        k[841]*y[IDX_COII] + k[856]*y[IDX_NH3II] - k[1589]*y[IDX_CI] +
        k[1683]*y[IDX_NI] + k[1694]*y[IDX_OI] + k[1698]*y[IDX_SI] +
        k[1743]*y[IDX_HI] + k[1999];
    data[5023] = 0.0 + k[21]*y[IDX_CII] + k[730]*y[IDX_CHII];
    data[5024] = 0.0 - k[692]*y[IDX_CI];
    data[5025] = 0.0 - k[1608]*y[IDX_CI];
    data[5026] = 0.0 + k[14]*y[IDX_CH2I] + k[15]*y[IDX_CHI] + k[16]*y[IDX_H2COI] +
        k[17]*y[IDX_H2SI] + k[18]*y[IDX_HCOI] + k[19]*y[IDX_MgI] +
        k[21]*y[IDX_NH3I] + k[22]*y[IDX_NOI] + k[23]*y[IDX_NSI] +
        k[24]*y[IDX_OCSI] + k[25]*y[IDX_SOI] + k[26]*y[IDX_SiI] +
        k[27]*y[IDX_SiC2I] + k[28]*y[IDX_SiC3I] + k[29]*y[IDX_SiCI] +
        k[30]*y[IDX_SiH2I] + k[31]*y[IDX_SiH3I] + k[32]*y[IDX_SiSI] +
        k[345]*y[IDX_SI] - k[2096]*y[IDX_CI] + k[2132]*y[IDX_EM];
    data[5027] = 0.0 + k[345]*y[IDX_CII] + k[650]*y[IDX_C2II] + k[740]*y[IDX_CHII] +
        k[1573]*y[IDX_C2I] + k[1698]*y[IDX_CHI] + k[1714]*y[IDX_CNI] -
        k[2106]*y[IDX_CI];
    data[5028] = 0.0 - k[1611]*y[IDX_CI] - k[1612]*y[IDX_CI];
    data[5029] = 0.0 + k[1332]*y[IDX_CNII] + k[1683]*y[IDX_CHI] + k[1790]*y[IDX_C2I] +
        k[1807]*y[IDX_CNI] - k[2102]*y[IDX_CI];
    data[5030] = 0.0 - k[209]*y[IDX_CI] + k[1177]*y[IDX_C2I] + k[1187]*y[IDX_C2HI] +
        k[1207]*y[IDX_CNI] + k[1211]*y[IDX_CO2I] + k[1214]*y[IDX_CSI] +
        k[1244]*y[IDX_HNCI] + k[1275]*y[IDX_SiC3I] + k[1276]*y[IDX_SiCI];
    data[5031] = 0.0 - k[1027]*y[IDX_CI];
    data[5032] = 0.0 + k[1490]*y[IDX_C2II] + k[1491]*y[IDX_C2HII] + k[1512]*y[IDX_SiCII]
        + k[1694]*y[IDX_CHI] + k[1867]*y[IDX_C2I] + k[1881]*y[IDX_CNI] +
        k[1884]*y[IDX_CSI] + k[1921]*y[IDX_SiCI] - k[2104]*y[IDX_CI];
    data[5033] = 0.0 - k[50]*y[IDX_C2II] - k[51]*y[IDX_CNII] - k[52]*y[IDX_COII] -
        k[53]*y[IDX_N2II] - k[54]*y[IDX_O2II] - k[209]*y[IDX_HeII] - k[356] -
        k[379] - k[685]*y[IDX_C2HII] - k[686]*y[IDX_CHII] - k[687]*y[IDX_CH2II]
        - k[688]*y[IDX_CH3II] - k[689]*y[IDX_CH5II] - k[690]*y[IDX_H2OII] -
        k[691]*y[IDX_H2SII] - k[692]*y[IDX_H3OII] - k[693]*y[IDX_HCNII] -
        k[694]*y[IDX_HCOII] - k[695]*y[IDX_HCO2II] - k[696]*y[IDX_HNOII] -
        k[697]*y[IDX_HSII] - k[698]*y[IDX_N2HII] - k[699]*y[IDX_NHII] -
        k[700]*y[IDX_O2II] - k[701]*y[IDX_O2HII] - k[702]*y[IDX_OHII] -
        k[703]*y[IDX_SiHII] - k[704]*y[IDX_SiOII] - k[912]*y[IDX_H2II] -
        k[1027]*y[IDX_H3II] - k[1582]*y[IDX_C2H3I] - k[1583]*y[IDX_C2H5I] -
        k[1584]*y[IDX_C2NI] - k[1585]*y[IDX_C3H2I] - k[1586]*y[IDX_CH2I] -
        k[1587]*y[IDX_CH2I] - k[1588]*y[IDX_CH3I] - k[1589]*y[IDX_CHI] -
        k[1590]*y[IDX_CNI] - k[1591]*y[IDX_COI] - k[1592]*y[IDX_CSI] -
        k[1593]*y[IDX_H2CNI] - k[1594]*y[IDX_HCOI] - k[1595]*y[IDX_HSI] -
        k[1596]*y[IDX_HSI] - k[1597]*y[IDX_N2I] - k[1598]*y[IDX_NCCNI] -
        k[1599]*y[IDX_NH2I] - k[1600]*y[IDX_NH2I] - k[1601]*y[IDX_NH2I] -
        k[1602]*y[IDX_NHI] - k[1603]*y[IDX_NHI] - k[1604]*y[IDX_NOI] -
        k[1605]*y[IDX_NOI] - k[1606]*y[IDX_NSI] - k[1607]*y[IDX_NSI] -
        k[1608]*y[IDX_O2I] - k[1609]*y[IDX_OCNI] - k[1610]*y[IDX_OCSI] -
        k[1611]*y[IDX_OHI] - k[1612]*y[IDX_OHI] - k[1613]*y[IDX_S2I] -
        k[1614]*y[IDX_SO2I] - k[1615]*y[IDX_SOI] - k[1616]*y[IDX_SOI] -
        k[1617]*y[IDX_SiHI] - k[1722]*y[IDX_H2I] - k[1788]*y[IDX_HNCOI] -
        k[1976] - k[2096]*y[IDX_CII] - k[2101]*y[IDX_CI] - k[2101]*y[IDX_CI] -
        k[2101]*y[IDX_CI] - k[2101]*y[IDX_CI] - k[2102]*y[IDX_NI] -
        k[2103]*y[IDX_OII] - k[2104]*y[IDX_OI] - k[2105]*y[IDX_SII] -
        k[2106]*y[IDX_SI] - k[2113]*y[IDX_H2I] - k[2123]*y[IDX_HI] - k[2206];
    data[5034] = 0.0 - k[694]*y[IDX_CI];
    data[5035] = 0.0 + k[719]*y[IDX_CHII];
    data[5036] = 0.0 + k[394] + k[1297]*y[IDX_NII] - k[1591]*y[IDX_CI] +
        k[1745]*y[IDX_HI] + k[1957]*y[IDX_SiI] + k[2004];
    data[5037] = 0.0 + k[2]*y[IDX_CHI] - k[1722]*y[IDX_CI] - k[2113]*y[IDX_CI];
    data[5038] = 0.0 + k[458]*y[IDX_C2II] + k[458]*y[IDX_C2II] + k[460]*y[IDX_C2HII] +
        k[469]*y[IDX_C2NII] + k[473]*y[IDX_C3II] + k[476]*y[IDX_C4NII] +
        k[477]*y[IDX_CHII] + k[478]*y[IDX_CH2II] + k[479]*y[IDX_CH2II] +
        k[498]*y[IDX_CNII] + k[499]*y[IDX_COII] + k[500]*y[IDX_CSII] +
        k[579]*y[IDX_OCSII] + k[587]*y[IDX_SiCII] + k[589]*y[IDX_SiC2II] +
        k[590]*y[IDX_SiC3II] + k[2132]*y[IDX_CII];
    data[5039] = 0.0 + k[9]*y[IDX_CHI] + k[1736]*y[IDX_C2I] + k[1743]*y[IDX_CHI] +
        k[1745]*y[IDX_COI] - k[2123]*y[IDX_CI];
    data[5040] = 0.0 + k[5]*y[IDX_H2I];
    data[5041] = 0.0 + k[994]*y[IDX_H2OI];
    data[5042] = 0.0 + k[874]*y[IDX_COI];
    data[5043] = 0.0 + k[880]*y[IDX_COI];
    data[5044] = 0.0 - k[1145]*y[IDX_HCOII];
    data[5045] = 0.0 - k[1146]*y[IDX_HCOII];
    data[5046] = 0.0 - k[1137]*y[IDX_HCOII];
    data[5047] = 0.0 - k[1142]*y[IDX_HCOII];
    data[5048] = 0.0 - k[1136]*y[IDX_HCOII];
    data[5049] = 0.0 - k[1153]*y[IDX_HCOII];
    data[5050] = 0.0 - k[1157]*y[IDX_HCOII];
    data[5051] = 0.0 - k[1150]*y[IDX_HCOII];
    data[5052] = 0.0 - k[1138]*y[IDX_HCOII];
    data[5053] = 0.0 - k[1148]*y[IDX_HCOII];
    data[5054] = 0.0 - k[1155]*y[IDX_HCOII];
    data[5055] = 0.0 + k[206]*y[IDX_HCOI] + k[864]*y[IDX_CHI];
    data[5056] = 0.0 - k[1154]*y[IDX_HCOII];
    data[5057] = 0.0 + k[993]*y[IDX_H2OI];
    data[5058] = 0.0 + k[875]*y[IDX_COI] + k[1500]*y[IDX_OI];
    data[5059] = 0.0 + k[1502]*y[IDX_OI];
    data[5060] = 0.0 - k[1156]*y[IDX_HCOII];
    data[5061] = 0.0 + k[889]*y[IDX_HII] - k[1139]*y[IDX_HCOII];
    data[5062] = 0.0 - k[1152]*y[IDX_HCOII];
    data[5063] = 0.0 + k[797]*y[IDX_COI];
    data[5064] = 0.0 - k[1140]*y[IDX_HCOII];
    data[5065] = 0.0 - k[224]*y[IDX_HCOII];
    data[5066] = 0.0 + k[1557]*y[IDX_C2H2I];
    data[5067] = 0.0 + k[33]*y[IDX_HCOI];
    data[5068] = 0.0 + k[878]*y[IDX_COI];
    data[5069] = 0.0 + k[96]*y[IDX_HCOI] + k[865]*y[IDX_H2COI] + k[996]*y[IDX_H2OI];
    data[5070] = 0.0 + k[877]*y[IDX_COI];
    data[5071] = 0.0 - k[1149]*y[IDX_HCOII];
    data[5072] = 0.0 + k[203]*y[IDX_HCOI];
    data[5073] = 0.0 + k[876]*y[IDX_COI];
    data[5074] = 0.0 + k[826]*y[IDX_COI];
    data[5075] = 0.0 + k[254]*y[IDX_HCOI] + k[1314]*y[IDX_H2COI];
    data[5076] = 0.0 - k[1147]*y[IDX_HCOII];
    data[5077] = 0.0 + k[165]*y[IDX_HCOI] + k[919]*y[IDX_COI] + k[921]*y[IDX_H2COI];
    data[5078] = 0.0 - k[1566]*y[IDX_HCOII];
    data[5079] = 0.0 + k[1482]*y[IDX_O2II] + k[1557]*y[IDX_SOII];
    data[5080] = 0.0 + k[1111]*y[IDX_COI];
    data[5081] = 0.0 - k[1168]*y[IDX_HCOII];
    data[5082] = 0.0 + k[103]*y[IDX_HCOI] + k[677]*y[IDX_C2HI] + k[756]*y[IDX_CH2I] +
        k[807]*y[IDX_CH4I] + k[841]*y[IDX_CHI] + k[871]*y[IDX_H2COI] +
        k[872]*y[IDX_H2SI] + k[941]*y[IDX_H2I] + k[997]*y[IDX_H2OI] +
        k[1398]*y[IDX_NH2I] + k[1423]*y[IDX_NH3I] + k[1448]*y[IDX_NHI] +
        k[1539]*y[IDX_OHI];
    data[5083] = 0.0 + k[714]*y[IDX_CHII] + k[891]*y[IDX_HII];
    data[5084] = 0.0 + k[1353]*y[IDX_COI] + k[1355]*y[IDX_H2COI];
    data[5085] = 0.0 + k[1398]*y[IDX_COII] - k[1406]*y[IDX_HCOII];
    data[5086] = 0.0 + k[204]*y[IDX_HCOI] + k[858]*y[IDX_CHI] + k[972]*y[IDX_H2COI] +
        k[1482]*y[IDX_C2H2I];
    data[5087] = 0.0 + k[243]*y[IDX_HCOI] + k[1298]*y[IDX_H2COI];
    data[5088] = 0.0 + k[267]*y[IDX_HCOI];
    data[5089] = 0.0 + k[1491]*y[IDX_OI];
    data[5090] = 0.0 + k[677]*y[IDX_COII] - k[680]*y[IDX_HCOII];
    data[5091] = 0.0 + k[314]*y[IDX_HCOI] + k[1471]*y[IDX_H2COI] + k[1474]*y[IDX_HCNI];
    data[5092] = 0.0 + k[202]*y[IDX_HCOI] + k[967]*y[IDX_O2I];
    data[5093] = 0.0 + k[742]*y[IDX_H2COI] + k[749]*y[IDX_O2I] + k[750]*y[IDX_OI];
    data[5094] = 0.0 + k[278]*y[IDX_HCOI];
    data[5095] = 0.0 + k[872]*y[IDX_COII] - k[1143]*y[IDX_HCOII];
    data[5096] = 0.0 + k[180]*y[IDX_HCOI] + k[978]*y[IDX_COI];
    data[5097] = 0.0 + k[334]*y[IDX_HCOI] + k[1521]*y[IDX_COI];
    data[5098] = 0.0 + k[72]*y[IDX_HCOI] + k[777]*y[IDX_H2COI] + k[784]*y[IDX_OI];
    data[5099] = 0.0 + k[44]*y[IDX_HCOI] + k[1492]*y[IDX_OI];
    data[5100] = 0.0 - k[653]*y[IDX_HCOII];
    data[5101] = 0.0 + k[756]*y[IDX_COII] - k[763]*y[IDX_HCOII];
    data[5102] = 0.0 + k[55]*y[IDX_HCOI] + k[714]*y[IDX_CO2I] + k[717]*y[IDX_H2COI] +
        k[720]*y[IDX_H2OI] + k[733]*y[IDX_O2I];
    data[5103] = 0.0 + k[1448]*y[IDX_COII] - k[1452]*y[IDX_HCOII];
    data[5104] = 0.0 + k[205]*y[IDX_HCOI] + k[975]*y[IDX_H2COI];
    data[5105] = 0.0 + k[618]*y[IDX_CII] + k[717]*y[IDX_CHII] + k[742]*y[IDX_CH2II] +
        k[777]*y[IDX_CH3II] + k[865]*y[IDX_CNII] + k[871]*y[IDX_COII] +
        k[893]*y[IDX_HII] + k[921]*y[IDX_H2II] + k[972]*y[IDX_O2II] +
        k[975]*y[IDX_SII] - k[1141]*y[IDX_HCOII] + k[1217]*y[IDX_HeII] +
        k[1298]*y[IDX_NII] + k[1314]*y[IDX_N2II] + k[1355]*y[IDX_NHII] +
        k[1471]*y[IDX_OII] + k[2014];
    data[5106] = 0.0 + k[807]*y[IDX_COII];
    data[5107] = 0.0 + k[18]*y[IDX_CII] + k[33]*y[IDX_C2II] + k[44]*y[IDX_C2H2II] +
        k[55]*y[IDX_CHII] + k[72]*y[IDX_CH3II] + k[96]*y[IDX_CNII] +
        k[103]*y[IDX_COII] + k[125]*y[IDX_HII] + k[165]*y[IDX_H2II] +
        k[180]*y[IDX_H2OII] + k[202]*y[IDX_H2COII] + k[203]*y[IDX_H2SII] +
        k[204]*y[IDX_O2II] + k[205]*y[IDX_SII] + k[206]*y[IDX_SiOII] +
        k[243]*y[IDX_NII] + k[254]*y[IDX_N2II] + k[267]*y[IDX_NH2II] +
        k[278]*y[IDX_NH3II] + k[314]*y[IDX_OII] + k[334]*y[IDX_OHII] + k[410] -
        k[1144]*y[IDX_HCOII] + k[2031];
    data[5108] = 0.0 - k[1124]*y[IDX_HCOII] + k[1474]*y[IDX_OII];
    data[5109] = 0.0 + k[0]*y[IDX_OI] + k[841]*y[IDX_COII] - k[849]*y[IDX_HCOII] +
        k[858]*y[IDX_O2II] + k[864]*y[IDX_SiOII];
    data[5110] = 0.0 + k[1423]*y[IDX_COII] - k[1433]*y[IDX_HCOII];
    data[5111] = 0.0 + k[692]*y[IDX_CI];
    data[5112] = 0.0 + k[733]*y[IDX_CHII] + k[749]*y[IDX_CH2II] + k[967]*y[IDX_H2COII];
    data[5113] = 0.0 + k[18]*y[IDX_HCOI] + k[618]*y[IDX_H2COI] + k[620]*y[IDX_H2OI];
    data[5114] = 0.0 - k[1151]*y[IDX_HCOII];
    data[5115] = 0.0 + k[1539]*y[IDX_COII] - k[1542]*y[IDX_HCOII] - k[1543]*y[IDX_HCOII];
    data[5116] = 0.0 + k[1217]*y[IDX_H2COI];
    data[5117] = 0.0 + k[1037]*y[IDX_COI];
    data[5118] = 0.0 + k[125]*y[IDX_HCOI] + k[889]*y[IDX_CH3OHI] + k[891]*y[IDX_CO2I] +
        k[893]*y[IDX_H2COI];
    data[5119] = 0.0 + k[0]*y[IDX_CHI] + k[750]*y[IDX_CH2II] + k[784]*y[IDX_CH3II] +
        k[1491]*y[IDX_C2HII] + k[1492]*y[IDX_C2H2II] + k[1500]*y[IDX_HCO2II] +
        k[1502]*y[IDX_HCSII];
    data[5120] = 0.0 + k[692]*y[IDX_H3OII] - k[694]*y[IDX_HCOII];
    data[5121] = 0.0 - k[224]*y[IDX_MgI] - k[542]*y[IDX_EM] - k[653]*y[IDX_C2I] -
        k[680]*y[IDX_C2HI] - k[694]*y[IDX_CI] - k[763]*y[IDX_CH2I] -
        k[849]*y[IDX_CHI] - k[1003]*y[IDX_H2OI] - k[1124]*y[IDX_HCNI] -
        k[1136]*y[IDX_C2H5OHI] - k[1137]*y[IDX_CH3CCHI] - k[1138]*y[IDX_CH3CNI]
        - k[1139]*y[IDX_CH3OHI] - k[1140]*y[IDX_CSI] - k[1141]*y[IDX_H2COI] -
        k[1142]*y[IDX_H2CSI] - k[1143]*y[IDX_H2SI] - k[1144]*y[IDX_HCOI] -
        k[1145]*y[IDX_HCOOCH3I] - k[1146]*y[IDX_HS2I] - k[1147]*y[IDX_HSI] -
        k[1148]*y[IDX_NSI] - k[1149]*y[IDX_OCSI] - k[1150]*y[IDX_S2I] -
        k[1151]*y[IDX_SI] - k[1152]*y[IDX_SOI] - k[1153]*y[IDX_SiH2I] -
        k[1154]*y[IDX_SiH4I] - k[1155]*y[IDX_SiHI] - k[1156]*y[IDX_SiOI] -
        k[1157]*y[IDX_SiSI] - k[1168]*y[IDX_HNCI] - k[1406]*y[IDX_NH2I] -
        k[1433]*y[IDX_NH3I] - k[1452]*y[IDX_NHI] - k[1542]*y[IDX_OHI] -
        k[1543]*y[IDX_OHI] - k[1566]*y[IDX_SiI] - k[2029] - k[2240];
    data[5122] = 0.0 + k[620]*y[IDX_CII] + k[720]*y[IDX_CHII] + k[993]*y[IDX_C2NII] +
        k[994]*y[IDX_C4NII] + k[996]*y[IDX_CNII] + k[997]*y[IDX_COII] -
        k[1003]*y[IDX_HCOII];
    data[5123] = 0.0 + k[797]*y[IDX_CH4II] + k[826]*y[IDX_CH5II] + k[874]*y[IDX_H2ClII] +
        k[875]*y[IDX_HCO2II] + k[876]*y[IDX_HNOII] + k[877]*y[IDX_N2HII] +
        k[878]*y[IDX_O2HII] + k[880]*y[IDX_SiH4II] + k[919]*y[IDX_H2II] +
        k[978]*y[IDX_H2OII] + k[1037]*y[IDX_H3II] + k[1111]*y[IDX_HCNII] +
        k[1353]*y[IDX_NHII] + k[1521]*y[IDX_OHII];
    data[5124] = 0.0 + k[5]*y[IDX_HOCII] + k[941]*y[IDX_COII];
    data[5125] = 0.0 - k[542]*y[IDX_HCOII];
    data[5126] = 0.0 + k[2311] + k[2312] + k[2313] + k[2314];
    data[5127] = 0.0 - k[994]*y[IDX_H2OI];
    data[5128] = 0.0 - k[1016]*y[IDX_H2OI];
    data[5129] = 0.0 - k[999]*y[IDX_H2OI];
    data[5130] = 0.0 - k[1015]*y[IDX_H2OI];
    data[5131] = 0.0 + k[464]*y[IDX_EM] + k[465]*y[IDX_EM];
    data[5132] = 0.0 - k[1008]*y[IDX_H2OI];
    data[5133] = 0.0 + k[1089]*y[IDX_H3OII];
    data[5134] = 0.0 - k[1006]*y[IDX_H2OI];
    data[5135] = 0.0 + k[1091]*y[IDX_H3OII];
    data[5136] = 0.0 + k[1082]*y[IDX_H3OII];
    data[5137] = 0.0 - k[1009]*y[IDX_H2OI];
    data[5138] = 0.0 + k[1769]*y[IDX_HI] + k[1946]*y[IDX_OHI];
    data[5139] = 0.0 + k[1931]*y[IDX_OHI];
    data[5140] = 0.0 + k[1023]*y[IDX_H3II] + k[1081]*y[IDX_H3OII];
    data[5141] = 0.0 + k[1094]*y[IDX_H3OII];
    data[5142] = 0.0 + k[486]*y[IDX_EM] + k[487]*y[IDX_EM] + k[1120]*y[IDX_HCNI];
    data[5143] = 0.0 + k[1092]*y[IDX_H3OII];
    data[5144] = 0.0 + k[1083]*y[IDX_H3OII];
    data[5145] = 0.0 + k[1942]*y[IDX_OHI];
    data[5146] = 0.0 + k[1095]*y[IDX_H3OII];
    data[5147] = 0.0 - k[1014]*y[IDX_H2OI];
    data[5148] = 0.0 + k[1930]*y[IDX_OHI];
    data[5149] = 0.0 - k[992]*y[IDX_H2OI] - k[993]*y[IDX_H2OI];
    data[5150] = 0.0 - k[1004]*y[IDX_H2OI];
    data[5151] = 0.0 + k[1096]*y[IDX_H3OII];
    data[5152] = 0.0 + k[887]*y[IDX_HII] + k[1031]*y[IDX_H3II] + k[1084]*y[IDX_H3OII] +
        k[1466]*y[IDX_OII];
    data[5153] = 0.0 + k[1464]*y[IDX_OII];
    data[5154] = 0.0 - k[799]*y[IDX_H2OI];
    data[5155] = 0.0 + k[1085]*y[IDX_H3OII];
    data[5156] = 0.0 + k[181]*y[IDX_H2OII];
    data[5157] = 0.0 + k[1021]*y[IDX_H2SI];
    data[5158] = 0.0 - k[990]*y[IDX_H2OI];
    data[5159] = 0.0 - k[1012]*y[IDX_H2OI];
    data[5160] = 0.0 - k[995]*y[IDX_H2OI] - k[996]*y[IDX_H2OI];
    data[5161] = 0.0 - k[1011]*y[IDX_H2OI];
    data[5162] = 0.0 + k[184]*y[IDX_H2OII];
    data[5163] = 0.0 - k[1000]*y[IDX_H2OI];
    data[5164] = 0.0 - k[1005]*y[IDX_H2OI];
    data[5165] = 0.0 - k[828]*y[IDX_H2OI];
    data[5166] = 0.0 - k[189]*y[IDX_H2OI] - k[1010]*y[IDX_H2OI];
    data[5167] = 0.0 - k[162]*y[IDX_H2OI] - k[922]*y[IDX_H2OI];
    data[5168] = 0.0 + k[522]*y[IDX_EM] - k[1001]*y[IDX_H2OI];
    data[5169] = 0.0 + k[186]*y[IDX_H2OII] + k[1093]*y[IDX_H3OII];
    data[5170] = 0.0 + k[176]*y[IDX_H2OII] + k[1927]*y[IDX_OHI];
    data[5171] = 0.0 - k[188]*y[IDX_H2OI] - k[1002]*y[IDX_H2OI];
    data[5172] = 0.0 + k[1090]*y[IDX_H3OII];
    data[5173] = 0.0 - k[1013]*y[IDX_H2OI];
    data[5174] = 0.0 - k[187]*y[IDX_H2OI] - k[997]*y[IDX_H2OI];
    data[5175] = 0.0 - k[1007]*y[IDX_H2OI];
    data[5176] = 0.0 - k[261]*y[IDX_H2OI] - k[1356]*y[IDX_H2OI] - k[1357]*y[IDX_H2OI] -
        k[1358]*y[IDX_H2OI] - k[1359]*y[IDX_H2OI];
    data[5177] = 0.0 + k[274]*y[IDX_H2OII] + k[1402]*y[IDX_H3OII] + k[1834]*y[IDX_NOI] +
        k[1836]*y[IDX_OHI];
    data[5178] = 0.0 - k[240]*y[IDX_H2OI];
    data[5179] = 0.0 - k[1378]*y[IDX_H2OI] - k[1379]*y[IDX_H2OI] - k[1380]*y[IDX_H2OI];
    data[5180] = 0.0 + k[177]*y[IDX_H2OII];
    data[5181] = 0.0 - k[312]*y[IDX_H2OI] + k[1464]*y[IDX_C2H4I] + k[1466]*y[IDX_CH3OHI]
        + k[1473]*y[IDX_H2SI];
    data[5182] = 0.0 - k[998]*y[IDX_H2OI];
    data[5183] = 0.0 - k[743]*y[IDX_H2OI];
    data[5184] = 0.0 - k[1414]*y[IDX_H2OI];
    data[5185] = 0.0 + k[179]*y[IDX_H2OII] + k[1021]*y[IDX_SOII] + k[1087]*y[IDX_H3OII] +
        k[1473]*y[IDX_OII] + k[1938]*y[IDX_OHI];
    data[5186] = 0.0 + k[66]*y[IDX_CH2I] + k[86]*y[IDX_CHI] + k[175]*y[IDX_C2I] +
        k[176]*y[IDX_C2H2I] + k[177]*y[IDX_C2HI] + k[178]*y[IDX_H2COI] +
        k[179]*y[IDX_H2SI] + k[180]*y[IDX_HCOI] + k[181]*y[IDX_MgI] +
        k[182]*y[IDX_NOI] + k[183]*y[IDX_O2I] + k[184]*y[IDX_OCSI] +
        k[185]*y[IDX_SI] + k[186]*y[IDX_SiI] + k[274]*y[IDX_NH2I] +
        k[285]*y[IDX_NH3I] - k[980]*y[IDX_H2OI];
    data[5187] = 0.0 - k[332]*y[IDX_H2OI] - k[1523]*y[IDX_H2OI];
    data[5188] = 0.0 - k[2107]*y[IDX_H2OI];
    data[5189] = 0.0 - k[991]*y[IDX_H2OI];
    data[5190] = 0.0 + k[175]*y[IDX_H2OII] + k[1080]*y[IDX_H3OII];
    data[5191] = 0.0 + k[66]*y[IDX_H2OII] + k[759]*y[IDX_H3OII] + k[1634]*y[IDX_O2I] +
        k[1642]*y[IDX_OHI];
    data[5192] = 0.0 - k[1652]*y[IDX_H2OI] + k[1659]*y[IDX_NOI] + k[1661]*y[IDX_O2I] +
        k[1668]*y[IDX_OHI];
    data[5193] = 0.0 - k[718]*y[IDX_H2OI] - k[719]*y[IDX_H2OI] - k[720]*y[IDX_H2OI];
    data[5194] = 0.0 - k[1841]*y[IDX_H2OI] + k[1853]*y[IDX_OHI];
    data[5195] = 0.0 + k[178]*y[IDX_H2OII] + k[1086]*y[IDX_H3OII] + k[1937]*y[IDX_OHI];
    data[5196] = 0.0 + k[1672]*y[IDX_OHI];
    data[5197] = 0.0 + k[180]*y[IDX_H2OII] + k[1941]*y[IDX_OHI];
    data[5198] = 0.0 + k[1088]*y[IDX_H3OII] + k[1120]*y[IDX_CH3OH2II] +
        k[1939]*y[IDX_OHI];
    data[5199] = 0.0 + k[182]*y[IDX_H2OII] + k[1659]*y[IDX_CH3I] + k[1834]*y[IDX_NH2I];
    data[5200] = 0.0 + k[86]*y[IDX_H2OII] + k[845]*y[IDX_H3OII];
    data[5201] = 0.0 + k[285]*y[IDX_H2OII] + k[1428]*y[IDX_H3OII] + k[1944]*y[IDX_OHI];
    data[5202] = 0.0 + k[528]*y[IDX_EM] + k[759]*y[IDX_CH2I] + k[845]*y[IDX_CHI] +
        k[1080]*y[IDX_C2I] + k[1081]*y[IDX_C2H5OHI] + k[1082]*y[IDX_CH3CCHI] +
        k[1083]*y[IDX_CH3CNI] + k[1084]*y[IDX_CH3OHI] + k[1085]*y[IDX_CSI] +
        k[1086]*y[IDX_H2COI] + k[1087]*y[IDX_H2SI] + k[1088]*y[IDX_HCNI] +
        k[1089]*y[IDX_HCOOCH3I] + k[1090]*y[IDX_HNCI] + k[1091]*y[IDX_HS2I] +
        k[1092]*y[IDX_S2I] + k[1093]*y[IDX_SiI] + k[1094]*y[IDX_SiH2I] +
        k[1095]*y[IDX_SiHI] + k[1096]*y[IDX_SiOI] + k[1402]*y[IDX_NH2I] +
        k[1428]*y[IDX_NH3I];
    data[5203] = 0.0 + k[183]*y[IDX_H2OII] + k[1634]*y[IDX_CH2I] + k[1661]*y[IDX_CH3I];
    data[5204] = 0.0 - k[620]*y[IDX_H2OI] - k[621]*y[IDX_H2OI];
    data[5205] = 0.0 + k[185]*y[IDX_H2OII];
    data[5206] = 0.0 + k[1642]*y[IDX_CH2I] + k[1668]*y[IDX_CH3I] + k[1672]*y[IDX_CH4I] +
        k[1734]*y[IDX_H2I] + k[1836]*y[IDX_NH2I] + k[1853]*y[IDX_NHI] +
        k[1927]*y[IDX_C2H2I] + k[1930]*y[IDX_C2H3I] + k[1931]*y[IDX_C2H5I] +
        k[1937]*y[IDX_H2COI] + k[1938]*y[IDX_H2SI] + k[1939]*y[IDX_HCNI] +
        k[1941]*y[IDX_HCOI] + k[1942]*y[IDX_HNOI] + k[1944]*y[IDX_NH3I] +
        k[1946]*y[IDX_O2HI] + k[1947]*y[IDX_OHI] + k[1947]*y[IDX_OHI] +
        k[2125]*y[IDX_HI];
    data[5207] = 0.0 - k[213]*y[IDX_H2OI] - k[1222]*y[IDX_H2OI] - k[1223]*y[IDX_H2OI];
    data[5208] = 0.0 + k[1023]*y[IDX_C2H5OHI] + k[1031]*y[IDX_CH3OHI] -
        k[1043]*y[IDX_H2OI];
    data[5209] = 0.0 - k[121]*y[IDX_H2OI] + k[887]*y[IDX_CH3OHI];
    data[5210] = 0.0 - k[1887]*y[IDX_H2OI];
    data[5211] = 0.0 - k[1003]*y[IDX_H2OI];
    data[5212] = 0.0 - k[4]*y[IDX_H2I] - k[11]*y[IDX_HI] - k[121]*y[IDX_HII] -
        k[162]*y[IDX_H2II] - k[187]*y[IDX_COII] - k[188]*y[IDX_HCNII] -
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
    data[5213] = 0.0 - k[4]*y[IDX_H2OI] + k[1734]*y[IDX_OHI];
    data[5214] = 0.0 + k[464]*y[IDX_C2H5OH2II] + k[465]*y[IDX_C2H5OH2II] +
        k[486]*y[IDX_CH3OH2II] + k[487]*y[IDX_CH3OH2II] + k[522]*y[IDX_H3COII] +
        k[528]*y[IDX_H3OII];
    data[5215] = 0.0 - k[11]*y[IDX_H2OI] - k[1748]*y[IDX_H2OI] + k[1769]*y[IDX_O2HI] +
        k[2125]*y[IDX_OHI];
    data[5216] = 0.0 + k[2343] + k[2344] + k[2345] + k[2346];
    data[5217] = 0.0 + k[1878]*y[IDX_OI];
    data[5218] = 0.0 + k[415] + k[900]*y[IDX_HII] + k[1788]*y[IDX_CI] + k[2037];
    data[5219] = 0.0 + k[551]*y[IDX_EM];
    data[5220] = 0.0 - k[874]*y[IDX_COI];
    data[5221] = 0.0 + k[1919]*y[IDX_OI];
    data[5222] = 0.0 - k[880]*y[IDX_COI];
    data[5223] = 0.0 + k[383] + k[1195]*y[IDX_HeII] + k[1740]*y[IDX_HI] + k[1983];
    data[5224] = 0.0 + k[1877]*y[IDX_OI];
    data[5225] = 0.0 + k[1145]*y[IDX_HCOII];
    data[5226] = 0.0 + k[1146]*y[IDX_HCOII];
    data[5227] = 0.0 + k[1137]*y[IDX_HCOII];
    data[5228] = 0.0 + k[1918]*y[IDX_OI];
    data[5229] = 0.0 - k[879]*y[IDX_COI];
    data[5230] = 0.0 + k[1142]*y[IDX_HCOII];
    data[5231] = 0.0 - k[1719]*y[IDX_COI];
    data[5232] = 0.0 - k[1717]*y[IDX_COI];
    data[5233] = 0.0 + k[1136]*y[IDX_HCOII];
    data[5234] = 0.0 + k[1876]*y[IDX_OI];
    data[5235] = 0.0 + k[1920]*y[IDX_OI];
    data[5236] = 0.0 + k[1894]*y[IDX_OI];
    data[5237] = 0.0 + k[1153]*y[IDX_HCOII];
    data[5238] = 0.0 + k[1157]*y[IDX_HCOII];
    data[5239] = 0.0 + k[1609]*y[IDX_CI] + k[1773]*y[IDX_HI] + k[1864]*y[IDX_O2I] +
        k[1910]*y[IDX_OI];
    data[5240] = 0.0 + k[1150]*y[IDX_HCOII];
    data[5241] = 0.0 + k[1138]*y[IDX_HCOII];
    data[5242] = 0.0 + k[1148]*y[IDX_HCOII];
    data[5243] = 0.0 + k[581]*y[IDX_EM];
    data[5244] = 0.0 + k[638]*y[IDX_CII] + k[1614]*y[IDX_CI];
    data[5245] = 0.0 - k[1716]*y[IDX_COI];
    data[5246] = 0.0 + k[1155]*y[IDX_HCOII];
    data[5247] = 0.0 + k[659]*y[IDX_C2I] + k[704]*y[IDX_CI] - k[881]*y[IDX_COI];
    data[5248] = 0.0 + k[1154]*y[IDX_HCOII];
    data[5249] = 0.0 + k[544]*y[IDX_EM] + k[545]*y[IDX_EM] - k[875]*y[IDX_COI];
    data[5250] = 0.0 + k[645]*y[IDX_CII] + k[1156]*y[IDX_HCOII];
    data[5251] = 0.0 + k[1139]*y[IDX_HCOII];
    data[5252] = 0.0 + k[640]*y[IDX_CII] + k[1152]*y[IDX_HCOII] + k[1616]*y[IDX_CI] +
        k[1699]*y[IDX_CHI];
    data[5253] = 0.0 - k[797]*y[IDX_COI];
    data[5254] = 0.0 + k[1140]*y[IDX_HCOII] + k[1883]*y[IDX_OI] + k[1935]*y[IDX_OHI];
    data[5255] = 0.0 + k[1556]*y[IDX_C2H2I];
    data[5256] = 0.0 + k[648]*y[IDX_HCOI] + k[649]*y[IDX_O2I];
    data[5257] = 0.0 - k[878]*y[IDX_COI];
    data[5258] = 0.0 - k[93]*y[IDX_COI] + k[867]*y[IDX_HCOI] + k[868]*y[IDX_O2I];
    data[5259] = 0.0 - k[877]*y[IDX_COI];
    data[5260] = 0.0 + k[441] + k[636]*y[IDX_CII] + k[736]*y[IDX_CHII] +
        k[751]*y[IDX_CH2II] + k[785]*y[IDX_CH3II] + k[904]*y[IDX_HII] +
        k[1149]*y[IDX_HCOII] + k[1266]*y[IDX_HeII] + k[1313]*y[IDX_NII] +
        k[1318]*y[IDX_N2II] + k[1552]*y[IDX_SII] + k[1565]*y[IDX_SiII] +
        k[1610]*y[IDX_CI] + k[1695]*y[IDX_CHI] + k[1775]*y[IDX_HI] +
        k[1913]*y[IDX_OI] + k[2067];
    data[5261] = 0.0 - k[876]*y[IDX_COI];
    data[5262] = 0.0 - k[826]*y[IDX_COI];
    data[5263] = 0.0 - k[107]*y[IDX_COI] + k[1317]*y[IDX_HCOI] + k[1318]*y[IDX_OCSI];
    data[5264] = 0.0 + k[1147]*y[IDX_HCOII];
    data[5265] = 0.0 - k[160]*y[IDX_COI] - k[919]*y[IDX_COI] + k[925]*y[IDX_HCOI];
    data[5266] = 0.0 + k[523]*y[IDX_EM];
    data[5267] = 0.0 + k[1566]*y[IDX_HCOII] + k[1956]*y[IDX_CO2I] - k[1957]*y[IDX_COI];
    data[5268] = 0.0 + k[1482]*y[IDX_O2II] + k[1556]*y[IDX_SOII] + k[1574]*y[IDX_NOI] +
        k[1868]*y[IDX_OI] + k[1929]*y[IDX_OHI];
    data[5269] = 0.0 - k[1111]*y[IDX_COI] + k[1115]*y[IDX_HCOI];
    data[5270] = 0.0 + k[1168]*y[IDX_HCOII];
    data[5271] = 0.0 + k[1565]*y[IDX_OCSI];
    data[5272] = 0.0 + k[37]*y[IDX_C2I] + k[48]*y[IDX_C2HI] + k[52]*y[IDX_CI] +
        k[64]*y[IDX_CH2I] + k[81]*y[IDX_CH4I] + k[84]*y[IDX_CHI] +
        k[101]*y[IDX_H2COI] + k[102]*y[IDX_H2SI] + k[103]*y[IDX_HCOI] +
        k[104]*y[IDX_NOI] + k[105]*y[IDX_O2I] + k[106]*y[IDX_SI] +
        k[187]*y[IDX_H2OI] + k[192]*y[IDX_HI] + k[200]*y[IDX_HCNI] +
        k[273]*y[IDX_NH2I] + k[283]*y[IDX_NH3I] + k[295]*y[IDX_NHI] +
        k[327]*y[IDX_OI] + k[341]*y[IDX_OHI];
    data[5273] = 0.0 + k[393] + k[616]*y[IDX_CII] + k[714]*y[IDX_CHII] +
        k[741]*y[IDX_CH2II] + k[1210]*y[IDX_HeII] + k[1351]*y[IDX_NHII] +
        k[1470]*y[IDX_OII] + k[1677]*y[IDX_CHI] + k[1744]*y[IDX_HI] +
        k[1808]*y[IDX_NI] + k[1882]*y[IDX_OI] + k[1956]*y[IDX_SiI] + k[2003];
    data[5274] = 0.0 + k[1351]*y[IDX_CO2I] - k[1353]*y[IDX_COI];
    data[5275] = 0.0 + k[273]*y[IDX_COII] + k[1406]*y[IDX_HCOII];
    data[5276] = 0.0 + k[656]*y[IDX_C2I] + k[1161]*y[IDX_HCOI] + k[1482]*y[IDX_C2H2I];
    data[5277] = 0.0 - k[238]*y[IDX_COI] - k[1297]*y[IDX_COI] + k[1303]*y[IDX_HCOI] +
        k[1313]*y[IDX_OCSI];
    data[5278] = 0.0 + k[663]*y[IDX_HCOI];
    data[5279] = 0.0 + k[48]*y[IDX_COII] + k[680]*y[IDX_HCOII] + k[1581]*y[IDX_O2I] +
        k[1875]*y[IDX_OI];
    data[5280] = 0.0 - k[310]*y[IDX_COI] + k[1470]*y[IDX_CO2I] + k[1476]*y[IDX_HCOI];
    data[5281] = 0.0 + k[503]*y[IDX_EM] + k[504]*y[IDX_EM] + k[1158]*y[IDX_HCOI];
    data[5282] = 0.0 + k[741]*y[IDX_CO2I] + k[747]*y[IDX_HCOI] + k[751]*y[IDX_OCSI];
    data[5283] = 0.0 + k[1416]*y[IDX_HCOI];
    data[5284] = 0.0 + k[102]*y[IDX_COII] + k[1143]*y[IDX_HCOII];
    data[5285] = 0.0 - k[978]*y[IDX_COI] + k[984]*y[IDX_HCOI];
    data[5286] = 0.0 - k[1521]*y[IDX_COI] + k[1526]*y[IDX_HCOI];
    data[5287] = 0.0 + k[779]*y[IDX_HCOI] + k[785]*y[IDX_OCSI];
    data[5288] = 0.0 + k[37]*y[IDX_COII] + k[653]*y[IDX_HCOII] + k[656]*y[IDX_O2II] +
        k[659]*y[IDX_SiOII] + k[1572]*y[IDX_O2I] + k[1572]*y[IDX_O2I] +
        k[1867]*y[IDX_OI];
    data[5289] = 0.0 + k[64]*y[IDX_COII] + k[763]*y[IDX_HCOII] + k[1625]*y[IDX_HCOI] +
        k[1634]*y[IDX_O2I] + k[1637]*y[IDX_OI] + k[1638]*y[IDX_OI];
    data[5290] = 0.0 + k[1654]*y[IDX_HCOI] + k[1664]*y[IDX_OI];
    data[5291] = 0.0 + k[714]*y[IDX_CO2I] + k[715]*y[IDX_H2COI] + k[726]*y[IDX_HCOI] +
        k[736]*y[IDX_OCSI];
    data[5292] = 0.0 + k[295]*y[IDX_COII] + k[1452]*y[IDX_HCOII];
    data[5293] = 0.0 + k[974]*y[IDX_H2COI] + k[1163]*y[IDX_HCOI] + k[1552]*y[IDX_OCSI];
    data[5294] = 0.0 + k[1706]*y[IDX_HCOI] + k[1710]*y[IDX_NOI] + k[1712]*y[IDX_O2I] +
        k[1880]*y[IDX_OI];
    data[5295] = 0.0 + k[101]*y[IDX_COII] + k[399] + k[617]*y[IDX_CII] +
        k[715]*y[IDX_CHII] + k[974]*y[IDX_SII] + k[1141]*y[IDX_HCOII] + k[2011]
        + k[2012];
    data[5296] = 0.0 + k[81]*y[IDX_COII];
    data[5297] = 0.0 + k[103]*y[IDX_COII] + k[409] + k[626]*y[IDX_CII] +
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
    data[5298] = 0.0 + k[200]*y[IDX_COII] + k[1124]*y[IDX_HCOII] + k[1890]*y[IDX_OI] +
        k[1940]*y[IDX_OHI];
    data[5299] = 0.0 + k[104]*y[IDX_COII] + k[1574]*y[IDX_C2H2I] + k[1605]*y[IDX_CI] +
        k[1710]*y[IDX_CNI] + k[1783]*y[IDX_HCOI];
    data[5300] = 0.0 + k[84]*y[IDX_COII] + k[849]*y[IDX_HCOII] + k[1677]*y[IDX_CO2I] +
        k[1679]*y[IDX_HCOI] + k[1688]*y[IDX_O2I] + k[1689]*y[IDX_O2I] +
        k[1693]*y[IDX_OI] + k[1695]*y[IDX_OCSI] + k[1699]*y[IDX_SOI];
    data[5301] = 0.0 + k[283]*y[IDX_COII] + k[1433]*y[IDX_HCOII];
    data[5302] = 0.0 + k[105]*y[IDX_COII] + k[634]*y[IDX_CII] + k[649]*y[IDX_C2II] +
        k[868]*y[IDX_CNII] + k[1572]*y[IDX_C2I] + k[1572]*y[IDX_C2I] +
        k[1581]*y[IDX_C2HI] + k[1608]*y[IDX_CI] + k[1634]*y[IDX_CH2I] +
        k[1688]*y[IDX_CHI] + k[1689]*y[IDX_CHI] + k[1712]*y[IDX_CNI] -
        k[1718]*y[IDX_COI] + k[1785]*y[IDX_HCOI] + k[1864]*y[IDX_OCNI];
    data[5303] = 0.0 + k[616]*y[IDX_CO2I] + k[617]*y[IDX_H2COI] + k[626]*y[IDX_HCOI] +
        k[634]*y[IDX_O2I] + k[636]*y[IDX_OCSI] + k[638]*y[IDX_SO2I] +
        k[640]*y[IDX_SOI] + k[645]*y[IDX_SiOI];
    data[5304] = 0.0 + k[106]*y[IDX_COII] + k[1151]*y[IDX_HCOII] + k[1952]*y[IDX_HCOI];
    data[5305] = 0.0 + k[341]*y[IDX_COII] + k[1542]*y[IDX_HCOII] + k[1611]*y[IDX_CI] +
        k[1929]*y[IDX_C2H2I] - k[1934]*y[IDX_COI] + k[1935]*y[IDX_CSI] +
        k[1940]*y[IDX_HCNI] + k[1941]*y[IDX_HCOI];
    data[5306] = 0.0 + k[1808]*y[IDX_CO2I] + k[1811]*y[IDX_HCOI];
    data[5307] = 0.0 + k[1195]*y[IDX_CH2COI] + k[1210]*y[IDX_CO2I] - k[1213]*y[IDX_COI] +
        k[1236]*y[IDX_HCOI] + k[1266]*y[IDX_OCSI];
    data[5308] = 0.0 - k[1037]*y[IDX_COI] - k[1038]*y[IDX_COI];
    data[5309] = 0.0 + k[898]*y[IDX_HCOI] + k[900]*y[IDX_HNCOI] + k[904]*y[IDX_OCSI];
    data[5310] = 0.0 + k[327]*y[IDX_COII] + k[1637]*y[IDX_CH2I] + k[1638]*y[IDX_CH2I] +
        k[1664]*y[IDX_CH3I] + k[1693]*y[IDX_CHI] + k[1867]*y[IDX_C2I] +
        k[1868]*y[IDX_C2H2I] + k[1875]*y[IDX_C2HI] + k[1876]*y[IDX_C2NI] +
        k[1877]*y[IDX_C3NI] + k[1878]*y[IDX_C4NI] + k[1880]*y[IDX_CNI] +
        k[1882]*y[IDX_CO2I] + k[1883]*y[IDX_CSI] + k[1890]*y[IDX_HCNI] +
        k[1893]*y[IDX_HCOI] + k[1894]*y[IDX_HCSI] + k[1910]*y[IDX_OCNI] +
        k[1913]*y[IDX_OCSI] + k[1918]*y[IDX_SiC2I] + k[1919]*y[IDX_SiC3I] +
        k[1920]*y[IDX_SiCI] + k[2104]*y[IDX_CI];
    data[5311] = 0.0 + k[52]*y[IDX_COII] + k[694]*y[IDX_HCOII] + k[704]*y[IDX_SiOII] -
        k[1591]*y[IDX_COI] + k[1594]*y[IDX_HCOI] + k[1605]*y[IDX_NOI] +
        k[1608]*y[IDX_O2I] + k[1609]*y[IDX_OCNI] + k[1610]*y[IDX_OCSI] +
        k[1611]*y[IDX_OHI] + k[1614]*y[IDX_SO2I] + k[1616]*y[IDX_SOI] +
        k[1788]*y[IDX_HNCOI] + k[2104]*y[IDX_OI];
    data[5312] = 0.0 + k[542]*y[IDX_EM] + k[653]*y[IDX_C2I] + k[680]*y[IDX_C2HI] +
        k[694]*y[IDX_CI] + k[763]*y[IDX_CH2I] + k[849]*y[IDX_CHI] +
        k[1003]*y[IDX_H2OI] + k[1124]*y[IDX_HCNI] + k[1136]*y[IDX_C2H5OHI] +
        k[1137]*y[IDX_CH3CCHI] + k[1138]*y[IDX_CH3CNI] + k[1139]*y[IDX_CH3OHI] +
        k[1140]*y[IDX_CSI] + k[1141]*y[IDX_H2COI] + k[1142]*y[IDX_H2CSI] +
        k[1143]*y[IDX_H2SI] + k[1144]*y[IDX_HCOI] + k[1145]*y[IDX_HCOOCH3I] +
        k[1146]*y[IDX_HS2I] + k[1147]*y[IDX_HSI] + k[1148]*y[IDX_NSI] +
        k[1149]*y[IDX_OCSI] + k[1150]*y[IDX_S2I] + k[1151]*y[IDX_SI] +
        k[1152]*y[IDX_SOI] + k[1153]*y[IDX_SiH2I] + k[1154]*y[IDX_SiH4I] +
        k[1155]*y[IDX_SiHI] + k[1156]*y[IDX_SiOI] + k[1157]*y[IDX_SiSI] +
        k[1168]*y[IDX_HNCI] + k[1406]*y[IDX_NH2I] + k[1433]*y[IDX_NH3I] +
        k[1452]*y[IDX_NHI] + k[1542]*y[IDX_OHI] + k[1566]*y[IDX_SiI];
    data[5313] = 0.0 + k[187]*y[IDX_COII] + k[1003]*y[IDX_HCOII];
    data[5314] = 0.0 - k[93]*y[IDX_CNII] - k[107]*y[IDX_N2II] - k[160]*y[IDX_H2II] -
        k[238]*y[IDX_NII] - k[310]*y[IDX_OII] - k[357] - k[394] -
        k[797]*y[IDX_CH4II] - k[826]*y[IDX_CH5II] - k[874]*y[IDX_H2ClII] -
        k[875]*y[IDX_HCO2II] - k[876]*y[IDX_HNOII] - k[877]*y[IDX_N2HII] -
        k[878]*y[IDX_O2HII] - k[879]*y[IDX_SO2II] - k[880]*y[IDX_SiH4II] -
        k[881]*y[IDX_SiOII] - k[919]*y[IDX_H2II] - k[978]*y[IDX_H2OII] -
        k[1037]*y[IDX_H3II] - k[1038]*y[IDX_H3II] - k[1111]*y[IDX_HCNII] -
        k[1213]*y[IDX_HeII] - k[1297]*y[IDX_NII] - k[1353]*y[IDX_NHII] -
        k[1521]*y[IDX_OHII] - k[1591]*y[IDX_CI] - k[1716]*y[IDX_HNOI] -
        k[1717]*y[IDX_NO2I] - k[1718]*y[IDX_O2I] - k[1719]*y[IDX_O2HI] -
        k[1745]*y[IDX_HI] - k[1934]*y[IDX_OHI] - k[1957]*y[IDX_SiI] - k[2004] -
        k[2169] - k[2207] - k[2296];
    data[5315] = 0.0 + k[503]*y[IDX_H2COII] + k[504]*y[IDX_H2COII] + k[523]*y[IDX_H3COII]
        + k[542]*y[IDX_HCOII] + k[544]*y[IDX_HCO2II] + k[545]*y[IDX_HCO2II] +
        k[551]*y[IDX_HOCII] + k[581]*y[IDX_OCSII];
    data[5316] = 0.0 + k[192]*y[IDX_COII] + k[1740]*y[IDX_CH2COI] + k[1744]*y[IDX_CO2I] -
        k[1745]*y[IDX_COI] + k[1751]*y[IDX_HCOI] + k[1773]*y[IDX_OCNI] +
        k[1775]*y[IDX_OCSI];
    data[5317] = 0.0 + k[511]*y[IDX_EM];
    data[5318] = 0.0 + k[405] + k[896]*y[IDX_HII] + k[2023];
    data[5319] = 0.0 - k[948]*y[IDX_H2I];
    data[5320] = 0.0 - k[951]*y[IDX_H2I];
    data[5321] = 0.0 - k[5]*y[IDX_H2I] + k[5]*y[IDX_H2I];
    data[5322] = 0.0 + k[1040]*y[IDX_H3II] - k[1720]*y[IDX_H2I];
    data[5323] = 0.0 - k[944]*y[IDX_H2I];
    data[5324] = 0.0 + k[1593]*y[IDX_CI] + k[1746]*y[IDX_HI] + k[1885]*y[IDX_OI];
    data[5325] = 0.0 + k[600]*y[IDX_EM];
    data[5326] = 0.0 + k[598]*y[IDX_EM] - k[963]*y[IDX_H2I];
    data[5327] = 0.0 + k[1047]*y[IDX_H3II];
    data[5328] = 0.0 + k[1049]*y[IDX_H3II] + k[1787]*y[IDX_HI];
    data[5329] = 0.0 + k[1052]*y[IDX_H3II];
    data[5330] = 0.0 + k[1197]*y[IDX_HeII] + k[1197]*y[IDX_HeII];
    data[5331] = 0.0 + k[526]*y[IDX_EM];
    data[5332] = 0.0 - k[962]*y[IDX_H2I];
    data[5333] = 0.0 + k[400] + k[1042]*y[IDX_H3II] + k[1219]*y[IDX_HeII] + k[2015];
    data[5334] = 0.0 + k[1770]*y[IDX_HI];
    data[5335] = 0.0 + k[1059]*y[IDX_H3II];
    data[5336] = 0.0 + k[371] + k[1968];
    data[5337] = 0.0 + k[1024]*y[IDX_H3II];
    data[5338] = 0.0 + k[1026]*y[IDX_H3II];
    data[5339] = 0.0 + k[899]*y[IDX_HII] + k[1048]*y[IDX_H3II] + k[1753]*y[IDX_HI];
    data[5340] = 0.0 + k[643]*y[IDX_CII] + k[905]*y[IDX_HII] + k[1072]*y[IDX_H3II] +
        k[1278]*y[IDX_HeII] + k[1922]*y[IDX_OI];
    data[5341] = 0.0 + k[597]*y[IDX_EM] + k[1515]*y[IDX_OI] - k[2120]*y[IDX_H2I];
    data[5342] = 0.0 + k[1077]*y[IDX_H3II];
    data[5343] = 0.0 + k[490]*y[IDX_EM] + k[969]*y[IDX_H2COI];
    data[5344] = 0.0 + k[1067]*y[IDX_H3II];
    data[5345] = 0.0 + k[906]*y[IDX_HII] + k[1073]*y[IDX_H3II] + k[1280]*y[IDX_HeII] +
        k[2087];
    data[5346] = 0.0 + k[1030]*y[IDX_H3II];
    data[5347] = 0.0 + k[1061]*y[IDX_H3II];
    data[5348] = 0.0 + k[593]*y[IDX_EM];
    data[5349] = 0.0 + k[1069]*y[IDX_H3II];
    data[5350] = 0.0 + k[901]*y[IDX_HII] + k[1051]*y[IDX_H3II] + k[1756]*y[IDX_HI];
    data[5351] = 0.0 + k[908]*y[IDX_HII] + k[1075]*y[IDX_H3II];
    data[5352] = 0.0 - k[964]*y[IDX_H2I];
    data[5353] = 0.0 + k[1108]*y[IDX_HI] - k[2119]*y[IDX_H2I];
    data[5354] = 0.0 + k[454] + k[671]*y[IDX_C2H2II] + k[836]*y[IDX_CH5II] +
        k[907]*y[IDX_HII] + k[1074]*y[IDX_H3II] + k[1282]*y[IDX_HeII] +
        k[1282]*y[IDX_HeII] + k[1283]*y[IDX_HeII] + k[2088] + k[2090];
    data[5355] = 0.0 + k[882]*y[IDX_HII] + k[1181]*y[IDX_HeII] + k[1738]*y[IDX_HI];
    data[5356] = 0.0 - k[943]*y[IDX_H2I];
    data[5357] = 0.0 + k[1076]*y[IDX_H3II];
    data[5358] = 0.0 + k[388] + k[888]*y[IDX_HII] + k[889]*y[IDX_HII] + k[889]*y[IDX_HII]
        + k[1031]*y[IDX_H3II] + k[1032]*y[IDX_H3II] + k[1990];
    data[5359] = 0.0 + k[533]*y[IDX_EM] + k[535]*y[IDX_EM] + k[1104]*y[IDX_HI];
    data[5360] = 0.0 + k[788]*y[IDX_CH3II] + k[1070]*y[IDX_H3II];
    data[5361] = 0.0 + k[370] + k[774]*y[IDX_CH3II] + k[883]*y[IDX_HII] +
        k[910]*y[IDX_H2II] + k[910]*y[IDX_H2II] + k[1183]*y[IDX_HeII] +
        k[1184]*y[IDX_HeII] + k[1871]*y[IDX_OI] + k[1967];
    data[5362] = 0.0 - k[939]*y[IDX_H2I] + k[1101]*y[IDX_HI] + k[1993];
    data[5363] = 0.0 + k[1039]*y[IDX_H3II];
    data[5364] = 0.0 + k[1054]*y[IDX_H3II];
    data[5365] = 0.0 - k[935]*y[IDX_H2I];
    data[5366] = 0.0 - k[959]*y[IDX_H2I];
    data[5367] = 0.0 - k[940]*y[IDX_H2I];
    data[5368] = 0.0 + k[1065]*y[IDX_H3II];
    data[5369] = 0.0 - k[946]*y[IDX_H2I] + k[1103]*y[IDX_HI] + k[1335]*y[IDX_NI] +
        k[1499]*y[IDX_OI];
    data[5370] = 0.0 + k[493]*y[IDX_EM] + k[494]*y[IDX_EM] + k[497]*y[IDX_EM] +
        k[497]*y[IDX_EM] + k[836]*y[IDX_SiH4I] + k[1102]*y[IDX_HI] +
        k[1494]*y[IDX_OI];
    data[5371] = 0.0 + k[815]*y[IDX_CH4I] - k[953]*y[IDX_H2I] + k[1316]*y[IDX_H2SI];
    data[5372] = 0.0 + k[780]*y[IDX_CH3II] + k[902]*y[IDX_HII] + k[1053]*y[IDX_H3II] -
        k[1727]*y[IDX_H2I] + k[1758]*y[IDX_HI];
    data[5373] = 0.0 + k[153]*y[IDX_C2I] + k[154]*y[IDX_C2H2I] + k[155]*y[IDX_C2HI] +
        k[156]*y[IDX_CH2I] + k[157]*y[IDX_CH4I] + k[158]*y[IDX_CHI] +
        k[159]*y[IDX_CNI] + k[160]*y[IDX_COI] + k[161]*y[IDX_H2COI] +
        k[162]*y[IDX_H2OI] + k[163]*y[IDX_H2SI] + k[164]*y[IDX_HCNI] +
        k[165]*y[IDX_HCOI] + k[166]*y[IDX_NH2I] + k[167]*y[IDX_NH3I] +
        k[168]*y[IDX_NHI] + k[169]*y[IDX_NOI] + k[170]*y[IDX_O2I] +
        k[171]*y[IDX_OHI] + k[193]*y[IDX_HI] + k[910]*y[IDX_C2H4I] +
        k[910]*y[IDX_C2H4I] + k[914]*y[IDX_CH4I] - k[920]*y[IDX_H2I] +
        k[921]*y[IDX_H2COI] + k[923]*y[IDX_H2SI] + k[924]*y[IDX_H2SI] +
        k[924]*y[IDX_H2SI];
    data[5374] = 0.0 + k[523]*y[IDX_EM];
    data[5375] = 0.0 + k[670]*y[IDX_C2H2II] + k[1071]*y[IDX_H3II] + k[1575]*y[IDX_C2H2I];
    data[5376] = 0.0 + k[154]*y[IDX_H2II] + k[1178]*y[IDX_HeII] + k[1575]*y[IDX_SiI] +
        k[1737]*y[IDX_HI];
    data[5377] = 0.0 - k[947]*y[IDX_H2I];
    data[5378] = 0.0 + k[1050]*y[IDX_H3II];
    data[5379] = 0.0 - k[2118]*y[IDX_H2I];
    data[5380] = 0.0 - k[941]*y[IDX_H2I] - k[942]*y[IDX_H2I];
    data[5381] = 0.0 + k[814]*y[IDX_CH4I] - k[949]*y[IDX_H2I] + k[1105]*y[IDX_HI] +
        k[1176]*y[IDX_H2SI] - k[2116]*y[IDX_H2I];
    data[5382] = 0.0 + k[1036]*y[IDX_H3II];
    data[5383] = 0.0 - k[954]*y[IDX_H2I] - k[955]*y[IDX_H2I] + k[1357]*y[IDX_H2OI];
    data[5384] = 0.0 + k[166]*y[IDX_H2II] + k[729]*y[IDX_CHII] + k[1056]*y[IDX_H3II] +
        k[1252]*y[IDX_HeII] - k[1729]*y[IDX_H2I] + k[1760]*y[IDX_HI];
    data[5385] = 0.0 - k[952]*y[IDX_H2I] + k[1294]*y[IDX_CH4I] + k[1306]*y[IDX_NH3I];
    data[5386] = 0.0 - k[956]*y[IDX_H2I];
    data[5387] = 0.0 - k[936]*y[IDX_H2I];
    data[5388] = 0.0 + k[572]*y[IDX_EM];
    data[5389] = 0.0 + k[155]*y[IDX_H2II] + k[706]*y[IDX_CHII] + k[884]*y[IDX_HII] +
        k[1025]*y[IDX_H3II] - k[1721]*y[IDX_H2I];
    data[5390] = 0.0 - k[958]*y[IDX_H2I];
    data[5391] = 0.0 + k[503]*y[IDX_EM];
    data[5392] = 0.0 + k[478]*y[IDX_EM] + k[746]*y[IDX_H2SI] - k[938]*y[IDX_H2I] +
        k[1099]*y[IDX_HI] + k[1978];
    data[5393] = 0.0 - k[957]*y[IDX_H2I] + k[1508]*y[IDX_OI];
    data[5394] = 0.0 + k[163]*y[IDX_H2II] + k[404] + k[722]*y[IDX_CHII] +
        k[746]*y[IDX_CH2II] + k[778]*y[IDX_CH3II] + k[894]*y[IDX_HII] +
        k[895]*y[IDX_HII] + k[923]*y[IDX_H2II] + k[924]*y[IDX_H2II] +
        k[924]*y[IDX_H2II] + k[1044]*y[IDX_H3II] + k[1176]*y[IDX_HSII] +
        k[1227]*y[IDX_HeII] + k[1316]*y[IDX_N2II] + k[1551]*y[IDX_SII] +
        k[1749]*y[IDX_HI] + k[2022];
    data[5395] = 0.0 + k[512]*y[IDX_EM] - k[945]*y[IDX_H2I] + k[1334]*y[IDX_NI] +
        k[1497]*y[IDX_OI];
    data[5396] = 0.0 - k[960]*y[IDX_H2I];
    data[5397] = 0.0 + k[482]*y[IDX_EM] + k[688]*y[IDX_CI] + k[774]*y[IDX_C2H4I] +
        k[778]*y[IDX_H2SI] + k[780]*y[IDX_HSI] + k[784]*y[IDX_OI] +
        k[786]*y[IDX_OHI] + k[787]*y[IDX_SI] + k[788]*y[IDX_SOI] +
        k[839]*y[IDX_CHI] + k[1100]*y[IDX_HI] + k[1446]*y[IDX_NHI] + k[1984] -
        k[2114]*y[IDX_H2I];
    data[5398] = 0.0 + k[1055]*y[IDX_H3II];
    data[5399] = 0.0 + k[670]*y[IDX_SiI] + k[671]*y[IDX_SiH4I] + k[1328]*y[IDX_NI];
    data[5400] = 0.0 + k[153]*y[IDX_H2II] + k[1022]*y[IDX_H3II];
    data[5401] = 0.0 + k[156]*y[IDX_H2II] + k[707]*y[IDX_CHII] + k[885]*y[IDX_HII] +
        k[1028]*y[IDX_H3II] + k[1192]*y[IDX_HeII] + k[1618]*y[IDX_CH2I] +
        k[1618]*y[IDX_CH2I] + k[1632]*y[IDX_O2I] + k[1637]*y[IDX_OI] +
        k[1644]*y[IDX_SI] - k[1723]*y[IDX_H2I] + k[1739]*y[IDX_HI];
    data[5402] = 0.0 + k[386] + k[609]*y[IDX_CII] + k[1029]*y[IDX_H3II] +
        k[1196]*y[IDX_HeII] + k[1647]*y[IDX_CH3I] + k[1647]*y[IDX_CH3I] +
        k[1664]*y[IDX_OI] + k[1667]*y[IDX_OHI] - k[1724]*y[IDX_H2I] +
        k[1741]*y[IDX_HI] + k[1805]*y[IDX_NI] + k[1988];
    data[5403] = 0.0 + k[706]*y[IDX_C2HI] + k[707]*y[IDX_CH2I] + k[711]*y[IDX_CH4I] +
        k[712]*y[IDX_CHI] + k[720]*y[IDX_H2OI] + k[722]*y[IDX_H2SI] +
        k[723]*y[IDX_HCNI] + k[729]*y[IDX_NH2I] + k[731]*y[IDX_NHI] +
        k[738]*y[IDX_OHI] - k[937]*y[IDX_H2I] + k[1098]*y[IDX_HI];
    data[5404] = 0.0 + k[168]*y[IDX_H2II] + k[731]*y[IDX_CHII] + k[1058]*y[IDX_H3II] +
        k[1446]*y[IDX_CH3II] - k[1730]*y[IDX_H2I] + k[1762]*y[IDX_HI] +
        k[1843]*y[IDX_NHI] + k[1843]*y[IDX_NHI];
    data[5405] = 0.0 + k[822]*y[IDX_CH4I] - k[961]*y[IDX_H2I] + k[1551]*y[IDX_H2SI] -
        k[2117]*y[IDX_H2I];
    data[5406] = 0.0 + k[159]*y[IDX_H2II] + k[1035]*y[IDX_H3II] - k[1726]*y[IDX_H2I];
    data[5407] = 0.0 + k[161]*y[IDX_H2II] + k[399] + k[892]*y[IDX_HII] +
        k[893]*y[IDX_HII] + k[921]*y[IDX_H2II] + k[969]*y[IDX_CH3OH2II] +
        k[1041]*y[IDX_H3II] + k[1216]*y[IDX_HeII] + k[1747]*y[IDX_HI] + k[2011];
    data[5408] = 0.0 + k[157]*y[IDX_H2II] + k[390] + k[614]*y[IDX_CII] +
        k[711]*y[IDX_CHII] + k[814]*y[IDX_HSII] + k[815]*y[IDX_N2II] +
        k[822]*y[IDX_SII] + k[890]*y[IDX_HII] + k[914]*y[IDX_H2II] +
        k[1033]*y[IDX_H3II] + k[1202]*y[IDX_HeII] + k[1203]*y[IDX_HeII] +
        k[1294]*y[IDX_NII] + k[1742]*y[IDX_HI] + k[1995] + k[1998];
    data[5409] = 0.0 + k[165]*y[IDX_H2II] + k[897]*y[IDX_HII] + k[1046]*y[IDX_H3II] +
        k[1751]*y[IDX_HI] + k[1780]*y[IDX_HCOI] + k[1780]*y[IDX_HCOI];
    data[5410] = 0.0 + k[164]*y[IDX_H2II] + k[723]*y[IDX_CHII] + k[1045]*y[IDX_H3II] +
        k[1750]*y[IDX_HI];
    data[5411] = 0.0 + k[169]*y[IDX_H2II] + k[1060]*y[IDX_H3II];
    data[5412] = 0.0 - k[2]*y[IDX_H2I] + k[2]*y[IDX_H2I] + k[158]*y[IDX_H2II] +
        k[712]*y[IDX_CHII] + k[839]*y[IDX_CH3II] + k[1034]*y[IDX_H3II] -
        k[1725]*y[IDX_H2I] + k[1743]*y[IDX_HI] - k[2115]*y[IDX_H2I];
    data[5413] = 0.0 + k[167]*y[IDX_H2II] + k[428] + k[630]*y[IDX_CII] +
        k[1057]*y[IDX_H3II] + k[1254]*y[IDX_HeII] + k[1306]*y[IDX_NII] +
        k[1761]*y[IDX_HI] + k[2053];
    data[5414] = 0.0 + k[529]*y[IDX_EM] + k[530]*y[IDX_EM] + k[692]*y[IDX_CI];
    data[5415] = 0.0 - k[6]*y[IDX_H2I] + k[6]*y[IDX_H2I] + k[170]*y[IDX_H2II] +
        k[1062]*y[IDX_H3II] + k[1632]*y[IDX_CH2I] - k[1731]*y[IDX_H2I] -
        k[1732]*y[IDX_H2I];
    data[5416] = 0.0 + k[609]*y[IDX_CH3I] + k[614]*y[IDX_CH4I] + k[630]*y[IDX_NH3I] +
        k[643]*y[IDX_SiH2I] - k[934]*y[IDX_H2I] - k[2112]*y[IDX_H2I];
    data[5417] = 0.0 + k[787]*y[IDX_CH3II] + k[1068]*y[IDX_H3II] + k[1644]*y[IDX_CH2I] -
        k[1735]*y[IDX_H2I];
    data[5418] = 0.0 - k[7]*y[IDX_H2I] + k[7]*y[IDX_H2I] + k[171]*y[IDX_H2II] +
        k[738]*y[IDX_CHII] + k[786]*y[IDX_CH3II] + k[1066]*y[IDX_H3II] +
        k[1667]*y[IDX_CH3I] - k[1734]*y[IDX_H2I] + k[1776]*y[IDX_HI];
    data[5419] = 0.0 + k[1328]*y[IDX_C2H2II] + k[1334]*y[IDX_H2OII] +
        k[1335]*y[IDX_H2SII] - k[1728]*y[IDX_H2I] + k[1805]*y[IDX_CH3I];
    data[5420] = 0.0 - k[172]*y[IDX_H2I] - k[950]*y[IDX_H2I] + k[1178]*y[IDX_C2H2I] +
        k[1181]*y[IDX_C2H3I] + k[1183]*y[IDX_C2H4I] + k[1184]*y[IDX_C2H4I] +
        k[1192]*y[IDX_CH2I] + k[1196]*y[IDX_CH3I] + k[1197]*y[IDX_CH3CCHI] +
        k[1197]*y[IDX_CH3CCHI] + k[1202]*y[IDX_CH4I] + k[1203]*y[IDX_CH4I] +
        k[1216]*y[IDX_H2COI] + k[1219]*y[IDX_H2CSI] + k[1227]*y[IDX_H2SI] +
        k[1252]*y[IDX_NH2I] + k[1254]*y[IDX_NH3I] + k[1278]*y[IDX_SiH2I] +
        k[1280]*y[IDX_SiH3I] + k[1282]*y[IDX_SiH4I] + k[1282]*y[IDX_SiH4I] +
        k[1283]*y[IDX_SiH4I];
    data[5421] = 0.0 + k[519]*y[IDX_EM] + k[1022]*y[IDX_C2I] + k[1024]*y[IDX_C2H5OHI] +
        k[1025]*y[IDX_C2HI] + k[1026]*y[IDX_C2NI] + k[1027]*y[IDX_CI] +
        k[1028]*y[IDX_CH2I] + k[1029]*y[IDX_CH3I] + k[1030]*y[IDX_CH3CNI] +
        k[1031]*y[IDX_CH3OHI] + k[1032]*y[IDX_CH3OHI] + k[1033]*y[IDX_CH4I] +
        k[1034]*y[IDX_CHI] + k[1035]*y[IDX_CNI] + k[1036]*y[IDX_CO2I] +
        k[1037]*y[IDX_COI] + k[1038]*y[IDX_COI] + k[1039]*y[IDX_CSI] +
        k[1040]*y[IDX_ClI] + k[1041]*y[IDX_H2COI] + k[1042]*y[IDX_H2CSI] +
        k[1043]*y[IDX_H2OI] + k[1044]*y[IDX_H2SI] + k[1045]*y[IDX_HCNI] +
        k[1046]*y[IDX_HCOI] + k[1047]*y[IDX_HCOOCH3I] + k[1048]*y[IDX_HCSI] +
        k[1049]*y[IDX_HClI] + k[1050]*y[IDX_HNCI] + k[1051]*y[IDX_HNOI] +
        k[1052]*y[IDX_HS2I] + k[1053]*y[IDX_HSI] + k[1054]*y[IDX_MgI] +
        k[1055]*y[IDX_N2I] + k[1056]*y[IDX_NH2I] + k[1057]*y[IDX_NH3I] +
        k[1058]*y[IDX_NHI] + k[1059]*y[IDX_NO2I] + k[1060]*y[IDX_NOI] +
        k[1061]*y[IDX_NSI] + k[1062]*y[IDX_O2I] + k[1064]*y[IDX_OI] +
        k[1065]*y[IDX_OCSI] + k[1066]*y[IDX_OHI] + k[1067]*y[IDX_S2I] +
        k[1068]*y[IDX_SI] + k[1069]*y[IDX_SO2I] + k[1070]*y[IDX_SOI] +
        k[1071]*y[IDX_SiI] + k[1072]*y[IDX_SiH2I] + k[1073]*y[IDX_SiH3I] +
        k[1074]*y[IDX_SiH4I] + k[1075]*y[IDX_SiHI] + k[1076]*y[IDX_SiOI] +
        k[1077]*y[IDX_SiSI] + k[2026];
    data[5422] = 0.0 + k[882]*y[IDX_C2H3I] + k[883]*y[IDX_C2H4I] + k[884]*y[IDX_C2HI] +
        k[885]*y[IDX_CH2I] + k[888]*y[IDX_CH3OHI] + k[889]*y[IDX_CH3OHI] +
        k[889]*y[IDX_CH3OHI] + k[890]*y[IDX_CH4I] + k[892]*y[IDX_H2COI] +
        k[893]*y[IDX_H2COI] + k[894]*y[IDX_H2SI] + k[895]*y[IDX_H2SI] +
        k[896]*y[IDX_H2SiOI] + k[897]*y[IDX_HCOI] + k[899]*y[IDX_HCSI] +
        k[901]*y[IDX_HNOI] + k[902]*y[IDX_HSI] + k[905]*y[IDX_SiH2I] +
        k[906]*y[IDX_SiH3I] + k[907]*y[IDX_SiH4I] + k[908]*y[IDX_SiHI];
    data[5423] = 0.0 + k[784]*y[IDX_CH3II] + k[1064]*y[IDX_H3II] + k[1494]*y[IDX_CH5II] +
        k[1497]*y[IDX_H2OII] + k[1499]*y[IDX_H2SII] + k[1508]*y[IDX_NH3II] +
        k[1515]*y[IDX_SiH3II] + k[1637]*y[IDX_CH2I] + k[1664]*y[IDX_CH3I] -
        k[1733]*y[IDX_H2I] + k[1871]*y[IDX_C2H4I] + k[1885]*y[IDX_H2CNI] +
        k[1922]*y[IDX_SiH2I];
    data[5424] = 0.0 + k[688]*y[IDX_CH3II] + k[692]*y[IDX_H3OII] + k[1027]*y[IDX_H3II] +
        k[1593]*y[IDX_H2CNI] - k[1722]*y[IDX_H2I] - k[2113]*y[IDX_H2I];
    data[5425] = 0.0 - k[4]*y[IDX_H2I] + k[4]*y[IDX_H2I] + k[162]*y[IDX_H2II] +
        k[720]*y[IDX_CHII] + k[1043]*y[IDX_H3II] + k[1357]*y[IDX_NHII] +
        k[1748]*y[IDX_HI];
    data[5426] = 0.0 + k[160]*y[IDX_H2II] + k[1037]*y[IDX_H3II] + k[1038]*y[IDX_H3II];
    data[5427] = 0.0 - k[2]*y[IDX_CHI] + k[2]*y[IDX_CHI] - k[3]*y[IDX_H2I] -
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
        k[2118]*y[IDX_SiII] - k[2119]*y[IDX_SiHII] - k[2120]*y[IDX_SiH3II] +
        (-H2dissociation);
    data[5428] = 0.0 - k[8]*y[IDX_H2I] + k[478]*y[IDX_CH2II] + k[482]*y[IDX_CH3II] +
        k[490]*y[IDX_CH3OH2II] + k[493]*y[IDX_CH5II] + k[494]*y[IDX_CH5II] +
        k[497]*y[IDX_CH5II] + k[497]*y[IDX_CH5II] + k[503]*y[IDX_H2COII] +
        k[511]*y[IDX_H2NOII] + k[512]*y[IDX_H2OII] + k[519]*y[IDX_H3II] +
        k[523]*y[IDX_H3COII] + k[526]*y[IDX_H3CSII] + k[529]*y[IDX_H3OII] +
        k[530]*y[IDX_H3OII] + k[533]*y[IDX_H3SII] + k[535]*y[IDX_H3SII] +
        k[572]*y[IDX_NH4II] + k[593]*y[IDX_SiH2II] + k[597]*y[IDX_SiH3II] +
        k[598]*y[IDX_SiH4II] + k[600]*y[IDX_SiH5II];
    data[5429] = 0.0 - k[10]*y[IDX_H2I] + k[193]*y[IDX_H2II] + k[1098]*y[IDX_CHII] +
        k[1099]*y[IDX_CH2II] + k[1100]*y[IDX_CH3II] + k[1101]*y[IDX_CH4II] +
        k[1102]*y[IDX_CH5II] + k[1103]*y[IDX_H2SII] + k[1104]*y[IDX_H3SII] +
        k[1105]*y[IDX_HSII] + k[1108]*y[IDX_SiHII] + k[1737]*y[IDX_C2H2I] +
        k[1738]*y[IDX_C2H3I] + k[1739]*y[IDX_CH2I] + k[1741]*y[IDX_CH3I] +
        k[1742]*y[IDX_CH4I] + k[1743]*y[IDX_CHI] + k[1746]*y[IDX_H2CNI] +
        k[1747]*y[IDX_H2COI] + k[1748]*y[IDX_H2OI] + k[1749]*y[IDX_H2SI] +
        k[1750]*y[IDX_HCNI] + k[1751]*y[IDX_HCOI] + k[1753]*y[IDX_HCSI] +
        k[1756]*y[IDX_HNOI] + k[1758]*y[IDX_HSI] + k[1760]*y[IDX_NH2I] +
        k[1761]*y[IDX_NH3I] + k[1762]*y[IDX_NHI] + k[1770]*y[IDX_O2HI] +
        k[1776]*y[IDX_OHI] + k[1787]*y[IDX_HClI] + (H2formation);
    data[5430] = 0.0 - k[510]*y[IDX_EM] - k[511]*y[IDX_EM];
    data[5431] = 0.0 - k[548]*y[IDX_EM];
    data[5432] = 0.0 - k[563]*y[IDX_EM];
    data[5433] = 0.0 - k[550]*y[IDX_EM];
    data[5434] = 0.0 - k[551]*y[IDX_EM];
    data[5435] = 0.0 - k[557]*y[IDX_EM];
    data[5436] = 0.0 - k[590]*y[IDX_EM] - k[591]*y[IDX_EM];
    data[5437] = 0.0 + k[358] + k[397] + k[2008];
    data[5438] = 0.0 - k[2134]*y[IDX_EM];
    data[5439] = 0.0 - k[470]*y[IDX_EM] - k[471]*y[IDX_EM];
    data[5440] = 0.0 - k[475]*y[IDX_EM] - k[476]*y[IDX_EM];
    data[5441] = 0.0 - k[600]*y[IDX_EM] - k[601]*y[IDX_EM];
    data[5442] = 0.0 - k[508]*y[IDX_EM] - k[509]*y[IDX_EM];
    data[5443] = 0.0 - k[536]*y[IDX_EM] - k[537]*y[IDX_EM];
    data[5444] = 0.0 - k[598]*y[IDX_EM] - k[599]*y[IDX_EM];
    data[5445] = 0.0 - k[464]*y[IDX_EM] - k[465]*y[IDX_EM] - k[466]*y[IDX_EM] -
        k[467]*y[IDX_EM];
    data[5446] = 0.0 - k[472]*y[IDX_EM];
    data[5447] = 0.0 - k[474]*y[IDX_EM];
    data[5448] = 0.0 - k[558]*y[IDX_EM] - k[559]*y[IDX_EM] - k[560]*y[IDX_EM];
    data[5449] = 0.0 - k[588]*y[IDX_EM] - k[589]*y[IDX_EM];
    data[5450] = 0.0 - k[473]*y[IDX_EM];
    data[5451] = 0.0 - k[517]*y[IDX_EM] - k[518]*y[IDX_EM];
    data[5452] = 0.0 + k[2035];
    data[5453] = 0.0 - k[552]*y[IDX_EM] - k[553]*y[IDX_EM];
    data[5454] = 0.0 + k[2041];
    data[5455] = 0.0 - k[576]*y[IDX_EM];
    data[5456] = 0.0 - k[526]*y[IDX_EM] - k[527]*y[IDX_EM];
    data[5457] = 0.0 - k[585]*y[IDX_EM] - k[586]*y[IDX_EM];
    data[5458] = 0.0 - k[506]*y[IDX_EM] - k[507]*y[IDX_EM] - k[2137]*y[IDX_EM];
    data[5459] = 0.0 - k[561]*y[IDX_EM] - k[562]*y[IDX_EM];
    data[5460] = 0.0 - k[605]*y[IDX_EM];
    data[5461] = 0.0 - k[555]*y[IDX_EM] - k[556]*y[IDX_EM];
    data[5462] = 0.0 - k[484]*y[IDX_EM] - k[485]*y[IDX_EM];
    data[5463] = 0.0 - k[583]*y[IDX_EM];
    data[5464] = 0.0 - k[587]*y[IDX_EM];
    data[5465] = 0.0 + k[2046];
    data[5466] = 0.0 + k[412] + k[2033];
    data[5467] = 0.0 + k[2083];
    data[5468] = 0.0 - k[596]*y[IDX_EM] - k[597]*y[IDX_EM];
    data[5469] = 0.0 - k[486]*y[IDX_EM] - k[487]*y[IDX_EM] - k[488]*y[IDX_EM] -
        k[489]*y[IDX_EM] - k[490]*y[IDX_EM];
    data[5470] = 0.0 + k[2071];
    data[5471] = 0.0 + k[2086];
    data[5472] = 0.0 - k[579]*y[IDX_EM] - k[580]*y[IDX_EM] - k[581]*y[IDX_EM];
    data[5473] = 0.0 - k[593]*y[IDX_EM] - k[594]*y[IDX_EM] - k[595]*y[IDX_EM];
    data[5474] = 0.0 - k[602]*y[IDX_EM];
    data[5475] = 0.0 - k[603]*y[IDX_EM] - k[604]*y[IDX_EM];
    data[5476] = 0.0 - k[592]*y[IDX_EM];
    data[5477] = 0.0 - k[468]*y[IDX_EM] - k[469]*y[IDX_EM];
    data[5478] = 0.0 - k[543]*y[IDX_EM] - k[544]*y[IDX_EM] - k[545]*y[IDX_EM];
    data[5479] = 0.0 - k[500]*y[IDX_EM];
    data[5480] = 0.0 - k[546]*y[IDX_EM] - k[547]*y[IDX_EM];
    data[5481] = 0.0 + k[2094];
    data[5482] = 0.0 + k[1991];
    data[5483] = 0.0 - k[532]*y[IDX_EM] - k[533]*y[IDX_EM] - k[534]*y[IDX_EM] -
        k[535]*y[IDX_EM];
    data[5484] = 0.0 + k[447] + k[2076];
    data[5485] = 0.0 - k[491]*y[IDX_EM] - k[492]*y[IDX_EM];
    data[5486] = 0.0 + k[395] + k[2006];
    data[5487] = 0.0 + k[420] + k[2044];
    data[5488] = 0.0 - k[2140]*y[IDX_EM];
    data[5489] = 0.0 - k[584]*y[IDX_EM];
    data[5490] = 0.0 - k[458]*y[IDX_EM];
    data[5491] = 0.0 - k[578]*y[IDX_EM];
    data[5492] = 0.0 - k[498]*y[IDX_EM];
    data[5493] = 0.0 - k[565]*y[IDX_EM] - k[566]*y[IDX_EM];
    data[5494] = 0.0 + k[440] + k[2066];
    data[5495] = 0.0 - k[515]*y[IDX_EM] - k[516]*y[IDX_EM] - k[2138]*y[IDX_EM];
    data[5496] = 0.0 - k[549]*y[IDX_EM];
    data[5497] = 0.0 - k[493]*y[IDX_EM] - k[494]*y[IDX_EM] - k[495]*y[IDX_EM] -
        k[496]*y[IDX_EM] - k[497]*y[IDX_EM];
    data[5498] = 0.0 - k[564]*y[IDX_EM];
    data[5499] = 0.0 - k[539]*y[IDX_EM] - k[540]*y[IDX_EM] - k[541]*y[IDX_EM];
    data[5500] = 0.0 - k[501]*y[IDX_EM];
    data[5501] = 0.0 - k[521]*y[IDX_EM] - k[522]*y[IDX_EM] - k[523]*y[IDX_EM] -
        k[524]*y[IDX_EM] - k[525]*y[IDX_EM];
    data[5502] = 0.0 + k[448] + k[2077];
    data[5503] = 0.0 + k[367] + k[1964];
    data[5504] = 0.0 - k[538]*y[IDX_EM];
    data[5505] = 0.0 - k[2144]*y[IDX_EM];
    data[5506] = 0.0 - k[499]*y[IDX_EM];
    data[5507] = 0.0 - k[554]*y[IDX_EM];
    data[5508] = 0.0 - k[567]*y[IDX_EM];
    data[5509] = 0.0 + k[424] + k[2049];
    data[5510] = 0.0 - k[577]*y[IDX_EM];
    data[5511] = 0.0 - k[2141]*y[IDX_EM];
    data[5512] = 0.0 - k[568]*y[IDX_EM] - k[569]*y[IDX_EM];
    data[5513] = 0.0 - k[459]*y[IDX_EM] - k[460]*y[IDX_EM];
    data[5514] = 0.0 - k[572]*y[IDX_EM] - k[573]*y[IDX_EM] - k[574]*y[IDX_EM];
    data[5515] = 0.0 + k[374] + k[1971];
    data[5516] = 0.0 - k[2142]*y[IDX_EM];
    data[5517] = 0.0 - k[575]*y[IDX_EM];
    data[5518] = 0.0 - k[502]*y[IDX_EM] - k[503]*y[IDX_EM] - k[504]*y[IDX_EM] -
        k[505]*y[IDX_EM] - k[2136]*y[IDX_EM];
    data[5519] = 0.0 - k[478]*y[IDX_EM] - k[479]*y[IDX_EM] - k[480]*y[IDX_EM];
    data[5520] = 0.0 - k[570]*y[IDX_EM] - k[571]*y[IDX_EM];
    data[5521] = 0.0 + k[403] + k[2020];
    data[5522] = 0.0 - k[512]*y[IDX_EM] - k[513]*y[IDX_EM] - k[514]*y[IDX_EM];
    data[5523] = 0.0 - k[582]*y[IDX_EM];
    data[5524] = 0.0 - k[481]*y[IDX_EM] - k[482]*y[IDX_EM] - k[483]*y[IDX_EM] -
        k[2133]*y[IDX_EM];
    data[5525] = 0.0 - k[461]*y[IDX_EM] - k[462]*y[IDX_EM] - k[463]*y[IDX_EM];
    data[5526] = 0.0 + k[1961];
    data[5527] = 0.0 + k[381] + k[1981];
    data[5528] = 0.0 + k[385] + k[1987];
    data[5529] = 0.0 - k[477]*y[IDX_EM];
    data[5530] = 0.0 + k[430] + k[2055];
    data[5531] = 0.0 - k[2143]*y[IDX_EM];
    data[5532] = 0.0 + k[2013] + k[2014];
    data[5533] = 0.0 + k[1997];
    data[5534] = 0.0 + k[410] + k[2031];
    data[5535] = 0.0 + k[432] + k[2057];
    data[5536] = 0.0 + k[0]*y[IDX_OI] + k[2000];
    data[5537] = 0.0 + k[427] + k[2052];
    data[5538] = 0.0 - k[528]*y[IDX_EM] - k[529]*y[IDX_EM] - k[530]*y[IDX_EM] -
        k[531]*y[IDX_EM];
    data[5539] = 0.0 + k[435] + k[2061];
    data[5540] = 0.0 - k[2132]*y[IDX_EM];
    data[5541] = 0.0 + k[444] + k[2073];
    data[5542] = 0.0 + k[2070];
    data[5543] = 0.0 + k[364] + k[422];
    data[5544] = 0.0 - k[2139]*y[IDX_EM];
    data[5545] = 0.0 + k[363] + k[419];
    data[5546] = 0.0 - k[519]*y[IDX_EM] - k[520]*y[IDX_EM];
    data[5547] = 0.0 - k[2135]*y[IDX_EM];
    data[5548] = 0.0 + k[0]*y[IDX_CHI] + k[365] + k[438];
    data[5549] = 0.0 + k[356] + k[379] + k[1976];
    data[5550] = 0.0 - k[542]*y[IDX_EM];
    data[5551] = 0.0 + k[2017];
    data[5552] = 0.0 + k[357];
    data[5553] = 0.0 - k[8]*y[IDX_EM] + k[8]*y[IDX_EM] + k[359] + k[360];
    data[5554] = 0.0 - k[8]*y[IDX_H2I] + k[8]*y[IDX_H2I] - k[458]*y[IDX_C2II] -
        k[459]*y[IDX_C2HII] - k[460]*y[IDX_C2HII] - k[461]*y[IDX_C2H2II] -
        k[462]*y[IDX_C2H2II] - k[463]*y[IDX_C2H2II] - k[464]*y[IDX_C2H5OH2II] -
        k[465]*y[IDX_C2H5OH2II] - k[466]*y[IDX_C2H5OH2II] -
        k[467]*y[IDX_C2H5OH2II] - k[468]*y[IDX_C2NII] - k[469]*y[IDX_C2NII] -
        k[470]*y[IDX_C2N2II] - k[471]*y[IDX_C2N2II] - k[472]*y[IDX_C2NHII] -
        k[473]*y[IDX_C3II] - k[474]*y[IDX_C3H5II] - k[475]*y[IDX_C4NII] -
        k[476]*y[IDX_C4NII] - k[477]*y[IDX_CHII] - k[478]*y[IDX_CH2II] -
        k[479]*y[IDX_CH2II] - k[480]*y[IDX_CH2II] - k[481]*y[IDX_CH3II] -
        k[482]*y[IDX_CH3II] - k[483]*y[IDX_CH3II] - k[484]*y[IDX_CH3CNHII] -
        k[485]*y[IDX_CH3CNHII] - k[486]*y[IDX_CH3OH2II] - k[487]*y[IDX_CH3OH2II]
        - k[488]*y[IDX_CH3OH2II] - k[489]*y[IDX_CH3OH2II] -
        k[490]*y[IDX_CH3OH2II] - k[491]*y[IDX_CH4II] - k[492]*y[IDX_CH4II] -
        k[493]*y[IDX_CH5II] - k[494]*y[IDX_CH5II] - k[495]*y[IDX_CH5II] -
        k[496]*y[IDX_CH5II] - k[497]*y[IDX_CH5II] - k[498]*y[IDX_CNII] -
        k[499]*y[IDX_COII] - k[500]*y[IDX_CSII] - k[501]*y[IDX_H2II] -
        k[502]*y[IDX_H2COII] - k[503]*y[IDX_H2COII] - k[504]*y[IDX_H2COII] -
        k[505]*y[IDX_H2COII] - k[506]*y[IDX_H2CSII] - k[507]*y[IDX_H2CSII] -
        k[508]*y[IDX_H2ClII] - k[509]*y[IDX_H2ClII] - k[510]*y[IDX_H2NOII] -
        k[511]*y[IDX_H2NOII] - k[512]*y[IDX_H2OII] - k[513]*y[IDX_H2OII] -
        k[514]*y[IDX_H2OII] - k[515]*y[IDX_H2SII] - k[516]*y[IDX_H2SII] -
        k[517]*y[IDX_H2S2II] - k[518]*y[IDX_H2S2II] - k[519]*y[IDX_H3II] -
        k[520]*y[IDX_H3II] - k[521]*y[IDX_H3COII] - k[522]*y[IDX_H3COII] -
        k[523]*y[IDX_H3COII] - k[524]*y[IDX_H3COII] - k[525]*y[IDX_H3COII] -
        k[526]*y[IDX_H3CSII] - k[527]*y[IDX_H3CSII] - k[528]*y[IDX_H3OII] -
        k[529]*y[IDX_H3OII] - k[530]*y[IDX_H3OII] - k[531]*y[IDX_H3OII] -
        k[532]*y[IDX_H3SII] - k[533]*y[IDX_H3SII] - k[534]*y[IDX_H3SII] -
        k[535]*y[IDX_H3SII] - k[536]*y[IDX_H5C2O2II] - k[537]*y[IDX_H5C2O2II] -
        k[538]*y[IDX_HCNII] - k[539]*y[IDX_HCNHII] - k[540]*y[IDX_HCNHII] -
        k[541]*y[IDX_HCNHII] - k[542]*y[IDX_HCOII] - k[543]*y[IDX_HCO2II] -
        k[544]*y[IDX_HCO2II] - k[545]*y[IDX_HCO2II] - k[546]*y[IDX_HCSII] -
        k[547]*y[IDX_HCSII] - k[548]*y[IDX_HClII] - k[549]*y[IDX_HNOII] -
        k[550]*y[IDX_HNSII] - k[551]*y[IDX_HOCII] - k[552]*y[IDX_HOCSII] -
        k[553]*y[IDX_HOCSII] - k[554]*y[IDX_HSII] - k[555]*y[IDX_HS2II] -
        k[556]*y[IDX_HS2II] - k[557]*y[IDX_HSOII] - k[558]*y[IDX_HSO2II] -
        k[559]*y[IDX_HSO2II] - k[560]*y[IDX_HSO2II] - k[561]*y[IDX_HSiSII] -
        k[562]*y[IDX_HSiSII] - k[563]*y[IDX_HeHII] - k[564]*y[IDX_N2II] -
        k[565]*y[IDX_N2HII] - k[566]*y[IDX_N2HII] - k[567]*y[IDX_NHII] -
        k[568]*y[IDX_NH2II] - k[569]*y[IDX_NH2II] - k[570]*y[IDX_NH3II] -
        k[571]*y[IDX_NH3II] - k[572]*y[IDX_NH4II] - k[573]*y[IDX_NH4II] -
        k[574]*y[IDX_NH4II] - k[575]*y[IDX_NOII] - k[576]*y[IDX_NSII] -
        k[577]*y[IDX_O2II] - k[578]*y[IDX_O2HII] - k[579]*y[IDX_OCSII] -
        k[580]*y[IDX_OCSII] - k[581]*y[IDX_OCSII] - k[582]*y[IDX_OHII] -
        k[583]*y[IDX_S2II] - k[584]*y[IDX_SOII] - k[585]*y[IDX_SO2II] -
        k[586]*y[IDX_SO2II] - k[587]*y[IDX_SiCII] - k[588]*y[IDX_SiC2II] -
        k[589]*y[IDX_SiC2II] - k[590]*y[IDX_SiC3II] - k[591]*y[IDX_SiC3II] -
        k[592]*y[IDX_SiHII] - k[593]*y[IDX_SiH2II] - k[594]*y[IDX_SiH2II] -
        k[595]*y[IDX_SiH2II] - k[596]*y[IDX_SiH3II] - k[597]*y[IDX_SiH3II] -
        k[598]*y[IDX_SiH4II] - k[599]*y[IDX_SiH4II] - k[600]*y[IDX_SiH5II] -
        k[601]*y[IDX_SiH5II] - k[602]*y[IDX_SiOII] - k[603]*y[IDX_SiOHII] -
        k[604]*y[IDX_SiOHII] - k[605]*y[IDX_SiSII] - k[2132]*y[IDX_CII] -
        k[2133]*y[IDX_CH3II] - k[2134]*y[IDX_ClII] - k[2135]*y[IDX_HII] -
        k[2136]*y[IDX_H2COII] - k[2137]*y[IDX_H2CSII] - k[2138]*y[IDX_H2SII] -
        k[2139]*y[IDX_HeII] - k[2140]*y[IDX_MgII] - k[2141]*y[IDX_NII] -
        k[2142]*y[IDX_OII] - k[2143]*y[IDX_SII] - k[2144]*y[IDX_SiII] - k[2302];
    data[5555] = 0.0 + k[362] + k[406];
    data[5556] = 0.0 + k[1585]*y[IDX_CI] + k[1797]*y[IDX_NI];
    data[5557] = 0.0 + k[122]*y[IDX_HII] + k[1225]*y[IDX_HeII];
    data[5558] = 0.0 + k[510]*y[IDX_EM] + k[2273];
    data[5559] = 0.0 + k[1228]*y[IDX_HeII] + k[2024] + k[2024];
    data[5560] = 0.0 + k[548]*y[IDX_EM] + k[948]*y[IDX_H2I];
    data[5561] = 0.0 + k[563]*y[IDX_EM] - k[1106]*y[IDX_HI];
    data[5562] = 0.0 + k[550]*y[IDX_EM] + k[2291];
    data[5563] = 0.0 + k[551]*y[IDX_EM];
    data[5564] = 0.0 + k[557]*y[IDX_EM] + k[2283];
    data[5565] = 0.0 + k[1799]*y[IDX_NI];
    data[5566] = 0.0 + k[109]*y[IDX_HII] + k[1720]*y[IDX_H2I];
    data[5567] = 0.0 - k[108]*y[IDX_HI] + k[944]*y[IDX_H2I];
    data[5568] = 0.0 + k[398] - k[1746]*y[IDX_HI] + k[2010];
    data[5569] = 0.0 - k[1097]*y[IDX_HI];
    data[5570] = 0.0 + k[601]*y[IDX_EM] + k[2193];
    data[5571] = 0.0 + k[508]*y[IDX_EM] + k[508]*y[IDX_EM] + k[509]*y[IDX_EM] + k[2198];
    data[5572] = 0.0 + k[537]*y[IDX_EM] + k[2167];
    data[5573] = 0.0 + k[145]*y[IDX_HII];
    data[5574] = 0.0 + k[599]*y[IDX_EM] + k[963]*y[IDX_H2I];
    data[5575] = 0.0 + k[464]*y[IDX_EM] + k[466]*y[IDX_EM] + k[467]*y[IDX_EM] + k[2151];
    data[5576] = 0.0 + k[472]*y[IDX_EM];
    data[5577] = 0.0 + k[474]*y[IDX_EM] + k[2148];
    data[5578] = 0.0 - k[1740]*y[IDX_HI];
    data[5579] = 0.0 + k[558]*y[IDX_EM] + k[559]*y[IDX_EM] + k[2295];
    data[5580] = 0.0 + k[1119]*y[IDX_HCNI];
    data[5581] = 0.0 + k[517]*y[IDX_EM];
    data[5582] = 0.0 + k[126]*y[IDX_HII] + k[413] + k[1241]*y[IDX_HeII] -
        k[1787]*y[IDX_HI] + k[2034];
    data[5583] = 0.0 + k[553]*y[IDX_EM] + k[2285];
    data[5584] = 0.0 + k[127]*y[IDX_HII] + k[1248]*y[IDX_HeII];
    data[5585] = 0.0 + k[526]*y[IDX_EM] + k[527]*y[IDX_EM] + k[2282];
    data[5586] = 0.0 + k[144]*y[IDX_HII];
    data[5587] = 0.0 + k[962]*y[IDX_H2I] - k[1107]*y[IDX_HI];
    data[5588] = 0.0 + k[120]*y[IDX_HII];
    data[5589] = 0.0 + k[506]*y[IDX_EM] + k[506]*y[IDX_EM] + k[507]*y[IDX_EM];
    data[5590] = 0.0 + k[625]*y[IDX_CII];
    data[5591] = 0.0 + k[562]*y[IDX_EM] + k[2191];
    data[5592] = 0.0 - k[1109]*y[IDX_HI];
    data[5593] = 0.0 + k[556]*y[IDX_EM];
    data[5594] = 0.0 + k[437] - k[1769]*y[IDX_HI] - k[1770]*y[IDX_HI] - k[1771]*y[IDX_HI]
        + k[2063];
    data[5595] = 0.0 + k[484]*y[IDX_EM];
    data[5596] = 0.0 - k[1763]*y[IDX_HI];
    data[5597] = 0.0 + k[1583]*y[IDX_CI];
    data[5598] = 0.0 + k[113]*y[IDX_HII];
    data[5599] = 0.0 - k[1759]*y[IDX_HI];
    data[5600] = 0.0 + k[146]*y[IDX_HII];
    data[5601] = 0.0 + k[1239]*y[IDX_HeII] - k[1753]*y[IDX_HI] + k[1895]*y[IDX_OI];
    data[5602] = 0.0 + k[147]*y[IDX_HII] + k[452] + k[1279]*y[IDX_HeII] +
        k[1923]*y[IDX_OI] + k[1923]*y[IDX_OI] + k[2084];
    data[5603] = 0.0 + k[596]*y[IDX_EM];
    data[5604] = 0.0 + k[152]*y[IDX_HII];
    data[5605] = 0.0 + k[486]*y[IDX_EM] + k[488]*y[IDX_EM] + k[489]*y[IDX_EM] +
        k[490]*y[IDX_EM] + k[2152];
    data[5606] = 0.0 - k[1772]*y[IDX_HI] - k[1773]*y[IDX_HI] - k[1774]*y[IDX_HI];
    data[5607] = 0.0 + k[139]*y[IDX_HII] - k[1777]*y[IDX_HI];
    data[5608] = 0.0 + k[148]*y[IDX_HII] + k[453] + k[1281]*y[IDX_HeII] +
        k[1924]*y[IDX_OI] + k[2085];
    data[5609] = 0.0 + k[134]*y[IDX_HII] - k[1766]*y[IDX_HI] - k[1767]*y[IDX_HI];
    data[5610] = 0.0 + k[594]*y[IDX_EM] + k[594]*y[IDX_EM] + k[595]*y[IDX_EM] +
        k[1514]*y[IDX_OI] + k[1568]*y[IDX_SI];
    data[5611] = 0.0 + k[141]*y[IDX_HII];
    data[5612] = 0.0 + k[416] + k[1245]*y[IDX_HeII] - k[1755]*y[IDX_HI] -
        k[1756]*y[IDX_HI] - k[1757]*y[IDX_HI] + k[1896]*y[IDX_OI] + k[2038];
    data[5613] = 0.0 + k[150]*y[IDX_HII] + k[455] + k[644]*y[IDX_CII] +
        k[1284]*y[IDX_HeII] + k[1569]*y[IDX_SII] + k[1617]*y[IDX_CI] +
        k[1926]*y[IDX_OI] + k[2091];
    data[5614] = 0.0 + k[964]*y[IDX_H2I];
    data[5615] = 0.0 + k[604]*y[IDX_EM];
    data[5616] = 0.0 + k[592]*y[IDX_EM] + k[703]*y[IDX_CI] - k[1108]*y[IDX_HI] +
        k[1513]*y[IDX_OI] + k[2082];
    data[5617] = 0.0 + k[149]*y[IDX_HII] + k[1283]*y[IDX_HeII] + k[2089] + k[2090];
    data[5618] = 0.0 + k[369] + k[1182]*y[IDX_HeII] + k[1582]*y[IDX_CI] -
        k[1738]*y[IDX_HI] + k[1869]*y[IDX_OI] + k[1966];
    data[5619] = 0.0 + k[543]*y[IDX_EM] + k[544]*y[IDX_EM] + k[2247];
    data[5620] = 0.0 + k[943]*y[IDX_H2I];
    data[5621] = 0.0 + k[547]*y[IDX_EM] + k[1501]*y[IDX_OI];
    data[5622] = 0.0 + k[151]*y[IDX_HII];
    data[5623] = 0.0 + k[1289]*y[IDX_NII] + k[1291]*y[IDX_NII] + k[1292]*y[IDX_NII] +
        k[1483]*y[IDX_O2II] + k[1991];
    data[5624] = 0.0 + k[532]*y[IDX_EM] + k[534]*y[IDX_EM] + k[534]*y[IDX_EM] +
        k[535]*y[IDX_EM] - k[1104]*y[IDX_HI] + k[1553]*y[IDX_SI] + k[2278];
    data[5625] = 0.0 + k[142]*y[IDX_HII] + k[1700]*y[IDX_CHI] - k[1778]*y[IDX_HI] -
        k[1779]*y[IDX_HI] + k[1949]*y[IDX_OHI];
    data[5626] = 0.0 + k[883]*y[IDX_HII] + k[1183]*y[IDX_HeII] + k[1675]*y[IDX_CHI];
    data[5627] = 0.0 + k[491]*y[IDX_EM] + k[491]*y[IDX_EM] + k[492]*y[IDX_EM] +
        k[939]*y[IDX_H2I] - k[1101]*y[IDX_HI] + k[1994];
    data[5628] = 0.0 + k[118]*y[IDX_HII] + k[1936]*y[IDX_OHI];
    data[5629] = 0.0 + k[129]*y[IDX_HII] + k[834]*y[IDX_CH5II] + k[1054]*y[IDX_H3II];
    data[5630] = 0.0 + k[837]*y[IDX_CHI] + k[935]*y[IDX_H2I] + k[1394]*y[IDX_NH2I] +
        k[1445]*y[IDX_NHI];
    data[5631] = 0.0 + k[578]*y[IDX_EM];
    data[5632] = 0.0 - k[191]*y[IDX_HI] + k[866]*y[IDX_HCNI] + k[940]*y[IDX_H2I];
    data[5633] = 0.0 + k[565]*y[IDX_EM] + k[2290];
    data[5634] = 0.0 + k[137]*y[IDX_HII] + k[1695]*y[IDX_CHI] - k[1775]*y[IDX_HI];
    data[5635] = 0.0 + k[515]*y[IDX_EM] + k[516]*y[IDX_EM] + k[516]*y[IDX_EM] +
        k[691]*y[IDX_CI] + k[946]*y[IDX_H2I] - k[1103]*y[IDX_HI];
    data[5636] = 0.0 + k[549]*y[IDX_EM];
    data[5637] = 0.0 + k[493]*y[IDX_EM] + k[495]*y[IDX_EM] + k[495]*y[IDX_EM] +
        k[496]*y[IDX_EM] + k[834]*y[IDX_MgI] - k[1102]*y[IDX_HI] + k[2248];
    data[5638] = 0.0 + k[816]*y[IDX_CH4I] + k[953]*y[IDX_H2I] + k[1314]*y[IDX_H2COI] +
        k[1315]*y[IDX_H2SI];
    data[5639] = 0.0 + k[128]*y[IDX_HII] + k[418] + k[628]*y[IDX_CII] +
        k[1249]*y[IDX_HeII] + k[1595]*y[IDX_CI] + k[1727]*y[IDX_H2I] -
        k[1758]*y[IDX_HI] + k[1816]*y[IDX_NI] + k[1900]*y[IDX_OI] +
        k[1953]*y[IDX_SI] + k[2043];
    data[5640] = 0.0 + k[539]*y[IDX_EM] + k[539]*y[IDX_EM] + k[540]*y[IDX_EM] +
        k[541]*y[IDX_EM] + k[2289];
    data[5641] = 0.0 - k[193]*y[IDX_HI] + k[501]*y[IDX_EM] + k[501]*y[IDX_EM] +
        k[909]*y[IDX_C2I] + k[911]*y[IDX_C2HI] + k[912]*y[IDX_CI] +
        k[913]*y[IDX_CH2I] + k[914]*y[IDX_CH4I] + k[915]*y[IDX_CH4I] +
        k[916]*y[IDX_CHI] + k[917]*y[IDX_CNI] + k[918]*y[IDX_CO2I] +
        k[919]*y[IDX_COI] + k[920]*y[IDX_H2I] + k[921]*y[IDX_H2COI] +
        k[922]*y[IDX_H2OI] + k[923]*y[IDX_H2SI] + k[926]*y[IDX_HeI] +
        k[927]*y[IDX_N2I] + k[928]*y[IDX_NI] + k[929]*y[IDX_NHI] +
        k[930]*y[IDX_NOI] + k[931]*y[IDX_O2I] + k[932]*y[IDX_OI] +
        k[933]*y[IDX_OHI] + k[2009];
    data[5642] = 0.0 + k[523]*y[IDX_EM] + k[524]*y[IDX_EM] + k[525]*y[IDX_EM] +
        k[525]*y[IDX_EM];
    data[5643] = 0.0 + k[143]*y[IDX_HII] + k[1950]*y[IDX_OHI];
    data[5644] = 0.0 + k[111]*y[IDX_HII] + k[368] + k[1179]*y[IDX_HeII] +
        k[1482]*y[IDX_O2II] + k[1570]*y[IDX_C2I] + k[1574]*y[IDX_NOI] +
        k[1674]*y[IDX_CHI] + k[1701]*y[IDX_CNI] - k[1737]*y[IDX_HI] +
        k[1928]*y[IDX_OHI] + k[1965];
    data[5645] = 0.0 - k[194]*y[IDX_HI] + k[538]*y[IDX_EM] + k[947]*y[IDX_H2I];
    data[5646] = 0.0 + k[414] + k[627]*y[IDX_CII] + k[1242]*y[IDX_HeII] +
        k[1243]*y[IDX_HeII] + k[1579]*y[IDX_C2HI] + k[1707]*y[IDX_CNI] -
        k[1754]*y[IDX_HI] + k[1754]*y[IDX_HI] + k[2036];
    data[5647] = 0.0 + k[684]*y[IDX_C2HI] + k[862]*y[IDX_CHI] + k[1013]*y[IDX_H2OI] +
        k[1549]*y[IDX_OHI] - k[2126]*y[IDX_HI];
    data[5648] = 0.0 - k[192]*y[IDX_HI] + k[941]*y[IDX_H2I] + k[942]*y[IDX_H2I];
    data[5649] = 0.0 + k[554]*y[IDX_EM] + k[697]*y[IDX_CI] + k[949]*y[IDX_H2I] -
        k[1105]*y[IDX_HI] + k[1336]*y[IDX_NI] + k[1504]*y[IDX_OI] + k[2039];
    data[5650] = 0.0 + k[918]*y[IDX_H2II] - k[1744]*y[IDX_HI];
    data[5651] = 0.0 + k[567]*y[IDX_EM] + k[955]*y[IDX_H2I] + k[1337]*y[IDX_NI] +
        k[1346]*y[IDX_C2I] + k[1373]*y[IDX_SI];
    data[5652] = 0.0 + k[130]*y[IDX_HII] + k[425] + k[629]*y[IDX_CII] +
        k[1253]*y[IDX_HeII] + k[1394]*y[IDX_C2II] + k[1599]*y[IDX_CI] +
        k[1600]*y[IDX_CI] + k[1729]*y[IDX_H2I] - k[1760]*y[IDX_HI] +
        k[1835]*y[IDX_NOI] + k[1902]*y[IDX_OI] + k[2050];
    data[5653] = 0.0 + k[972]*y[IDX_H2COI] + k[1482]*y[IDX_C2H2I] + k[1483]*y[IDX_CH3OHI];
    data[5654] = 0.0 + k[852]*y[IDX_CHI] + k[952]*y[IDX_H2I] + k[1289]*y[IDX_CH3OHI] +
        k[1291]*y[IDX_CH3OHI] + k[1292]*y[IDX_CH3OHI] + k[1293]*y[IDX_CH4I] +
        k[1294]*y[IDX_CH4I] + k[1295]*y[IDX_CH4I] + k[1295]*y[IDX_CH4I] +
        k[1302]*y[IDX_H2SI] + k[1308]*y[IDX_NHI];
    data[5655] = 0.0 + k[568]*y[IDX_EM] + k[568]*y[IDX_EM] + k[569]*y[IDX_EM] +
        k[956]*y[IDX_H2I] + k[1338]*y[IDX_NI] + k[1392]*y[IDX_SI] +
        k[1507]*y[IDX_OI];
    data[5656] = 0.0 + k[459]*y[IDX_EM] + k[685]*y[IDX_CI] + k[936]*y[IDX_H2I] +
        k[1326]*y[IDX_NI] + k[1963];
    data[5657] = 0.0 + k[573]*y[IDX_EM] + k[573]*y[IDX_EM] + k[574]*y[IDX_EM] + k[2288];
    data[5658] = 0.0 + k[112]*y[IDX_HII] + k[373] + k[607]*y[IDX_CII] +
        k[684]*y[IDX_SiII] + k[911]*y[IDX_H2II] + k[1186]*y[IDX_HeII] +
        k[1578]*y[IDX_HCNI] + k[1579]*y[IDX_HNCI] + k[1721]*y[IDX_H2I] +
        k[1795]*y[IDX_NI] + k[1970];
    data[5659] = 0.0 - k[196]*y[IDX_HI] + k[857]*y[IDX_CHI] + k[958]*y[IDX_H2I] +
        k[1457]*y[IDX_NHI] + k[1480]*y[IDX_OHI];
    data[5660] = 0.0 + k[504]*y[IDX_EM] + k[504]*y[IDX_EM] + k[505]*y[IDX_EM];
    data[5661] = 0.0 + k[479]*y[IDX_EM] + k[479]*y[IDX_EM] + k[480]*y[IDX_EM] +
        k[687]*y[IDX_CI] + k[743]*y[IDX_H2OI] + k[744]*y[IDX_H2SI] +
        k[746]*y[IDX_H2SI] + k[750]*y[IDX_OI] + k[753]*y[IDX_SI] +
        k[938]*y[IDX_H2I] - k[1099]*y[IDX_HI] + k[1331]*y[IDX_NI] + k[1979];
    data[5662] = 0.0 + k[570]*y[IDX_EM] + k[571]*y[IDX_EM] + k[571]*y[IDX_EM] +
        k[957]*y[IDX_H2I];
    data[5663] = 0.0 + k[123]*y[IDX_HII] + k[622]*y[IDX_CII] + k[744]*y[IDX_CH2II] +
        k[746]*y[IDX_CH2II] + k[895]*y[IDX_HII] + k[923]*y[IDX_H2II] +
        k[1226]*y[IDX_HeII] + k[1302]*y[IDX_NII] + k[1315]*y[IDX_N2II] +
        k[1550]*y[IDX_SII] - k[1749]*y[IDX_HI] + k[2021];
    data[5664] = 0.0 + k[513]*y[IDX_EM] + k[513]*y[IDX_EM] + k[514]*y[IDX_EM] +
        k[945]*y[IDX_H2I] + k[988]*y[IDX_SI] + k[1333]*y[IDX_NI] + k[2016];
    data[5665] = 0.0 + k[582]*y[IDX_EM] + k[960]*y[IDX_H2I] + k[1340]*y[IDX_NI] +
        k[1511]*y[IDX_OI] + k[1534]*y[IDX_SI] + k[2068];
    data[5666] = 0.0 + k[481]*y[IDX_EM] + k[483]*y[IDX_EM] + k[483]*y[IDX_EM] +
        k[783]*y[IDX_OI] - k[1100]*y[IDX_HI] + k[1985];
    data[5667] = 0.0 + k[927]*y[IDX_H2II];
    data[5668] = 0.0 + k[461]*y[IDX_EM] + k[461]*y[IDX_EM] + k[462]*y[IDX_EM] +
        k[806]*y[IDX_CH4I] + k[1329]*y[IDX_NI];
    data[5669] = 0.0 + k[110]*y[IDX_HII] + k[705]*y[IDX_CHII] + k[909]*y[IDX_H2II] +
        k[1346]*y[IDX_NHII] + k[1570]*y[IDX_C2H2I] + k[1571]*y[IDX_HCNI] -
        k[1736]*y[IDX_HI];
    data[5670] = 0.0 + k[114]*y[IDX_HII] + k[382] + k[608]*y[IDX_CII] + k[772]*y[IDX_SII]
        + k[913]*y[IDX_H2II] + k[1193]*y[IDX_HeII] + k[1586]*y[IDX_CI] +
        k[1619]*y[IDX_CH2I] + k[1619]*y[IDX_CH2I] + k[1619]*y[IDX_CH2I] +
        k[1619]*y[IDX_CH2I] + k[1620]*y[IDX_CH2I] + k[1620]*y[IDX_CH2I] +
        k[1631]*y[IDX_NOI] + k[1633]*y[IDX_O2I] + k[1633]*y[IDX_O2I] +
        k[1638]*y[IDX_OI] + k[1638]*y[IDX_OI] + k[1639]*y[IDX_OI] +
        k[1641]*y[IDX_OHI] + k[1645]*y[IDX_SI] + k[1723]*y[IDX_H2I] -
        k[1739]*y[IDX_HI] + k[1801]*y[IDX_NI] + k[1802]*y[IDX_NI] + k[1982];
    data[5671] = 0.0 + k[115]*y[IDX_HII] + k[384] + k[610]*y[IDX_CII] + k[790]*y[IDX_SII]
        + k[1588]*y[IDX_CI] + k[1648]*y[IDX_CH3I] + k[1648]*y[IDX_CH3I] +
        k[1664]*y[IDX_OI] + k[1665]*y[IDX_OI] + k[1669]*y[IDX_SI] +
        k[1724]*y[IDX_H2I] - k[1741]*y[IDX_HI] + k[1804]*y[IDX_NI] +
        k[1806]*y[IDX_NI] + k[1806]*y[IDX_NI] + k[1986];
    data[5672] = 0.0 + k[380] + k[477]*y[IDX_EM] + k[686]*y[IDX_CI] + k[705]*y[IDX_C2I] +
        k[711]*y[IDX_CH4I] + k[713]*y[IDX_CNI] + k[718]*y[IDX_H2OI] +
        k[724]*y[IDX_HCNI] + k[728]*y[IDX_NI] + k[735]*y[IDX_OI] +
        k[739]*y[IDX_SI] + k[937]*y[IDX_H2I] - k[1098]*y[IDX_HI];
    data[5673] = 0.0 + k[132]*y[IDX_HII] + k[429] + k[631]*y[IDX_CII] +
        k[929]*y[IDX_H2II] + k[1256]*y[IDX_HeII] + k[1308]*y[IDX_NII] +
        k[1445]*y[IDX_C2II] + k[1457]*y[IDX_OII] + k[1461]*y[IDX_SII] +
        k[1602]*y[IDX_CI] + k[1730]*y[IDX_H2I] - k[1762]*y[IDX_HI] +
        k[1819]*y[IDX_NI] + k[1844]*y[IDX_NHI] + k[1844]*y[IDX_NHI] +
        k[1844]*y[IDX_NHI] + k[1844]*y[IDX_NHI] + k[1847]*y[IDX_NOI] +
        k[1851]*y[IDX_OI] + k[1854]*y[IDX_OHI] + k[1857]*y[IDX_SI] + k[2054];
    data[5674] = 0.0 + k[772]*y[IDX_CH2I] + k[790]*y[IDX_CH3I] + k[821]*y[IDX_CH4I] +
        k[822]*y[IDX_CH4I] + k[861]*y[IDX_CHI] + k[961]*y[IDX_H2I] +
        k[1461]*y[IDX_NHI] + k[1548]*y[IDX_OHI] + k[1550]*y[IDX_H2SI] +
        k[1569]*y[IDX_SiHI];
    data[5675] = 0.0 + k[713]*y[IDX_CHII] + k[917]*y[IDX_H2II] + k[1701]*y[IDX_C2H2I] +
        k[1705]*y[IDX_HCNI] + k[1707]*y[IDX_HNCI] + k[1726]*y[IDX_H2I] +
        k[1933]*y[IDX_OHI];
    data[5676] = 0.0 + k[119]*y[IDX_HII] + k[892]*y[IDX_HII] + k[921]*y[IDX_H2II] +
        k[972]*y[IDX_O2II] + k[1217]*y[IDX_HeII] + k[1314]*y[IDX_N2II] -
        k[1747]*y[IDX_HI] + k[2012] + k[2012] + k[2014];
    data[5677] = 0.0 + k[116]*y[IDX_HII] + k[711]*y[IDX_CHII] + k[806]*y[IDX_C2H2II] +
        k[816]*y[IDX_N2II] + k[821]*y[IDX_SII] + k[822]*y[IDX_SII] +
        k[914]*y[IDX_H2II] + k[915]*y[IDX_H2II] + k[1202]*y[IDX_HeII] +
        k[1204]*y[IDX_HeII] + k[1293]*y[IDX_NII] + k[1294]*y[IDX_NII] +
        k[1295]*y[IDX_NII] + k[1295]*y[IDX_NII] + k[1676]*y[IDX_CHI] -
        k[1742]*y[IDX_HI] + k[1996] + k[1998];
    data[5678] = 0.0 + k[125]*y[IDX_HII] + k[409] + k[1235]*y[IDX_HeII] -
        k[1751]*y[IDX_HI] - k[1752]*y[IDX_HI] + k[1813]*y[IDX_NI] +
        k[1892]*y[IDX_OI] + k[1951]*y[IDX_SI] + k[2030];
    data[5679] = 0.0 + k[124]*y[IDX_HII] + k[408] + k[724]*y[IDX_CHII] +
        k[866]*y[IDX_CNII] + k[1119]*y[IDX_C3II] + k[1231]*y[IDX_HeII] +
        k[1233]*y[IDX_HeII] + k[1571]*y[IDX_C2I] + k[1578]*y[IDX_C2HI] +
        k[1705]*y[IDX_CNI] - k[1750]*y[IDX_HI] + k[1891]*y[IDX_OI] + k[2028];
    data[5680] = 0.0 + k[133]*y[IDX_HII] + k[930]*y[IDX_H2II] + k[1574]*y[IDX_C2H2I] +
        k[1631]*y[IDX_CH2I] + k[1686]*y[IDX_CHI] - k[1764]*y[IDX_HI] -
        k[1765]*y[IDX_HI] + k[1835]*y[IDX_NH2I] + k[1847]*y[IDX_NHI] +
        k[1945]*y[IDX_OHI];
    data[5681] = 0.0 + k[2]*y[IDX_H2I] - k[9]*y[IDX_HI] + k[9]*y[IDX_HI] + k[9]*y[IDX_HI]
        + k[117]*y[IDX_HII] + k[391] + k[615]*y[IDX_CII] + k[837]*y[IDX_C2II] +
        k[852]*y[IDX_NII] + k[857]*y[IDX_OII] + k[861]*y[IDX_SII] +
        k[862]*y[IDX_SiII] + k[916]*y[IDX_H2II] + k[1206]*y[IDX_HeII] +
        k[1589]*y[IDX_CI] + k[1674]*y[IDX_C2H2I] + k[1675]*y[IDX_C2H4I] +
        k[1676]*y[IDX_CH4I] + k[1682]*y[IDX_NI] + k[1686]*y[IDX_NOI] +
        k[1687]*y[IDX_O2I] + k[1688]*y[IDX_O2I] + k[1693]*y[IDX_OI] +
        k[1695]*y[IDX_OCSI] + k[1696]*y[IDX_OHI] + k[1697]*y[IDX_SI] +
        k[1700]*y[IDX_SOI] + k[1725]*y[IDX_H2I] - k[1743]*y[IDX_HI] + k[1999];
    data[5682] = 0.0 + k[131]*y[IDX_HII] + k[426] + k[1255]*y[IDX_HeII] -
        k[1761]*y[IDX_HI] + k[2051];
    data[5683] = 0.0 + k[528]*y[IDX_EM] + k[529]*y[IDX_EM] + k[531]*y[IDX_EM] +
        k[531]*y[IDX_EM] + k[2246];
    data[5684] = 0.0 - k[12]*y[IDX_HI] + k[12]*y[IDX_HI] + k[135]*y[IDX_HII] +
        k[931]*y[IDX_H2II] + k[1633]*y[IDX_CH2I] + k[1633]*y[IDX_CH2I] +
        k[1687]*y[IDX_CHI] + k[1688]*y[IDX_CHI] + k[1731]*y[IDX_H2I] -
        k[1768]*y[IDX_HI];
    data[5685] = 0.0 + k[607]*y[IDX_C2HI] + k[608]*y[IDX_CH2I] + k[610]*y[IDX_CH3I] +
        k[615]*y[IDX_CHI] + k[620]*y[IDX_H2OI] + k[621]*y[IDX_H2OI] +
        k[622]*y[IDX_H2SI] + k[625]*y[IDX_HC3NI] + k[627]*y[IDX_HNCI] +
        k[628]*y[IDX_HSI] + k[629]*y[IDX_NH2I] + k[631]*y[IDX_NHI] +
        k[637]*y[IDX_OHI] + k[644]*y[IDX_SiHI] + k[934]*y[IDX_H2I] -
        k[2122]*y[IDX_HI];
    data[5686] = 0.0 + k[140]*y[IDX_HII] + k[739]*y[IDX_CHII] + k[753]*y[IDX_CH2II] +
        k[988]*y[IDX_H2OII] + k[1373]*y[IDX_NHII] + k[1392]*y[IDX_NH2II] +
        k[1534]*y[IDX_OHII] + k[1553]*y[IDX_H3SII] + k[1568]*y[IDX_SiH2II] +
        k[1645]*y[IDX_CH2I] + k[1669]*y[IDX_CH3I] + k[1697]*y[IDX_CHI] +
        k[1735]*y[IDX_H2I] + k[1857]*y[IDX_NHI] + k[1948]*y[IDX_OHI] +
        k[1951]*y[IDX_HCOI] + k[1953]*y[IDX_HSI];
    data[5687] = 0.0 + k[7]*y[IDX_H2I] - k[13]*y[IDX_HI] + k[13]*y[IDX_HI] +
        k[13]*y[IDX_HI] + k[138]*y[IDX_HII] + k[442] + k[637]*y[IDX_CII] +
        k[933]*y[IDX_H2II] + k[1268]*y[IDX_HeII] + k[1480]*y[IDX_OII] +
        k[1543]*y[IDX_HCOII] + k[1548]*y[IDX_SII] + k[1549]*y[IDX_SiII] +
        k[1611]*y[IDX_CI] + k[1641]*y[IDX_CH2I] + k[1696]*y[IDX_CHI] +
        k[1734]*y[IDX_H2I] - k[1776]*y[IDX_HI] + k[1827]*y[IDX_NI] +
        k[1854]*y[IDX_NHI] + k[1914]*y[IDX_OI] + k[1928]*y[IDX_C2H2I] +
        k[1933]*y[IDX_CNI] + k[1934]*y[IDX_COI] + k[1936]*y[IDX_CSI] +
        k[1945]*y[IDX_NOI] + k[1948]*y[IDX_SI] + k[1949]*y[IDX_SOI] +
        k[1950]*y[IDX_SiI] + k[2069] - k[2125]*y[IDX_HI];
    data[5688] = 0.0 + k[728]*y[IDX_CHII] + k[928]*y[IDX_H2II] + k[1326]*y[IDX_C2HII] +
        k[1329]*y[IDX_C2H2II] + k[1331]*y[IDX_CH2II] + k[1333]*y[IDX_H2OII] +
        k[1336]*y[IDX_HSII] + k[1337]*y[IDX_NHII] + k[1338]*y[IDX_NH2II] +
        k[1340]*y[IDX_OHII] + k[1682]*y[IDX_CHI] + k[1728]*y[IDX_H2I] +
        k[1795]*y[IDX_C2HI] + k[1797]*y[IDX_C3H2I] + k[1799]*y[IDX_C4HI] +
        k[1801]*y[IDX_CH2I] + k[1802]*y[IDX_CH2I] + k[1804]*y[IDX_CH3I] +
        k[1806]*y[IDX_CH3I] + k[1806]*y[IDX_CH3I] + k[1813]*y[IDX_HCOI] +
        k[1816]*y[IDX_HSI] + k[1819]*y[IDX_NHI] + k[1827]*y[IDX_OHI];
    data[5689] = 0.0 - k[195]*y[IDX_HI] + k[950]*y[IDX_H2I] + k[1179]*y[IDX_C2H2I] +
        k[1182]*y[IDX_C2H3I] + k[1183]*y[IDX_C2H4I] + k[1186]*y[IDX_C2HI] +
        k[1193]*y[IDX_CH2I] + k[1202]*y[IDX_CH4I] + k[1204]*y[IDX_CH4I] +
        k[1206]*y[IDX_CHI] + k[1217]*y[IDX_H2COI] + k[1222]*y[IDX_H2OI] +
        k[1225]*y[IDX_H2S2I] + k[1226]*y[IDX_H2SI] + k[1228]*y[IDX_H2SiOI] +
        k[1231]*y[IDX_HCNI] + k[1233]*y[IDX_HCNI] + k[1235]*y[IDX_HCOI] +
        k[1239]*y[IDX_HCSI] + k[1241]*y[IDX_HClI] + k[1242]*y[IDX_HNCI] +
        k[1243]*y[IDX_HNCI] + k[1245]*y[IDX_HNOI] + k[1248]*y[IDX_HS2I] +
        k[1249]*y[IDX_HSI] + k[1253]*y[IDX_NH2I] + k[1255]*y[IDX_NH3I] +
        k[1256]*y[IDX_NHI] + k[1268]*y[IDX_OHI] + k[1279]*y[IDX_SiH2I] +
        k[1281]*y[IDX_SiH3I] + k[1283]*y[IDX_SiH4I] + k[1284]*y[IDX_SiHI];
    data[5690] = 0.0 + k[926]*y[IDX_H2II];
    data[5691] = 0.0 + k[519]*y[IDX_EM] + k[520]*y[IDX_EM] + k[520]*y[IDX_EM] +
        k[520]*y[IDX_EM] + k[1054]*y[IDX_MgI] + k[1063]*y[IDX_OI] + k[2025];
    data[5692] = 0.0 + k[109]*y[IDX_ClI] + k[110]*y[IDX_C2I] + k[111]*y[IDX_C2H2I] +
        k[112]*y[IDX_C2HI] + k[113]*y[IDX_C2NI] + k[114]*y[IDX_CH2I] +
        k[115]*y[IDX_CH3I] + k[116]*y[IDX_CH4I] + k[117]*y[IDX_CHI] +
        k[118]*y[IDX_CSI] + k[119]*y[IDX_H2COI] + k[120]*y[IDX_H2CSI] +
        k[121]*y[IDX_H2OI] + k[122]*y[IDX_H2S2I] + k[123]*y[IDX_H2SI] +
        k[124]*y[IDX_HCNI] + k[125]*y[IDX_HCOI] + k[126]*y[IDX_HClI] +
        k[127]*y[IDX_HS2I] + k[128]*y[IDX_HSI] + k[129]*y[IDX_MgI] +
        k[130]*y[IDX_NH2I] + k[131]*y[IDX_NH3I] + k[132]*y[IDX_NHI] +
        k[133]*y[IDX_NOI] + k[134]*y[IDX_NSI] + k[135]*y[IDX_O2I] +
        k[136]*y[IDX_OI] + k[137]*y[IDX_OCSI] + k[138]*y[IDX_OHI] +
        k[139]*y[IDX_S2I] + k[140]*y[IDX_SI] + k[141]*y[IDX_SO2I] +
        k[142]*y[IDX_SOI] + k[143]*y[IDX_SiI] + k[144]*y[IDX_SiC2I] +
        k[145]*y[IDX_SiC3I] + k[146]*y[IDX_SiCI] + k[147]*y[IDX_SiH2I] +
        k[148]*y[IDX_SiH3I] + k[149]*y[IDX_SiH4I] + k[150]*y[IDX_SiHI] +
        k[151]*y[IDX_SiOI] + k[152]*y[IDX_SiSI] + k[883]*y[IDX_C2H4I] +
        k[892]*y[IDX_H2COI] + k[895]*y[IDX_H2SI] - k[2110]*y[IDX_HI] +
        k[2135]*y[IDX_EM];
    data[5693] = 0.0 + k[136]*y[IDX_HII] + k[735]*y[IDX_CHII] + k[750]*y[IDX_CH2II] +
        k[783]*y[IDX_CH3II] + k[932]*y[IDX_H2II] + k[1063]*y[IDX_H3II] +
        k[1501]*y[IDX_HCSII] + k[1504]*y[IDX_HSII] + k[1507]*y[IDX_NH2II] +
        k[1511]*y[IDX_OHII] + k[1513]*y[IDX_SiHII] + k[1514]*y[IDX_SiH2II] +
        k[1638]*y[IDX_CH2I] + k[1638]*y[IDX_CH2I] + k[1639]*y[IDX_CH2I] +
        k[1664]*y[IDX_CH3I] + k[1665]*y[IDX_CH3I] + k[1693]*y[IDX_CHI] +
        k[1733]*y[IDX_H2I] + k[1851]*y[IDX_NHI] + k[1869]*y[IDX_C2H3I] +
        k[1891]*y[IDX_HCNI] + k[1892]*y[IDX_HCOI] + k[1895]*y[IDX_HCSI] +
        k[1896]*y[IDX_HNOI] + k[1900]*y[IDX_HSI] + k[1902]*y[IDX_NH2I] +
        k[1914]*y[IDX_OHI] + k[1923]*y[IDX_SiH2I] + k[1923]*y[IDX_SiH2I] +
        k[1924]*y[IDX_SiH3I] + k[1926]*y[IDX_SiHI] - k[2124]*y[IDX_HI];
    data[5694] = 0.0 + k[685]*y[IDX_C2HII] + k[686]*y[IDX_CHII] + k[687]*y[IDX_CH2II] +
        k[691]*y[IDX_H2SII] + k[697]*y[IDX_HSII] + k[703]*y[IDX_SiHII] +
        k[912]*y[IDX_H2II] + k[1582]*y[IDX_C2H3I] + k[1583]*y[IDX_C2H5I] +
        k[1585]*y[IDX_C3H2I] + k[1586]*y[IDX_CH2I] + k[1588]*y[IDX_CH3I] +
        k[1589]*y[IDX_CHI] + k[1595]*y[IDX_HSI] + k[1599]*y[IDX_NH2I] +
        k[1600]*y[IDX_NH2I] + k[1602]*y[IDX_NHI] + k[1611]*y[IDX_OHI] +
        k[1617]*y[IDX_SiHI] + k[1722]*y[IDX_H2I] - k[2123]*y[IDX_HI];
    data[5695] = 0.0 + k[542]*y[IDX_EM] + k[1543]*y[IDX_OHI] + k[2029];
    data[5696] = 0.0 + k[4]*y[IDX_H2I] - k[11]*y[IDX_HI] + k[11]*y[IDX_HI] +
        k[11]*y[IDX_HI] + k[121]*y[IDX_HII] + k[401] + k[620]*y[IDX_CII] +
        k[621]*y[IDX_CII] + k[718]*y[IDX_CHII] + k[743]*y[IDX_CH2II] +
        k[922]*y[IDX_H2II] + k[1013]*y[IDX_SiII] + k[1222]*y[IDX_HeII] -
        k[1748]*y[IDX_HI] + k[2018];
    data[5697] = 0.0 + k[919]*y[IDX_H2II] - k[1745]*y[IDX_HI] + k[1934]*y[IDX_OHI];
    data[5698] = 0.0 + k[2]*y[IDX_CHI] + k[3]*y[IDX_H2I] + k[3]*y[IDX_H2I] +
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
        k[1734]*y[IDX_OHI] + k[1735]*y[IDX_SI] + (2.0 * H2dissociation);
    data[5699] = 0.0 + k[8]*y[IDX_H2I] + k[8]*y[IDX_H2I] + k[459]*y[IDX_C2HII] +
        k[461]*y[IDX_C2H2II] + k[461]*y[IDX_C2H2II] + k[462]*y[IDX_C2H2II] +
        k[464]*y[IDX_C2H5OH2II] + k[466]*y[IDX_C2H5OH2II] +
        k[467]*y[IDX_C2H5OH2II] + k[472]*y[IDX_C2NHII] + k[474]*y[IDX_C3H5II] +
        k[477]*y[IDX_CHII] + k[479]*y[IDX_CH2II] + k[479]*y[IDX_CH2II] +
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
    data[5700] = 0.0 - k[9]*y[IDX_CHI] + k[9]*y[IDX_CHI] + k[9]*y[IDX_CHI] -
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
        k[2126]*y[IDX_SiII] + (-2.0 * H2formation);
    
    // clang-format on

    /* */

    return NAUNET_SUCCESS;
}