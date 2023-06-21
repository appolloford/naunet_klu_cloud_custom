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
    realtype ksp = u_data->ksp;
    
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
    rowptrs[4] = 7;
    rowptrs[5] = 10;
    rowptrs[6] = 13;
    rowptrs[7] = 16;
    rowptrs[8] = 19;
    rowptrs[9] = 22;
    rowptrs[10] = 25;
    rowptrs[11] = 28;
    rowptrs[12] = 31;
    rowptrs[13] = 34;
    rowptrs[14] = 38;
    rowptrs[15] = 41;
    rowptrs[16] = 46;
    rowptrs[17] = 50;
    rowptrs[18] = 54;
    rowptrs[19] = 61;
    rowptrs[20] = 67;
    rowptrs[21] = 75;
    rowptrs[22] = 84;
    rowptrs[23] = 89;
    rowptrs[24] = 95;
    rowptrs[25] = 106;
    rowptrs[26] = 112;
    rowptrs[27] = 118;
    rowptrs[28] = 124;
    rowptrs[29] = 132;
    rowptrs[30] = 141;
    rowptrs[31] = 148;
    rowptrs[32] = 160;
    rowptrs[33] = 168;
    rowptrs[34] = 174;
    rowptrs[35] = 183;
    rowptrs[36] = 192;
    rowptrs[37] = 205;
    rowptrs[38] = 219;
    rowptrs[39] = 231;
    rowptrs[40] = 249;
    rowptrs[41] = 263;
    rowptrs[42] = 279;
    rowptrs[43] = 295;
    rowptrs[44] = 308;
    rowptrs[45] = 327;
    rowptrs[46] = 347;
    rowptrs[47] = 359;
    rowptrs[48] = 375;
    rowptrs[49] = 388;
    rowptrs[50] = 406;
    rowptrs[51] = 424;
    rowptrs[52] = 447;
    rowptrs[53] = 459;
    rowptrs[54] = 478;
    rowptrs[55] = 496;
    rowptrs[56] = 517;
    rowptrs[57] = 542;
    rowptrs[58] = 570;
    rowptrs[59] = 599;
    rowptrs[60] = 629;
    rowptrs[61] = 654;
    rowptrs[62] = 686;
    rowptrs[63] = 720;
    rowptrs[64] = 748;
    rowptrs[65] = 778;
    rowptrs[66] = 805;
    rowptrs[67] = 840;
    rowptrs[68] = 873;
    rowptrs[69] = 906;
    rowptrs[70] = 935;
    rowptrs[71] = 968;
    rowptrs[72] = 1006;
    rowptrs[73] = 1044;
    rowptrs[74] = 1069;
    rowptrs[75] = 1106;
    rowptrs[76] = 1143;
    rowptrs[77] = 1179;
    rowptrs[78] = 1210;
    rowptrs[79] = 1242;
    rowptrs[80] = 1283;
    rowptrs[81] = 1329;
    rowptrs[82] = 1372;
    rowptrs[83] = 1410;
    rowptrs[84] = 1449;
    rowptrs[85] = 1490;
    rowptrs[86] = 1528;
    rowptrs[87] = 1564;
    rowptrs[88] = 1605;
    rowptrs[89] = 1656;
    rowptrs[90] = 1706;
    rowptrs[91] = 1757;
    rowptrs[92] = 1802;
    rowptrs[93] = 1857;
    rowptrs[94] = 1898;
    rowptrs[95] = 1943;
    rowptrs[96] = 1996;
    rowptrs[97] = 2046;
    rowptrs[98] = 2083;
    rowptrs[99] = 2140;
    rowptrs[100] = 2176;
    rowptrs[101] = 2216;
    rowptrs[102] = 2270;
    rowptrs[103] = 2322;
    rowptrs[104] = 2382;
    rowptrs[105] = 2440;
    rowptrs[106] = 2500;
    rowptrs[107] = 2547;
    rowptrs[108] = 2604;
    rowptrs[109] = 2662;
    rowptrs[110] = 2733;
    rowptrs[111] = 2806;
    rowptrs[112] = 2872;
    rowptrs[113] = 2944;
    rowptrs[114] = 3028;
    
    // the column index of non-zero elements
    colvals[0] = 0;
    colvals[1] = 23;
    colvals[2] = 1;
    colvals[3] = 63;
    colvals[4] = 2;
    colvals[5] = 39;
    colvals[6] = 3;
    colvals[7] = 4;
    colvals[8] = 68;
    colvals[9] = 111;
    colvals[10] = 5;
    colvals[11] = 27;
    colvals[12] = 42;
    colvals[13] = 6;
    colvals[14] = 53;
    colvals[15] = 54;
    colvals[16] = 7;
    colvals[17] = 84;
    colvals[18] = 101;
    colvals[19] = 8;
    colvals[20] = 78;
    colvals[21] = 104;
    colvals[22] = 9;
    colvals[23] = 37;
    colvals[24] = 60;
    colvals[25] = 10;
    colvals[26] = 36;
    colvals[27] = 38;
    colvals[28] = 11;
    colvals[29] = 30;
    colvals[30] = 32;
    colvals[31] = 12;
    colvals[32] = 22;
    colvals[33] = 33;
    colvals[34] = 13;
    colvals[35] = 52;
    colvals[36] = 66;
    colvals[37] = 111;
    colvals[38] = 14;
    colvals[39] = 50;
    colvals[40] = 72;
    colvals[41] = 15;
    colvals[42] = 26;
    colvals[43] = 44;
    colvals[44] = 49;
    colvals[45] = 56;
    colvals[46] = 16;
    colvals[47] = 25;
    colvals[48] = 51;
    colvals[49] = 64;
    colvals[50] = 17;
    colvals[51] = 59;
    colvals[52] = 65;
    colvals[53] = 86;
    colvals[54] = 18;
    colvals[55] = 28;
    colvals[56] = 91;
    colvals[57] = 95;
    colvals[58] = 96;
    colvals[59] = 107;
    colvals[60] = 111;
    colvals[61] = 19;
    colvals[62] = 57;
    colvals[63] = 58;
    colvals[64] = 75;
    colvals[65] = 88;
    colvals[66] = 94;
    colvals[67] = 20;
    colvals[68] = 74;
    colvals[69] = 82;
    colvals[70] = 85;
    colvals[71] = 93;
    colvals[72] = 103;
    colvals[73] = 108;
    colvals[74] = 109;
    colvals[75] = 21;
    colvals[76] = 70;
    colvals[77] = 73;
    colvals[78] = 76;
    colvals[79] = 77;
    colvals[80] = 80;
    colvals[81] = 83;
    colvals[82] = 92;
    colvals[83] = 102;
    colvals[84] = 22;
    colvals[85] = 33;
    colvals[86] = 87;
    colvals[87] = 106;
    colvals[88] = 110;
    colvals[89] = 0;
    colvals[90] = 23;
    colvals[91] = 71;
    colvals[92] = 102;
    colvals[93] = 109;
    colvals[94] = 113;
    colvals[95] = 24;
    colvals[96] = 55;
    colvals[97] = 67;
    colvals[98] = 71;
    colvals[99] = 79;
    colvals[100] = 81;
    colvals[101] = 87;
    colvals[102] = 89;
    colvals[103] = 90;
    colvals[104] = 98;
    colvals[105] = 105;
    colvals[106] = 25;
    colvals[107] = 51;
    colvals[108] = 76;
    colvals[109] = 99;
    colvals[110] = 104;
    colvals[111] = 110;
    colvals[112] = 15;
    colvals[113] = 26;
    colvals[114] = 48;
    colvals[115] = 97;
    colvals[116] = 106;
    colvals[117] = 109;
    colvals[118] = 5;
    colvals[119] = 27;
    colvals[120] = 90;
    colvals[121] = 101;
    colvals[122] = 105;
    colvals[123] = 106;
    colvals[124] = 28;
    colvals[125] = 68;
    colvals[126] = 87;
    colvals[127] = 99;
    colvals[128] = 108;
    colvals[129] = 110;
    colvals[130] = 111;
    colvals[131] = 112;
    colvals[132] = 29;
    colvals[133] = 69;
    colvals[134] = 96;
    colvals[135] = 97;
    colvals[136] = 100;
    colvals[137] = 106;
    colvals[138] = 110;
    colvals[139] = 112;
    colvals[140] = 113;
    colvals[141] = 30;
    colvals[142] = 32;
    colvals[143] = 33;
    colvals[144] = 87;
    colvals[145] = 97;
    colvals[146] = 106;
    colvals[147] = 110;
    colvals[148] = 31;
    colvals[149] = 34;
    colvals[150] = 35;
    colvals[151] = 40;
    colvals[152] = 41;
    colvals[153] = 43;
    colvals[154] = 45;
    colvals[155] = 46;
    colvals[156] = 47;
    colvals[157] = 48;
    colvals[158] = 61;
    colvals[159] = 62;
    colvals[160] = 11;
    colvals[161] = 22;
    colvals[162] = 32;
    colvals[163] = 33;
    colvals[164] = 87;
    colvals[165] = 106;
    colvals[166] = 109;
    colvals[167] = 110;
    colvals[168] = 12;
    colvals[169] = 33;
    colvals[170] = 87;
    colvals[171] = 97;
    colvals[172] = 106;
    colvals[173] = 109;
    colvals[174] = 34;
    colvals[175] = 35;
    colvals[176] = 40;
    colvals[177] = 46;
    colvals[178] = 99;
    colvals[179] = 107;
    colvals[180] = 108;
    colvals[181] = 110;
    colvals[182] = 112;
    colvals[183] = 35;
    colvals[184] = 46;
    colvals[185] = 48;
    colvals[186] = 99;
    colvals[187] = 106;
    colvals[188] = 108;
    colvals[189] = 110;
    colvals[190] = 111;
    colvals[191] = 112;
    colvals[192] = 36;
    colvals[193] = 38;
    colvals[194] = 43;
    colvals[195] = 45;
    colvals[196] = 47;
    colvals[197] = 61;
    colvals[198] = 87;
    colvals[199] = 98;
    colvals[200] = 102;
    colvals[201] = 105;
    colvals[202] = 106;
    colvals[203] = 109;
    colvals[204] = 110;
    colvals[205] = 9;
    colvals[206] = 37;
    colvals[207] = 67;
    colvals[208] = 71;
    colvals[209] = 91;
    colvals[210] = 96;
    colvals[211] = 98;
    colvals[212] = 102;
    colvals[213] = 103;
    colvals[214] = 104;
    colvals[215] = 109;
    colvals[216] = 111;
    colvals[217] = 112;
    colvals[218] = 113;
    colvals[219] = 10;
    colvals[220] = 30;
    colvals[221] = 32;
    colvals[222] = 38;
    colvals[223] = 47;
    colvals[224] = 87;
    colvals[225] = 97;
    colvals[226] = 102;
    colvals[227] = 105;
    colvals[228] = 106;
    colvals[229] = 109;
    colvals[230] = 110;
    colvals[231] = 2;
    colvals[232] = 39;
    colvals[233] = 42;
    colvals[234] = 51;
    colvals[235] = 71;
    colvals[236] = 74;
    colvals[237] = 90;
    colvals[238] = 92;
    colvals[239] = 94;
    colvals[240] = 99;
    colvals[241] = 101;
    colvals[242] = 102;
    colvals[243] = 103;
    colvals[244] = 104;
    colvals[245] = 106;
    colvals[246] = 109;
    colvals[247] = 111;
    colvals[248] = 113;
    colvals[249] = 40;
    colvals[250] = 43;
    colvals[251] = 45;
    colvals[252] = 46;
    colvals[253] = 48;
    colvals[254] = 79;
    colvals[255] = 85;
    colvals[256] = 87;
    colvals[257] = 99;
    colvals[258] = 106;
    colvals[259] = 107;
    colvals[260] = 109;
    colvals[261] = 110;
    colvals[262] = 112;
    colvals[263] = 41;
    colvals[264] = 43;
    colvals[265] = 47;
    colvals[266] = 48;
    colvals[267] = 61;
    colvals[268] = 85;
    colvals[269] = 87;
    colvals[270] = 93;
    colvals[271] = 97;
    colvals[272] = 99;
    colvals[273] = 104;
    colvals[274] = 106;
    colvals[275] = 107;
    colvals[276] = 109;
    colvals[277] = 110;
    colvals[278] = 112;
    colvals[279] = 23;
    colvals[280] = 39;
    colvals[281] = 42;
    colvals[282] = 87;
    colvals[283] = 88;
    colvals[284] = 94;
    colvals[285] = 96;
    colvals[286] = 97;
    colvals[287] = 98;
    colvals[288] = 101;
    colvals[289] = 102;
    colvals[290] = 103;
    colvals[291] = 104;
    colvals[292] = 105;
    colvals[293] = 109;
    colvals[294] = 113;
    colvals[295] = 35;
    colvals[296] = 40;
    colvals[297] = 43;
    colvals[298] = 46;
    colvals[299] = 48;
    colvals[300] = 85;
    colvals[301] = 87;
    colvals[302] = 97;
    colvals[303] = 99;
    colvals[304] = 106;
    colvals[305] = 107;
    colvals[306] = 109;
    colvals[307] = 110;
    colvals[308] = 26;
    colvals[309] = 40;
    colvals[310] = 41;
    colvals[311] = 44;
    colvals[312] = 49;
    colvals[313] = 52;
    colvals[314] = 56;
    colvals[315] = 61;
    colvals[316] = 85;
    colvals[317] = 93;
    colvals[318] = 97;
    colvals[319] = 99;
    colvals[320] = 104;
    colvals[321] = 106;
    colvals[322] = 107;
    colvals[323] = 108;
    colvals[324] = 109;
    colvals[325] = 110;
    colvals[326] = 112;
    colvals[327] = 43;
    colvals[328] = 45;
    colvals[329] = 46;
    colvals[330] = 47;
    colvals[331] = 48;
    colvals[332] = 61;
    colvals[333] = 62;
    colvals[334] = 85;
    colvals[335] = 93;
    colvals[336] = 97;
    colvals[337] = 98;
    colvals[338] = 99;
    colvals[339] = 105;
    colvals[340] = 106;
    colvals[341] = 107;
    colvals[342] = 108;
    colvals[343] = 109;
    colvals[344] = 110;
    colvals[345] = 112;
    colvals[346] = 113;
    colvals[347] = 31;
    colvals[348] = 34;
    colvals[349] = 46;
    colvals[350] = 79;
    colvals[351] = 94;
    colvals[352] = 97;
    colvals[353] = 99;
    colvals[354] = 106;
    colvals[355] = 107;
    colvals[356] = 108;
    colvals[357] = 109;
    colvals[358] = 110;
    colvals[359] = 40;
    colvals[360] = 41;
    colvals[361] = 43;
    colvals[362] = 46;
    colvals[363] = 47;
    colvals[364] = 48;
    colvals[365] = 85;
    colvals[366] = 87;
    colvals[367] = 93;
    colvals[368] = 97;
    colvals[369] = 99;
    colvals[370] = 105;
    colvals[371] = 106;
    colvals[372] = 107;
    colvals[373] = 109;
    colvals[374] = 110;
    colvals[375] = 34;
    colvals[376] = 35;
    colvals[377] = 46;
    colvals[378] = 48;
    colvals[379] = 87;
    colvals[380] = 94;
    colvals[381] = 97;
    colvals[382] = 99;
    colvals[383] = 106;
    colvals[384] = 108;
    colvals[385] = 109;
    colvals[386] = 110;
    colvals[387] = 111;
    colvals[388] = 36;
    colvals[389] = 45;
    colvals[390] = 49;
    colvals[391] = 53;
    colvals[392] = 56;
    colvals[393] = 61;
    colvals[394] = 90;
    colvals[395] = 96;
    colvals[396] = 98;
    colvals[397] = 101;
    colvals[398] = 102;
    colvals[399] = 103;
    colvals[400] = 105;
    colvals[401] = 106;
    colvals[402] = 109;
    colvals[403] = 110;
    colvals[404] = 111;
    colvals[405] = 112;
    colvals[406] = 50;
    colvals[407] = 55;
    colvals[408] = 59;
    colvals[409] = 60;
    colvals[410] = 64;
    colvals[411] = 69;
    colvals[412] = 72;
    colvals[413] = 75;
    colvals[414] = 77;
    colvals[415] = 93;
    colvals[416] = 99;
    colvals[417] = 103;
    colvals[418] = 105;
    colvals[419] = 107;
    colvals[420] = 108;
    colvals[421] = 109;
    colvals[422] = 110;
    colvals[423] = 111;
    colvals[424] = 16;
    colvals[425] = 25;
    colvals[426] = 39;
    colvals[427] = 51;
    colvals[428] = 64;
    colvals[429] = 71;
    colvals[430] = 80;
    colvals[431] = 90;
    colvals[432] = 92;
    colvals[433] = 94;
    colvals[434] = 96;
    colvals[435] = 97;
    colvals[436] = 98;
    colvals[437] = 99;
    colvals[438] = 101;
    colvals[439] = 102;
    colvals[440] = 103;
    colvals[441] = 104;
    colvals[442] = 106;
    colvals[443] = 109;
    colvals[444] = 110;
    colvals[445] = 111;
    colvals[446] = 113;
    colvals[447] = 13;
    colvals[448] = 52;
    colvals[449] = 61;
    colvals[450] = 73;
    colvals[451] = 74;
    colvals[452] = 78;
    colvals[453] = 79;
    colvals[454] = 87;
    colvals[455] = 89;
    colvals[456] = 97;
    colvals[457] = 99;
    colvals[458] = 106;
    colvals[459] = 6;
    colvals[460] = 49;
    colvals[461] = 53;
    colvals[462] = 54;
    colvals[463] = 61;
    colvals[464] = 65;
    colvals[465] = 73;
    colvals[466] = 78;
    colvals[467] = 79;
    colvals[468] = 82;
    colvals[469] = 83;
    colvals[470] = 84;
    colvals[471] = 87;
    colvals[472] = 89;
    colvals[473] = 91;
    colvals[474] = 99;
    colvals[475] = 106;
    colvals[476] = 107;
    colvals[477] = 110;
    colvals[478] = 49;
    colvals[479] = 53;
    colvals[480] = 54;
    colvals[481] = 61;
    colvals[482] = 65;
    colvals[483] = 73;
    colvals[484] = 78;
    colvals[485] = 79;
    colvals[486] = 82;
    colvals[487] = 83;
    colvals[488] = 84;
    colvals[489] = 87;
    colvals[490] = 89;
    colvals[491] = 91;
    colvals[492] = 99;
    colvals[493] = 106;
    colvals[494] = 107;
    colvals[495] = 110;
    colvals[496] = 55;
    colvals[497] = 67;
    colvals[498] = 68;
    colvals[499] = 69;
    colvals[500] = 70;
    colvals[501] = 71;
    colvals[502] = 72;
    colvals[503] = 73;
    colvals[504] = 74;
    colvals[505] = 79;
    colvals[506] = 95;
    colvals[507] = 96;
    colvals[508] = 97;
    colvals[509] = 99;
    colvals[510] = 104;
    colvals[511] = 106;
    colvals[512] = 108;
    colvals[513] = 109;
    colvals[514] = 110;
    colvals[515] = 111;
    colvals[516] = 113;
    colvals[517] = 3;
    colvals[518] = 26;
    colvals[519] = 38;
    colvals[520] = 43;
    colvals[521] = 44;
    colvals[522] = 47;
    colvals[523] = 49;
    colvals[524] = 53;
    colvals[525] = 56;
    colvals[526] = 62;
    colvals[527] = 72;
    colvals[528] = 85;
    colvals[529] = 87;
    colvals[530] = 93;
    colvals[531] = 96;
    colvals[532] = 97;
    colvals[533] = 99;
    colvals[534] = 101;
    colvals[535] = 103;
    colvals[536] = 104;
    colvals[537] = 106;
    colvals[538] = 107;
    colvals[539] = 109;
    colvals[540] = 110;
    colvals[541] = 111;
    colvals[542] = 42;
    colvals[543] = 57;
    colvals[544] = 63;
    colvals[545] = 65;
    colvals[546] = 69;
    colvals[547] = 73;
    colvals[548] = 80;
    colvals[549] = 87;
    colvals[550] = 88;
    colvals[551] = 89;
    colvals[552] = 90;
    colvals[553] = 92;
    colvals[554] = 94;
    colvals[555] = 95;
    colvals[556] = 96;
    colvals[557] = 97;
    colvals[558] = 98;
    colvals[559] = 101;
    colvals[560] = 102;
    colvals[561] = 103;
    colvals[562] = 104;
    colvals[563] = 105;
    colvals[564] = 108;
    colvals[565] = 109;
    colvals[566] = 110;
    colvals[567] = 111;
    colvals[568] = 112;
    colvals[569] = 113;
    colvals[570] = 58;
    colvals[571] = 59;
    colvals[572] = 60;
    colvals[573] = 63;
    colvals[574] = 64;
    colvals[575] = 66;
    colvals[576] = 67;
    colvals[577] = 70;
    colvals[578] = 73;
    colvals[579] = 75;
    colvals[580] = 76;
    colvals[581] = 77;
    colvals[582] = 79;
    colvals[583] = 80;
    colvals[584] = 82;
    colvals[585] = 85;
    colvals[586] = 88;
    colvals[587] = 89;
    colvals[588] = 90;
    colvals[589] = 91;
    colvals[590] = 92;
    colvals[591] = 93;
    colvals[592] = 95;
    colvals[593] = 96;
    colvals[594] = 98;
    colvals[595] = 99;
    colvals[596] = 107;
    colvals[597] = 110;
    colvals[598] = 112;
    colvals[599] = 59;
    colvals[600] = 60;
    colvals[601] = 63;
    colvals[602] = 64;
    colvals[603] = 65;
    colvals[604] = 69;
    colvals[605] = 70;
    colvals[606] = 72;
    colvals[607] = 73;
    colvals[608] = 76;
    colvals[609] = 77;
    colvals[610] = 80;
    colvals[611] = 86;
    colvals[612] = 88;
    colvals[613] = 90;
    colvals[614] = 92;
    colvals[615] = 93;
    colvals[616] = 95;
    colvals[617] = 96;
    colvals[618] = 98;
    colvals[619] = 99;
    colvals[620] = 101;
    colvals[621] = 102;
    colvals[622] = 103;
    colvals[623] = 105;
    colvals[624] = 108;
    colvals[625] = 109;
    colvals[626] = 110;
    colvals[627] = 111;
    colvals[628] = 112;
    colvals[629] = 60;
    colvals[630] = 63;
    colvals[631] = 69;
    colvals[632] = 72;
    colvals[633] = 77;
    colvals[634] = 78;
    colvals[635] = 80;
    colvals[636] = 86;
    colvals[637] = 88;
    colvals[638] = 90;
    colvals[639] = 92;
    colvals[640] = 94;
    colvals[641] = 95;
    colvals[642] = 96;
    colvals[643] = 98;
    colvals[644] = 99;
    colvals[645] = 101;
    colvals[646] = 103;
    colvals[647] = 104;
    colvals[648] = 105;
    colvals[649] = 108;
    colvals[650] = 109;
    colvals[651] = 110;
    colvals[652] = 111;
    colvals[653] = 112;
    colvals[654] = 36;
    colvals[655] = 38;
    colvals[656] = 43;
    colvals[657] = 45;
    colvals[658] = 46;
    colvals[659] = 47;
    colvals[660] = 49;
    colvals[661] = 52;
    colvals[662] = 53;
    colvals[663] = 56;
    colvals[664] = 61;
    colvals[665] = 62;
    colvals[666] = 78;
    colvals[667] = 82;
    colvals[668] = 83;
    colvals[669] = 84;
    colvals[670] = 87;
    colvals[671] = 89;
    colvals[672] = 90;
    colvals[673] = 91;
    colvals[674] = 97;
    colvals[675] = 98;
    colvals[676] = 102;
    colvals[677] = 103;
    colvals[678] = 105;
    colvals[679] = 106;
    colvals[680] = 108;
    colvals[681] = 109;
    colvals[682] = 110;
    colvals[683] = 111;
    colvals[684] = 112;
    colvals[685] = 113;
    colvals[686] = 36;
    colvals[687] = 38;
    colvals[688] = 41;
    colvals[689] = 44;
    colvals[690] = 45;
    colvals[691] = 47;
    colvals[692] = 49;
    colvals[693] = 53;
    colvals[694] = 56;
    colvals[695] = 61;
    colvals[696] = 62;
    colvals[697] = 72;
    colvals[698] = 78;
    colvals[699] = 82;
    colvals[700] = 83;
    colvals[701] = 84;
    colvals[702] = 85;
    colvals[703] = 87;
    colvals[704] = 89;
    colvals[705] = 91;
    colvals[706] = 93;
    colvals[707] = 97;
    colvals[708] = 98;
    colvals[709] = 99;
    colvals[710] = 101;
    colvals[711] = 102;
    colvals[712] = 103;
    colvals[713] = 104;
    colvals[714] = 106;
    colvals[715] = 107;
    colvals[716] = 108;
    colvals[717] = 109;
    colvals[718] = 110;
    colvals[719] = 111;
    colvals[720] = 1;
    colvals[721] = 27;
    colvals[722] = 58;
    colvals[723] = 59;
    colvals[724] = 60;
    colvals[725] = 63;
    colvals[726] = 64;
    colvals[727] = 66;
    colvals[728] = 75;
    colvals[729] = 76;
    colvals[730] = 77;
    colvals[731] = 80;
    colvals[732] = 82;
    colvals[733] = 85;
    colvals[734] = 89;
    colvals[735] = 90;
    colvals[736] = 91;
    colvals[737] = 93;
    colvals[738] = 95;
    colvals[739] = 97;
    colvals[740] = 98;
    colvals[741] = 99;
    colvals[742] = 102;
    colvals[743] = 105;
    colvals[744] = 106;
    colvals[745] = 107;
    colvals[746] = 110;
    colvals[747] = 113;
    colvals[748] = 60;
    colvals[749] = 63;
    colvals[750] = 64;
    colvals[751] = 69;
    colvals[752] = 72;
    colvals[753] = 76;
    colvals[754] = 77;
    colvals[755] = 78;
    colvals[756] = 80;
    colvals[757] = 82;
    colvals[758] = 83;
    colvals[759] = 86;
    colvals[760] = 88;
    colvals[761] = 90;
    colvals[762] = 92;
    colvals[763] = 93;
    colvals[764] = 94;
    colvals[765] = 95;
    colvals[766] = 96;
    colvals[767] = 98;
    colvals[768] = 99;
    colvals[769] = 101;
    colvals[770] = 102;
    colvals[771] = 103;
    colvals[772] = 104;
    colvals[773] = 105;
    colvals[774] = 108;
    colvals[775] = 109;
    colvals[776] = 110;
    colvals[777] = 111;
    colvals[778] = 53;
    colvals[779] = 57;
    colvals[780] = 65;
    colvals[781] = 67;
    colvals[782] = 70;
    colvals[783] = 73;
    colvals[784] = 77;
    colvals[785] = 80;
    colvals[786] = 86;
    colvals[787] = 88;
    colvals[788] = 90;
    colvals[789] = 92;
    colvals[790] = 94;
    colvals[791] = 95;
    colvals[792] = 96;
    colvals[793] = 97;
    colvals[794] = 98;
    colvals[795] = 101;
    colvals[796] = 102;
    colvals[797] = 103;
    colvals[798] = 104;
    colvals[799] = 105;
    colvals[800] = 108;
    colvals[801] = 109;
    colvals[802] = 110;
    colvals[803] = 111;
    colvals[804] = 112;
    colvals[805] = 52;
    colvals[806] = 55;
    colvals[807] = 58;
    colvals[808] = 59;
    colvals[809] = 60;
    colvals[810] = 63;
    colvals[811] = 64;
    colvals[812] = 66;
    colvals[813] = 67;
    colvals[814] = 73;
    colvals[815] = 74;
    colvals[816] = 75;
    colvals[817] = 76;
    colvals[818] = 77;
    colvals[819] = 78;
    colvals[820] = 79;
    colvals[821] = 80;
    colvals[822] = 81;
    colvals[823] = 82;
    colvals[824] = 85;
    colvals[825] = 87;
    colvals[826] = 88;
    colvals[827] = 89;
    colvals[828] = 91;
    colvals[829] = 92;
    colvals[830] = 93;
    colvals[831] = 95;
    colvals[832] = 96;
    colvals[833] = 98;
    colvals[834] = 99;
    colvals[835] = 104;
    colvals[836] = 106;
    colvals[837] = 107;
    colvals[838] = 108;
    colvals[839] = 110;
    colvals[840] = 24;
    colvals[841] = 37;
    colvals[842] = 46;
    colvals[843] = 51;
    colvals[844] = 52;
    colvals[845] = 55;
    colvals[846] = 65;
    colvals[847] = 67;
    colvals[848] = 68;
    colvals[849] = 69;
    colvals[850] = 70;
    colvals[851] = 71;
    colvals[852] = 73;
    colvals[853] = 74;
    colvals[854] = 75;
    colvals[855] = 79;
    colvals[856] = 80;
    colvals[857] = 82;
    colvals[858] = 90;
    colvals[859] = 91;
    colvals[860] = 92;
    colvals[861] = 93;
    colvals[862] = 94;
    colvals[863] = 95;
    colvals[864] = 96;
    colvals[865] = 97;
    colvals[866] = 103;
    colvals[867] = 104;
    colvals[868] = 106;
    colvals[869] = 108;
    colvals[870] = 109;
    colvals[871] = 112;
    colvals[872] = 113;
    colvals[873] = 42;
    colvals[874] = 57;
    colvals[875] = 65;
    colvals[876] = 67;
    colvals[877] = 68;
    colvals[878] = 69;
    colvals[879] = 70;
    colvals[880] = 72;
    colvals[881] = 73;
    colvals[882] = 74;
    colvals[883] = 78;
    colvals[884] = 80;
    colvals[885] = 87;
    colvals[886] = 88;
    colvals[887] = 89;
    colvals[888] = 90;
    colvals[889] = 92;
    colvals[890] = 95;
    colvals[891] = 96;
    colvals[892] = 97;
    colvals[893] = 98;
    colvals[894] = 101;
    colvals[895] = 103;
    colvals[896] = 104;
    colvals[897] = 105;
    colvals[898] = 106;
    colvals[899] = 107;
    colvals[900] = 108;
    colvals[901] = 109;
    colvals[902] = 110;
    colvals[903] = 111;
    colvals[904] = 112;
    colvals[905] = 113;
    colvals[906] = 29;
    colvals[907] = 67;
    colvals[908] = 69;
    colvals[909] = 70;
    colvals[910] = 72;
    colvals[911] = 80;
    colvals[912] = 86;
    colvals[913] = 88;
    colvals[914] = 90;
    colvals[915] = 92;
    colvals[916] = 94;
    colvals[917] = 95;
    colvals[918] = 96;
    colvals[919] = 97;
    colvals[920] = 98;
    colvals[921] = 99;
    colvals[922] = 100;
    colvals[923] = 101;
    colvals[924] = 102;
    colvals[925] = 103;
    colvals[926] = 104;
    colvals[927] = 105;
    colvals[928] = 106;
    colvals[929] = 108;
    colvals[930] = 109;
    colvals[931] = 110;
    colvals[932] = 111;
    colvals[933] = 112;
    colvals[934] = 113;
    colvals[935] = 21;
    colvals[936] = 53;
    colvals[937] = 55;
    colvals[938] = 62;
    colvals[939] = 65;
    colvals[940] = 67;
    colvals[941] = 68;
    colvals[942] = 69;
    colvals[943] = 70;
    colvals[944] = 71;
    colvals[945] = 73;
    colvals[946] = 74;
    colvals[947] = 75;
    colvals[948] = 76;
    colvals[949] = 77;
    colvals[950] = 78;
    colvals[951] = 80;
    colvals[952] = 82;
    colvals[953] = 83;
    colvals[954] = 87;
    colvals[955] = 89;
    colvals[956] = 91;
    colvals[957] = 92;
    colvals[958] = 93;
    colvals[959] = 94;
    colvals[960] = 96;
    colvals[961] = 97;
    colvals[962] = 101;
    colvals[963] = 103;
    colvals[964] = 106;
    colvals[965] = 109;
    colvals[966] = 112;
    colvals[967] = 113;
    colvals[968] = 37;
    colvals[969] = 39;
    colvals[970] = 51;
    colvals[971] = 52;
    colvals[972] = 53;
    colvals[973] = 55;
    colvals[974] = 61;
    colvals[975] = 67;
    colvals[976] = 68;
    colvals[977] = 70;
    colvals[978] = 71;
    colvals[979] = 72;
    colvals[980] = 73;
    colvals[981] = 75;
    colvals[982] = 79;
    colvals[983] = 80;
    colvals[984] = 81;
    colvals[985] = 82;
    colvals[986] = 90;
    colvals[987] = 91;
    colvals[988] = 92;
    colvals[989] = 94;
    colvals[990] = 95;
    colvals[991] = 96;
    colvals[992] = 97;
    colvals[993] = 98;
    colvals[994] = 99;
    colvals[995] = 101;
    colvals[996] = 102;
    colvals[997] = 103;
    colvals[998] = 104;
    colvals[999] = 106;
    colvals[1000] = 108;
    colvals[1001] = 109;
    colvals[1002] = 110;
    colvals[1003] = 111;
    colvals[1004] = 112;
    colvals[1005] = 113;
    colvals[1006] = 14;
    colvals[1007] = 37;
    colvals[1008] = 39;
    colvals[1009] = 42;
    colvals[1010] = 49;
    colvals[1011] = 50;
    colvals[1012] = 51;
    colvals[1013] = 55;
    colvals[1014] = 59;
    colvals[1015] = 60;
    colvals[1016] = 62;
    colvals[1017] = 64;
    colvals[1018] = 69;
    colvals[1019] = 72;
    colvals[1020] = 73;
    colvals[1021] = 74;
    colvals[1022] = 75;
    colvals[1023] = 77;
    colvals[1024] = 81;
    colvals[1025] = 87;
    colvals[1026] = 89;
    colvals[1027] = 90;
    colvals[1028] = 93;
    colvals[1029] = 96;
    colvals[1030] = 97;
    colvals[1031] = 98;
    colvals[1032] = 99;
    colvals[1033] = 101;
    colvals[1034] = 102;
    colvals[1035] = 103;
    colvals[1036] = 104;
    colvals[1037] = 105;
    colvals[1038] = 106;
    colvals[1039] = 108;
    colvals[1040] = 109;
    colvals[1041] = 110;
    colvals[1042] = 111;
    colvals[1043] = 113;
    colvals[1044] = 52;
    colvals[1045] = 53;
    colvals[1046] = 65;
    colvals[1047] = 67;
    colvals[1048] = 70;
    colvals[1049] = 72;
    colvals[1050] = 73;
    colvals[1051] = 80;
    colvals[1052] = 86;
    colvals[1053] = 88;
    colvals[1054] = 90;
    colvals[1055] = 92;
    colvals[1056] = 94;
    colvals[1057] = 95;
    colvals[1058] = 96;
    colvals[1059] = 97;
    colvals[1060] = 98;
    colvals[1061] = 101;
    colvals[1062] = 102;
    colvals[1063] = 103;
    colvals[1064] = 104;
    colvals[1065] = 108;
    colvals[1066] = 110;
    colvals[1067] = 111;
    colvals[1068] = 112;
    colvals[1069] = 39;
    colvals[1070] = 42;
    colvals[1071] = 52;
    colvals[1072] = 56;
    colvals[1073] = 57;
    colvals[1074] = 65;
    colvals[1075] = 67;
    colvals[1076] = 68;
    colvals[1077] = 70;
    colvals[1078] = 72;
    colvals[1079] = 73;
    colvals[1080] = 74;
    colvals[1081] = 78;
    colvals[1082] = 80;
    colvals[1083] = 86;
    colvals[1084] = 87;
    colvals[1085] = 88;
    colvals[1086] = 89;
    colvals[1087] = 90;
    colvals[1088] = 92;
    colvals[1089] = 93;
    colvals[1090] = 94;
    colvals[1091] = 95;
    colvals[1092] = 96;
    colvals[1093] = 97;
    colvals[1094] = 98;
    colvals[1095] = 101;
    colvals[1096] = 103;
    colvals[1097] = 104;
    colvals[1098] = 105;
    colvals[1099] = 106;
    colvals[1100] = 108;
    colvals[1101] = 109;
    colvals[1102] = 110;
    colvals[1103] = 111;
    colvals[1104] = 112;
    colvals[1105] = 113;
    colvals[1106] = 57;
    colvals[1107] = 60;
    colvals[1108] = 63;
    colvals[1109] = 64;
    colvals[1110] = 65;
    colvals[1111] = 67;
    colvals[1112] = 68;
    colvals[1113] = 69;
    colvals[1114] = 70;
    colvals[1115] = 72;
    colvals[1116] = 73;
    colvals[1117] = 75;
    colvals[1118] = 77;
    colvals[1119] = 80;
    colvals[1120] = 81;
    colvals[1121] = 87;
    colvals[1122] = 88;
    colvals[1123] = 89;
    colvals[1124] = 90;
    colvals[1125] = 92;
    colvals[1126] = 93;
    colvals[1127] = 94;
    colvals[1128] = 95;
    colvals[1129] = 96;
    colvals[1130] = 98;
    colvals[1131] = 99;
    colvals[1132] = 101;
    colvals[1133] = 102;
    colvals[1134] = 103;
    colvals[1135] = 104;
    colvals[1136] = 105;
    colvals[1137] = 106;
    colvals[1138] = 108;
    colvals[1139] = 110;
    colvals[1140] = 111;
    colvals[1141] = 112;
    colvals[1142] = 113;
    colvals[1143] = 27;
    colvals[1144] = 57;
    colvals[1145] = 59;
    colvals[1146] = 60;
    colvals[1147] = 63;
    colvals[1148] = 64;
    colvals[1149] = 65;
    colvals[1150] = 68;
    colvals[1151] = 69;
    colvals[1152] = 70;
    colvals[1153] = 73;
    colvals[1154] = 74;
    colvals[1155] = 75;
    colvals[1156] = 76;
    colvals[1157] = 77;
    colvals[1158] = 78;
    colvals[1159] = 80;
    colvals[1160] = 82;
    colvals[1161] = 88;
    colvals[1162] = 90;
    colvals[1163] = 92;
    colvals[1164] = 93;
    colvals[1165] = 95;
    colvals[1166] = 96;
    colvals[1167] = 97;
    colvals[1168] = 98;
    colvals[1169] = 99;
    colvals[1170] = 101;
    colvals[1171] = 102;
    colvals[1172] = 104;
    colvals[1173] = 106;
    colvals[1174] = 107;
    colvals[1175] = 108;
    colvals[1176] = 109;
    colvals[1177] = 110;
    colvals[1178] = 112;
    colvals[1179] = 57;
    colvals[1180] = 63;
    colvals[1181] = 65;
    colvals[1182] = 68;
    colvals[1183] = 69;
    colvals[1184] = 70;
    colvals[1185] = 72;
    colvals[1186] = 73;
    colvals[1187] = 74;
    colvals[1188] = 77;
    colvals[1189] = 80;
    colvals[1190] = 86;
    colvals[1191] = 88;
    colvals[1192] = 90;
    colvals[1193] = 92;
    colvals[1194] = 94;
    colvals[1195] = 95;
    colvals[1196] = 96;
    colvals[1197] = 97;
    colvals[1198] = 98;
    colvals[1199] = 101;
    colvals[1200] = 102;
    colvals[1201] = 103;
    colvals[1202] = 104;
    colvals[1203] = 105;
    colvals[1204] = 106;
    colvals[1205] = 108;
    colvals[1206] = 109;
    colvals[1207] = 110;
    colvals[1208] = 111;
    colvals[1209] = 112;
    colvals[1210] = 52;
    colvals[1211] = 53;
    colvals[1212] = 55;
    colvals[1213] = 57;
    colvals[1214] = 62;
    colvals[1215] = 65;
    colvals[1216] = 68;
    colvals[1217] = 69;
    colvals[1218] = 70;
    colvals[1219] = 72;
    colvals[1220] = 73;
    colvals[1221] = 74;
    colvals[1222] = 75;
    colvals[1223] = 77;
    colvals[1224] = 78;
    colvals[1225] = 80;
    colvals[1226] = 82;
    colvals[1227] = 90;
    colvals[1228] = 92;
    colvals[1229] = 93;
    colvals[1230] = 95;
    colvals[1231] = 96;
    colvals[1232] = 97;
    colvals[1233] = 98;
    colvals[1234] = 101;
    colvals[1235] = 102;
    colvals[1236] = 103;
    colvals[1237] = 104;
    colvals[1238] = 105;
    colvals[1239] = 106;
    colvals[1240] = 109;
    colvals[1241] = 110;
    colvals[1242] = 46;
    colvals[1243] = 52;
    colvals[1244] = 53;
    colvals[1245] = 55;
    colvals[1246] = 58;
    colvals[1247] = 59;
    colvals[1248] = 60;
    colvals[1249] = 64;
    colvals[1250] = 65;
    colvals[1251] = 67;
    colvals[1252] = 69;
    colvals[1253] = 71;
    colvals[1254] = 73;
    colvals[1255] = 74;
    colvals[1256] = 75;
    colvals[1257] = 76;
    colvals[1258] = 77;
    colvals[1259] = 79;
    colvals[1260] = 81;
    colvals[1261] = 82;
    colvals[1262] = 83;
    colvals[1263] = 85;
    colvals[1264] = 87;
    colvals[1265] = 89;
    colvals[1266] = 90;
    colvals[1267] = 91;
    colvals[1268] = 92;
    colvals[1269] = 93;
    colvals[1270] = 95;
    colvals[1271] = 96;
    colvals[1272] = 97;
    colvals[1273] = 99;
    colvals[1274] = 101;
    colvals[1275] = 103;
    colvals[1276] = 104;
    colvals[1277] = 106;
    colvals[1278] = 107;
    colvals[1279] = 109;
    colvals[1280] = 110;
    colvals[1281] = 112;
    colvals[1282] = 113;
    colvals[1283] = 51;
    colvals[1284] = 57;
    colvals[1285] = 58;
    colvals[1286] = 59;
    colvals[1287] = 60;
    colvals[1288] = 64;
    colvals[1289] = 65;
    colvals[1290] = 66;
    colvals[1291] = 67;
    colvals[1292] = 68;
    colvals[1293] = 69;
    colvals[1294] = 70;
    colvals[1295] = 71;
    colvals[1296] = 73;
    colvals[1297] = 74;
    colvals[1298] = 75;
    colvals[1299] = 76;
    colvals[1300] = 77;
    colvals[1301] = 78;
    colvals[1302] = 80;
    colvals[1303] = 82;
    colvals[1304] = 83;
    colvals[1305] = 85;
    colvals[1306] = 87;
    colvals[1307] = 88;
    colvals[1308] = 89;
    colvals[1309] = 90;
    colvals[1310] = 91;
    colvals[1311] = 92;
    colvals[1312] = 93;
    colvals[1313] = 94;
    colvals[1314] = 95;
    colvals[1315] = 96;
    colvals[1316] = 97;
    colvals[1317] = 98;
    colvals[1318] = 99;
    colvals[1319] = 101;
    colvals[1320] = 103;
    colvals[1321] = 105;
    colvals[1322] = 106;
    colvals[1323] = 107;
    colvals[1324] = 108;
    colvals[1325] = 109;
    colvals[1326] = 110;
    colvals[1327] = 112;
    colvals[1328] = 113;
    colvals[1329] = 45;
    colvals[1330] = 55;
    colvals[1331] = 57;
    colvals[1332] = 58;
    colvals[1333] = 59;
    colvals[1334] = 60;
    colvals[1335] = 64;
    colvals[1336] = 65;
    colvals[1337] = 66;
    colvals[1338] = 67;
    colvals[1339] = 68;
    colvals[1340] = 69;
    colvals[1341] = 72;
    colvals[1342] = 73;
    colvals[1343] = 74;
    colvals[1344] = 75;
    colvals[1345] = 76;
    colvals[1346] = 77;
    colvals[1347] = 78;
    colvals[1348] = 79;
    colvals[1349] = 81;
    colvals[1350] = 82;
    colvals[1351] = 85;
    colvals[1352] = 87;
    colvals[1353] = 89;
    colvals[1354] = 90;
    colvals[1355] = 91;
    colvals[1356] = 93;
    colvals[1357] = 95;
    colvals[1358] = 96;
    colvals[1359] = 97;
    colvals[1360] = 98;
    colvals[1361] = 99;
    colvals[1362] = 101;
    colvals[1363] = 102;
    colvals[1364] = 104;
    colvals[1365] = 106;
    colvals[1366] = 107;
    colvals[1367] = 108;
    colvals[1368] = 109;
    colvals[1369] = 110;
    colvals[1370] = 112;
    colvals[1371] = 113;
    colvals[1372] = 53;
    colvals[1373] = 59;
    colvals[1374] = 60;
    colvals[1375] = 62;
    colvals[1376] = 63;
    colvals[1377] = 64;
    colvals[1378] = 65;
    colvals[1379] = 67;
    colvals[1380] = 68;
    colvals[1381] = 69;
    colvals[1382] = 70;
    colvals[1383] = 73;
    colvals[1384] = 74;
    colvals[1385] = 75;
    colvals[1386] = 77;
    colvals[1387] = 80;
    colvals[1388] = 82;
    colvals[1389] = 88;
    colvals[1390] = 90;
    colvals[1391] = 92;
    colvals[1392] = 93;
    colvals[1393] = 95;
    colvals[1394] = 96;
    colvals[1395] = 97;
    colvals[1396] = 98;
    colvals[1397] = 99;
    colvals[1398] = 101;
    colvals[1399] = 102;
    colvals[1400] = 103;
    colvals[1401] = 104;
    colvals[1402] = 105;
    colvals[1403] = 106;
    colvals[1404] = 107;
    colvals[1405] = 108;
    colvals[1406] = 109;
    colvals[1407] = 110;
    colvals[1408] = 111;
    colvals[1409] = 112;
    colvals[1410] = 53;
    colvals[1411] = 55;
    colvals[1412] = 58;
    colvals[1413] = 59;
    colvals[1414] = 60;
    colvals[1415] = 62;
    colvals[1416] = 64;
    colvals[1417] = 65;
    colvals[1418] = 66;
    colvals[1419] = 68;
    colvals[1420] = 69;
    colvals[1421] = 70;
    colvals[1422] = 73;
    colvals[1423] = 74;
    colvals[1424] = 75;
    colvals[1425] = 76;
    colvals[1426] = 77;
    colvals[1427] = 78;
    colvals[1428] = 80;
    colvals[1429] = 82;
    colvals[1430] = 83;
    colvals[1431] = 85;
    colvals[1432] = 87;
    colvals[1433] = 89;
    colvals[1434] = 90;
    colvals[1435] = 91;
    colvals[1436] = 92;
    colvals[1437] = 93;
    colvals[1438] = 95;
    colvals[1439] = 96;
    colvals[1440] = 97;
    colvals[1441] = 99;
    colvals[1442] = 101;
    colvals[1443] = 106;
    colvals[1444] = 107;
    colvals[1445] = 108;
    colvals[1446] = 109;
    colvals[1447] = 110;
    colvals[1448] = 112;
    colvals[1449] = 39;
    colvals[1450] = 49;
    colvals[1451] = 51;
    colvals[1452] = 52;
    colvals[1453] = 53;
    colvals[1454] = 57;
    colvals[1455] = 62;
    colvals[1456] = 64;
    colvals[1457] = 65;
    colvals[1458] = 68;
    colvals[1459] = 69;
    colvals[1460] = 72;
    colvals[1461] = 73;
    colvals[1462] = 74;
    colvals[1463] = 75;
    colvals[1464] = 76;
    colvals[1465] = 77;
    colvals[1466] = 78;
    colvals[1467] = 79;
    colvals[1468] = 81;
    colvals[1469] = 82;
    colvals[1470] = 83;
    colvals[1471] = 84;
    colvals[1472] = 86;
    colvals[1473] = 87;
    colvals[1474] = 88;
    colvals[1475] = 89;
    colvals[1476] = 91;
    colvals[1477] = 92;
    colvals[1478] = 93;
    colvals[1479] = 94;
    colvals[1480] = 95;
    colvals[1481] = 97;
    colvals[1482] = 99;
    colvals[1483] = 101;
    colvals[1484] = 102;
    colvals[1485] = 104;
    colvals[1486] = 106;
    colvals[1487] = 109;
    colvals[1488] = 110;
    colvals[1489] = 111;
    colvals[1490] = 34;
    colvals[1491] = 35;
    colvals[1492] = 43;
    colvals[1493] = 45;
    colvals[1494] = 47;
    colvals[1495] = 50;
    colvals[1496] = 55;
    colvals[1497] = 56;
    colvals[1498] = 59;
    colvals[1499] = 60;
    colvals[1500] = 62;
    colvals[1501] = 63;
    colvals[1502] = 64;
    colvals[1503] = 66;
    colvals[1504] = 67;
    colvals[1505] = 69;
    colvals[1506] = 75;
    colvals[1507] = 76;
    colvals[1508] = 77;
    colvals[1509] = 80;
    colvals[1510] = 82;
    colvals[1511] = 85;
    colvals[1512] = 88;
    colvals[1513] = 89;
    colvals[1514] = 90;
    colvals[1515] = 91;
    colvals[1516] = 92;
    colvals[1517] = 93;
    colvals[1518] = 95;
    colvals[1519] = 96;
    colvals[1520] = 98;
    colvals[1521] = 99;
    colvals[1522] = 103;
    colvals[1523] = 105;
    colvals[1524] = 107;
    colvals[1525] = 108;
    colvals[1526] = 110;
    colvals[1527] = 112;
    colvals[1528] = 17;
    colvals[1529] = 39;
    colvals[1530] = 42;
    colvals[1531] = 53;
    colvals[1532] = 59;
    colvals[1533] = 60;
    colvals[1534] = 63;
    colvals[1535] = 64;
    colvals[1536] = 65;
    colvals[1537] = 67;
    colvals[1538] = 69;
    colvals[1539] = 70;
    colvals[1540] = 72;
    colvals[1541] = 74;
    colvals[1542] = 77;
    colvals[1543] = 80;
    colvals[1544] = 86;
    colvals[1545] = 88;
    colvals[1546] = 90;
    colvals[1547] = 92;
    colvals[1548] = 93;
    colvals[1549] = 94;
    colvals[1550] = 95;
    colvals[1551] = 96;
    colvals[1552] = 97;
    colvals[1553] = 98;
    colvals[1554] = 99;
    colvals[1555] = 101;
    colvals[1556] = 102;
    colvals[1557] = 103;
    colvals[1558] = 104;
    colvals[1559] = 105;
    colvals[1560] = 108;
    colvals[1561] = 109;
    colvals[1562] = 110;
    colvals[1563] = 111;
    colvals[1564] = 32;
    colvals[1565] = 33;
    colvals[1566] = 38;
    colvals[1567] = 42;
    colvals[1568] = 43;
    colvals[1569] = 47;
    colvals[1570] = 48;
    colvals[1571] = 52;
    colvals[1572] = 53;
    colvals[1573] = 56;
    colvals[1574] = 57;
    colvals[1575] = 62;
    colvals[1576] = 63;
    colvals[1577] = 65;
    colvals[1578] = 68;
    colvals[1579] = 70;
    colvals[1580] = 72;
    colvals[1581] = 78;
    colvals[1582] = 80;
    colvals[1583] = 81;
    colvals[1584] = 87;
    colvals[1585] = 88;
    colvals[1586] = 89;
    colvals[1587] = 90;
    colvals[1588] = 92;
    colvals[1589] = 94;
    colvals[1590] = 95;
    colvals[1591] = 96;
    colvals[1592] = 97;
    colvals[1593] = 98;
    colvals[1594] = 101;
    colvals[1595] = 102;
    colvals[1596] = 103;
    colvals[1597] = 104;
    colvals[1598] = 105;
    colvals[1599] = 108;
    colvals[1600] = 109;
    colvals[1601] = 110;
    colvals[1602] = 111;
    colvals[1603] = 112;
    colvals[1604] = 113;
    colvals[1605] = 19;
    colvals[1606] = 23;
    colvals[1607] = 42;
    colvals[1608] = 46;
    colvals[1609] = 51;
    colvals[1610] = 57;
    colvals[1611] = 58;
    colvals[1612] = 59;
    colvals[1613] = 60;
    colvals[1614] = 63;
    colvals[1615] = 64;
    colvals[1616] = 65;
    colvals[1617] = 66;
    colvals[1618] = 67;
    colvals[1619] = 68;
    colvals[1620] = 69;
    colvals[1621] = 70;
    colvals[1622] = 71;
    colvals[1623] = 73;
    colvals[1624] = 74;
    colvals[1625] = 75;
    colvals[1626] = 76;
    colvals[1627] = 77;
    colvals[1628] = 80;
    colvals[1629] = 82;
    colvals[1630] = 85;
    colvals[1631] = 86;
    colvals[1632] = 88;
    colvals[1633] = 89;
    colvals[1634] = 90;
    colvals[1635] = 91;
    colvals[1636] = 92;
    colvals[1637] = 93;
    colvals[1638] = 94;
    colvals[1639] = 95;
    colvals[1640] = 96;
    colvals[1641] = 97;
    colvals[1642] = 98;
    colvals[1643] = 99;
    colvals[1644] = 101;
    colvals[1645] = 102;
    colvals[1646] = 103;
    colvals[1647] = 104;
    colvals[1648] = 105;
    colvals[1649] = 106;
    colvals[1650] = 107;
    colvals[1651] = 108;
    colvals[1652] = 109;
    colvals[1653] = 110;
    colvals[1654] = 112;
    colvals[1655] = 113;
    colvals[1656] = 50;
    colvals[1657] = 52;
    colvals[1658] = 53;
    colvals[1659] = 57;
    colvals[1660] = 59;
    colvals[1661] = 60;
    colvals[1662] = 62;
    colvals[1663] = 63;
    colvals[1664] = 64;
    colvals[1665] = 65;
    colvals[1666] = 67;
    colvals[1667] = 68;
    colvals[1668] = 69;
    colvals[1669] = 70;
    colvals[1670] = 71;
    colvals[1671] = 72;
    colvals[1672] = 73;
    colvals[1673] = 74;
    colvals[1674] = 75;
    colvals[1675] = 76;
    colvals[1676] = 77;
    colvals[1677] = 78;
    colvals[1678] = 79;
    colvals[1679] = 80;
    colvals[1680] = 81;
    colvals[1681] = 82;
    colvals[1682] = 87;
    colvals[1683] = 88;
    colvals[1684] = 89;
    colvals[1685] = 90;
    colvals[1686] = 91;
    colvals[1687] = 92;
    colvals[1688] = 93;
    colvals[1689] = 95;
    colvals[1690] = 96;
    colvals[1691] = 97;
    colvals[1692] = 98;
    colvals[1693] = 99;
    colvals[1694] = 101;
    colvals[1695] = 102;
    colvals[1696] = 103;
    colvals[1697] = 104;
    colvals[1698] = 105;
    colvals[1699] = 106;
    colvals[1700] = 107;
    colvals[1701] = 108;
    colvals[1702] = 109;
    colvals[1703] = 110;
    colvals[1704] = 112;
    colvals[1705] = 113;
    colvals[1706] = 37;
    colvals[1707] = 39;
    colvals[1708] = 49;
    colvals[1709] = 51;
    colvals[1710] = 52;
    colvals[1711] = 55;
    colvals[1712] = 57;
    colvals[1713] = 58;
    colvals[1714] = 59;
    colvals[1715] = 60;
    colvals[1716] = 64;
    colvals[1717] = 65;
    colvals[1718] = 66;
    colvals[1719] = 67;
    colvals[1720] = 68;
    colvals[1721] = 69;
    colvals[1722] = 71;
    colvals[1723] = 73;
    colvals[1724] = 74;
    colvals[1725] = 75;
    colvals[1726] = 76;
    colvals[1727] = 77;
    colvals[1728] = 78;
    colvals[1729] = 79;
    colvals[1730] = 81;
    colvals[1731] = 82;
    colvals[1732] = 83;
    colvals[1733] = 85;
    colvals[1734] = 86;
    colvals[1735] = 87;
    colvals[1736] = 89;
    colvals[1737] = 90;
    colvals[1738] = 91;
    colvals[1739] = 93;
    colvals[1740] = 94;
    colvals[1741] = 95;
    colvals[1742] = 96;
    colvals[1743] = 97;
    colvals[1744] = 98;
    colvals[1745] = 99;
    colvals[1746] = 101;
    colvals[1747] = 102;
    colvals[1748] = 103;
    colvals[1749] = 104;
    colvals[1750] = 105;
    colvals[1751] = 106;
    colvals[1752] = 107;
    colvals[1753] = 109;
    colvals[1754] = 110;
    colvals[1755] = 112;
    colvals[1756] = 113;
    colvals[1757] = 52;
    colvals[1758] = 53;
    colvals[1759] = 55;
    colvals[1760] = 57;
    colvals[1761] = 59;
    colvals[1762] = 60;
    colvals[1763] = 62;
    colvals[1764] = 63;
    colvals[1765] = 64;
    colvals[1766] = 65;
    colvals[1767] = 67;
    colvals[1768] = 68;
    colvals[1769] = 69;
    colvals[1770] = 70;
    colvals[1771] = 72;
    colvals[1772] = 73;
    colvals[1773] = 74;
    colvals[1774] = 75;
    colvals[1775] = 76;
    colvals[1776] = 77;
    colvals[1777] = 78;
    colvals[1778] = 79;
    colvals[1779] = 80;
    colvals[1780] = 81;
    colvals[1781] = 82;
    colvals[1782] = 87;
    colvals[1783] = 88;
    colvals[1784] = 89;
    colvals[1785] = 90;
    colvals[1786] = 91;
    colvals[1787] = 92;
    colvals[1788] = 93;
    colvals[1789] = 95;
    colvals[1790] = 96;
    colvals[1791] = 97;
    colvals[1792] = 98;
    colvals[1793] = 99;
    colvals[1794] = 101;
    colvals[1795] = 103;
    colvals[1796] = 104;
    colvals[1797] = 106;
    colvals[1798] = 107;
    colvals[1799] = 108;
    colvals[1800] = 109;
    colvals[1801] = 110;
    colvals[1802] = 23;
    colvals[1803] = 27;
    colvals[1804] = 37;
    colvals[1805] = 39;
    colvals[1806] = 42;
    colvals[1807] = 51;
    colvals[1808] = 52;
    colvals[1809] = 57;
    colvals[1810] = 59;
    colvals[1811] = 60;
    colvals[1812] = 63;
    colvals[1813] = 64;
    colvals[1814] = 65;
    colvals[1815] = 67;
    colvals[1816] = 68;
    colvals[1817] = 69;
    colvals[1818] = 70;
    colvals[1819] = 71;
    colvals[1820] = 73;
    colvals[1821] = 74;
    colvals[1822] = 75;
    colvals[1823] = 76;
    colvals[1824] = 77;
    colvals[1825] = 78;
    colvals[1826] = 79;
    colvals[1827] = 80;
    colvals[1828] = 82;
    colvals[1829] = 83;
    colvals[1830] = 86;
    colvals[1831] = 87;
    colvals[1832] = 88;
    colvals[1833] = 89;
    colvals[1834] = 90;
    colvals[1835] = 91;
    colvals[1836] = 92;
    colvals[1837] = 93;
    colvals[1838] = 94;
    colvals[1839] = 95;
    colvals[1840] = 96;
    colvals[1841] = 97;
    colvals[1842] = 98;
    colvals[1843] = 99;
    colvals[1844] = 101;
    colvals[1845] = 102;
    colvals[1846] = 103;
    colvals[1847] = 104;
    colvals[1848] = 105;
    colvals[1849] = 106;
    colvals[1850] = 107;
    colvals[1851] = 108;
    colvals[1852] = 109;
    colvals[1853] = 110;
    colvals[1854] = 111;
    colvals[1855] = 112;
    colvals[1856] = 113;
    colvals[1857] = 47;
    colvals[1858] = 52;
    colvals[1859] = 56;
    colvals[1860] = 57;
    colvals[1861] = 59;
    colvals[1862] = 60;
    colvals[1863] = 62;
    colvals[1864] = 63;
    colvals[1865] = 65;
    colvals[1866] = 67;
    colvals[1867] = 68;
    colvals[1868] = 69;
    colvals[1869] = 70;
    colvals[1870] = 72;
    colvals[1871] = 73;
    colvals[1872] = 74;
    colvals[1873] = 77;
    colvals[1874] = 80;
    colvals[1875] = 82;
    colvals[1876] = 86;
    colvals[1877] = 88;
    colvals[1878] = 90;
    colvals[1879] = 92;
    colvals[1880] = 93;
    colvals[1881] = 94;
    colvals[1882] = 95;
    colvals[1883] = 96;
    colvals[1884] = 97;
    colvals[1885] = 98;
    colvals[1886] = 99;
    colvals[1887] = 101;
    colvals[1888] = 102;
    colvals[1889] = 103;
    colvals[1890] = 104;
    colvals[1891] = 105;
    colvals[1892] = 106;
    colvals[1893] = 108;
    colvals[1894] = 109;
    colvals[1895] = 110;
    colvals[1896] = 111;
    colvals[1897] = 112;
    colvals[1898] = 36;
    colvals[1899] = 38;
    colvals[1900] = 39;
    colvals[1901] = 42;
    colvals[1902] = 46;
    colvals[1903] = 51;
    colvals[1904] = 57;
    colvals[1905] = 58;
    colvals[1906] = 60;
    colvals[1907] = 63;
    colvals[1908] = 64;
    colvals[1909] = 65;
    colvals[1910] = 67;
    colvals[1911] = 69;
    colvals[1912] = 70;
    colvals[1913] = 71;
    colvals[1914] = 72;
    colvals[1915] = 73;
    colvals[1916] = 74;
    colvals[1917] = 75;
    colvals[1918] = 77;
    colvals[1919] = 80;
    colvals[1920] = 86;
    colvals[1921] = 87;
    colvals[1922] = 88;
    colvals[1923] = 90;
    colvals[1924] = 92;
    colvals[1925] = 93;
    colvals[1926] = 94;
    colvals[1927] = 95;
    colvals[1928] = 96;
    colvals[1929] = 97;
    colvals[1930] = 98;
    colvals[1931] = 99;
    colvals[1932] = 101;
    colvals[1933] = 102;
    colvals[1934] = 103;
    colvals[1935] = 104;
    colvals[1936] = 105;
    colvals[1937] = 108;
    colvals[1938] = 109;
    colvals[1939] = 110;
    colvals[1940] = 111;
    colvals[1941] = 112;
    colvals[1942] = 113;
    colvals[1943] = 18;
    colvals[1944] = 37;
    colvals[1945] = 39;
    colvals[1946] = 49;
    colvals[1947] = 51;
    colvals[1948] = 52;
    colvals[1949] = 53;
    colvals[1950] = 55;
    colvals[1951] = 57;
    colvals[1952] = 58;
    colvals[1953] = 59;
    colvals[1954] = 60;
    colvals[1955] = 62;
    colvals[1956] = 63;
    colvals[1957] = 64;
    colvals[1958] = 65;
    colvals[1959] = 66;
    colvals[1960] = 68;
    colvals[1961] = 69;
    colvals[1962] = 70;
    colvals[1963] = 71;
    colvals[1964] = 73;
    colvals[1965] = 74;
    colvals[1966] = 75;
    colvals[1967] = 76;
    colvals[1968] = 77;
    colvals[1969] = 78;
    colvals[1970] = 79;
    colvals[1971] = 80;
    colvals[1972] = 81;
    colvals[1973] = 82;
    colvals[1974] = 85;
    colvals[1975] = 87;
    colvals[1976] = 88;
    colvals[1977] = 89;
    colvals[1978] = 90;
    colvals[1979] = 91;
    colvals[1980] = 93;
    colvals[1981] = 94;
    colvals[1982] = 95;
    colvals[1983] = 96;
    colvals[1984] = 97;
    colvals[1985] = 98;
    colvals[1986] = 99;
    colvals[1987] = 101;
    colvals[1988] = 103;
    colvals[1989] = 104;
    colvals[1990] = 106;
    colvals[1991] = 107;
    colvals[1992] = 108;
    colvals[1993] = 109;
    colvals[1994] = 110;
    colvals[1995] = 113;
    colvals[1996] = 37;
    colvals[1997] = 49;
    colvals[1998] = 51;
    colvals[1999] = 52;
    colvals[2000] = 53;
    colvals[2001] = 57;
    colvals[2002] = 59;
    colvals[2003] = 60;
    colvals[2004] = 63;
    colvals[2005] = 64;
    colvals[2006] = 65;
    colvals[2007] = 66;
    colvals[2008] = 68;
    colvals[2009] = 69;
    colvals[2010] = 71;
    colvals[2011] = 72;
    colvals[2012] = 73;
    colvals[2013] = 74;
    colvals[2014] = 75;
    colvals[2015] = 76;
    colvals[2016] = 77;
    colvals[2017] = 78;
    colvals[2018] = 79;
    colvals[2019] = 80;
    colvals[2020] = 81;
    colvals[2021] = 82;
    colvals[2022] = 83;
    colvals[2023] = 87;
    colvals[2024] = 88;
    colvals[2025] = 89;
    colvals[2026] = 90;
    colvals[2027] = 91;
    colvals[2028] = 93;
    colvals[2029] = 94;
    colvals[2030] = 95;
    colvals[2031] = 96;
    colvals[2032] = 97;
    colvals[2033] = 98;
    colvals[2034] = 99;
    colvals[2035] = 101;
    colvals[2036] = 102;
    colvals[2037] = 103;
    colvals[2038] = 104;
    colvals[2039] = 105;
    colvals[2040] = 106;
    colvals[2041] = 107;
    colvals[2042] = 108;
    colvals[2043] = 109;
    colvals[2044] = 110;
    colvals[2045] = 113;
    colvals[2046] = 26;
    colvals[2047] = 33;
    colvals[2048] = 38;
    colvals[2049] = 42;
    colvals[2050] = 43;
    colvals[2051] = 46;
    colvals[2052] = 47;
    colvals[2053] = 48;
    colvals[2054] = 51;
    colvals[2055] = 52;
    colvals[2056] = 56;
    colvals[2057] = 62;
    colvals[2058] = 63;
    colvals[2059] = 67;
    colvals[2060] = 70;
    colvals[2061] = 71;
    colvals[2062] = 72;
    colvals[2063] = 80;
    colvals[2064] = 86;
    colvals[2065] = 88;
    colvals[2066] = 90;
    colvals[2067] = 92;
    colvals[2068] = 94;
    colvals[2069] = 95;
    colvals[2070] = 96;
    colvals[2071] = 97;
    colvals[2072] = 98;
    colvals[2073] = 100;
    colvals[2074] = 101;
    colvals[2075] = 103;
    colvals[2076] = 104;
    colvals[2077] = 105;
    colvals[2078] = 108;
    colvals[2079] = 110;
    colvals[2080] = 111;
    colvals[2081] = 112;
    colvals[2082] = 113;
    colvals[2083] = 37;
    colvals[2084] = 45;
    colvals[2085] = 49;
    colvals[2086] = 51;
    colvals[2087] = 52;
    colvals[2088] = 53;
    colvals[2089] = 57;
    colvals[2090] = 58;
    colvals[2091] = 59;
    colvals[2092] = 60;
    colvals[2093] = 61;
    colvals[2094] = 62;
    colvals[2095] = 64;
    colvals[2096] = 65;
    colvals[2097] = 66;
    colvals[2098] = 67;
    colvals[2099] = 68;
    colvals[2100] = 69;
    colvals[2101] = 70;
    colvals[2102] = 71;
    colvals[2103] = 72;
    colvals[2104] = 73;
    colvals[2105] = 74;
    colvals[2106] = 75;
    colvals[2107] = 76;
    colvals[2108] = 77;
    colvals[2109] = 78;
    colvals[2110] = 79;
    colvals[2111] = 80;
    colvals[2112] = 81;
    colvals[2113] = 82;
    colvals[2114] = 85;
    colvals[2115] = 86;
    colvals[2116] = 87;
    colvals[2117] = 88;
    colvals[2118] = 89;
    colvals[2119] = 90;
    colvals[2120] = 91;
    colvals[2121] = 92;
    colvals[2122] = 93;
    colvals[2123] = 94;
    colvals[2124] = 95;
    colvals[2125] = 96;
    colvals[2126] = 97;
    colvals[2127] = 98;
    colvals[2128] = 99;
    colvals[2129] = 101;
    colvals[2130] = 102;
    colvals[2131] = 103;
    colvals[2132] = 104;
    colvals[2133] = 105;
    colvals[2134] = 106;
    colvals[2135] = 107;
    colvals[2136] = 109;
    colvals[2137] = 110;
    colvals[2138] = 112;
    colvals[2139] = 113;
    colvals[2140] = 29;
    colvals[2141] = 39;
    colvals[2142] = 43;
    colvals[2143] = 46;
    colvals[2144] = 47;
    colvals[2145] = 48;
    colvals[2146] = 51;
    colvals[2147] = 52;
    colvals[2148] = 53;
    colvals[2149] = 56;
    colvals[2150] = 60;
    colvals[2151] = 62;
    colvals[2152] = 63;
    colvals[2153] = 69;
    colvals[2154] = 71;
    colvals[2155] = 72;
    colvals[2156] = 77;
    colvals[2157] = 80;
    colvals[2158] = 86;
    colvals[2159] = 88;
    colvals[2160] = 90;
    colvals[2161] = 92;
    colvals[2162] = 94;
    colvals[2163] = 95;
    colvals[2164] = 96;
    colvals[2165] = 98;
    colvals[2166] = 99;
    colvals[2167] = 101;
    colvals[2168] = 103;
    colvals[2169] = 104;
    colvals[2170] = 105;
    colvals[2171] = 108;
    colvals[2172] = 109;
    colvals[2173] = 110;
    colvals[2174] = 111;
    colvals[2175] = 112;
    colvals[2176] = 26;
    colvals[2177] = 29;
    colvals[2178] = 33;
    colvals[2179] = 38;
    colvals[2180] = 42;
    colvals[2181] = 43;
    colvals[2182] = 46;
    colvals[2183] = 47;
    colvals[2184] = 48;
    colvals[2185] = 51;
    colvals[2186] = 52;
    colvals[2187] = 56;
    colvals[2188] = 62;
    colvals[2189] = 63;
    colvals[2190] = 67;
    colvals[2191] = 69;
    colvals[2192] = 70;
    colvals[2193] = 71;
    colvals[2194] = 72;
    colvals[2195] = 80;
    colvals[2196] = 86;
    colvals[2197] = 88;
    colvals[2198] = 90;
    colvals[2199] = 92;
    colvals[2200] = 94;
    colvals[2201] = 95;
    colvals[2202] = 96;
    colvals[2203] = 97;
    colvals[2204] = 98;
    colvals[2205] = 100;
    colvals[2206] = 101;
    colvals[2207] = 103;
    colvals[2208] = 104;
    colvals[2209] = 105;
    colvals[2210] = 106;
    colvals[2211] = 108;
    colvals[2212] = 110;
    colvals[2213] = 111;
    colvals[2214] = 112;
    colvals[2215] = 113;
    colvals[2216] = 7;
    colvals[2217] = 25;
    colvals[2218] = 39;
    colvals[2219] = 42;
    colvals[2220] = 49;
    colvals[2221] = 51;
    colvals[2222] = 52;
    colvals[2223] = 53;
    colvals[2224] = 57;
    colvals[2225] = 60;
    colvals[2226] = 62;
    colvals[2227] = 63;
    colvals[2228] = 64;
    colvals[2229] = 65;
    colvals[2230] = 68;
    colvals[2231] = 69;
    colvals[2232] = 71;
    colvals[2233] = 72;
    colvals[2234] = 73;
    colvals[2235] = 75;
    colvals[2236] = 76;
    colvals[2237] = 77;
    colvals[2238] = 78;
    colvals[2239] = 79;
    colvals[2240] = 80;
    colvals[2241] = 81;
    colvals[2242] = 82;
    colvals[2243] = 83;
    colvals[2244] = 84;
    colvals[2245] = 86;
    colvals[2246] = 87;
    colvals[2247] = 88;
    colvals[2248] = 89;
    colvals[2249] = 90;
    colvals[2250] = 91;
    colvals[2251] = 92;
    colvals[2252] = 93;
    colvals[2253] = 94;
    colvals[2254] = 95;
    colvals[2255] = 96;
    colvals[2256] = 97;
    colvals[2257] = 98;
    colvals[2258] = 99;
    colvals[2259] = 101;
    colvals[2260] = 102;
    colvals[2261] = 103;
    colvals[2262] = 104;
    colvals[2263] = 105;
    colvals[2264] = 106;
    colvals[2265] = 108;
    colvals[2266] = 109;
    colvals[2267] = 110;
    colvals[2268] = 111;
    colvals[2269] = 113;
    colvals[2270] = 23;
    colvals[2271] = 36;
    colvals[2272] = 37;
    colvals[2273] = 38;
    colvals[2274] = 39;
    colvals[2275] = 49;
    colvals[2276] = 51;
    colvals[2277] = 53;
    colvals[2278] = 57;
    colvals[2279] = 59;
    colvals[2280] = 62;
    colvals[2281] = 63;
    colvals[2282] = 65;
    colvals[2283] = 67;
    colvals[2284] = 68;
    colvals[2285] = 69;
    colvals[2286] = 70;
    colvals[2287] = 71;
    colvals[2288] = 72;
    colvals[2289] = 73;
    colvals[2290] = 74;
    colvals[2291] = 76;
    colvals[2292] = 77;
    colvals[2293] = 78;
    colvals[2294] = 80;
    colvals[2295] = 81;
    colvals[2296] = 82;
    colvals[2297] = 84;
    colvals[2298] = 86;
    colvals[2299] = 87;
    colvals[2300] = 88;
    colvals[2301] = 89;
    colvals[2302] = 90;
    colvals[2303] = 91;
    colvals[2304] = 92;
    colvals[2305] = 93;
    colvals[2306] = 94;
    colvals[2307] = 95;
    colvals[2308] = 96;
    colvals[2309] = 97;
    colvals[2310] = 98;
    colvals[2311] = 101;
    colvals[2312] = 102;
    colvals[2313] = 103;
    colvals[2314] = 104;
    colvals[2315] = 105;
    colvals[2316] = 108;
    colvals[2317] = 109;
    colvals[2318] = 110;
    colvals[2319] = 111;
    colvals[2320] = 112;
    colvals[2321] = 113;
    colvals[2322] = 37;
    colvals[2323] = 39;
    colvals[2324] = 41;
    colvals[2325] = 42;
    colvals[2326] = 44;
    colvals[2327] = 46;
    colvals[2328] = 50;
    colvals[2329] = 51;
    colvals[2330] = 52;
    colvals[2331] = 55;
    colvals[2332] = 57;
    colvals[2333] = 59;
    colvals[2334] = 60;
    colvals[2335] = 61;
    colvals[2336] = 62;
    colvals[2337] = 63;
    colvals[2338] = 64;
    colvals[2339] = 65;
    colvals[2340] = 66;
    colvals[2341] = 67;
    colvals[2342] = 68;
    colvals[2343] = 69;
    colvals[2344] = 70;
    colvals[2345] = 71;
    colvals[2346] = 72;
    colvals[2347] = 73;
    colvals[2348] = 74;
    colvals[2349] = 75;
    colvals[2350] = 76;
    colvals[2351] = 77;
    colvals[2352] = 79;
    colvals[2353] = 80;
    colvals[2354] = 81;
    colvals[2355] = 82;
    colvals[2356] = 85;
    colvals[2357] = 87;
    colvals[2358] = 88;
    colvals[2359] = 89;
    colvals[2360] = 90;
    colvals[2361] = 92;
    colvals[2362] = 93;
    colvals[2363] = 94;
    colvals[2364] = 95;
    colvals[2365] = 96;
    colvals[2366] = 97;
    colvals[2367] = 98;
    colvals[2368] = 99;
    colvals[2369] = 101;
    colvals[2370] = 102;
    colvals[2371] = 103;
    colvals[2372] = 104;
    colvals[2373] = 105;
    colvals[2374] = 106;
    colvals[2375] = 107;
    colvals[2376] = 108;
    colvals[2377] = 109;
    colvals[2378] = 110;
    colvals[2379] = 111;
    colvals[2380] = 112;
    colvals[2381] = 113;
    colvals[2382] = 8;
    colvals[2383] = 37;
    colvals[2384] = 39;
    colvals[2385] = 41;
    colvals[2386] = 42;
    colvals[2387] = 49;
    colvals[2388] = 50;
    colvals[2389] = 51;
    colvals[2390] = 52;
    colvals[2391] = 53;
    colvals[2392] = 55;
    colvals[2393] = 57;
    colvals[2394] = 60;
    colvals[2395] = 62;
    colvals[2396] = 63;
    colvals[2397] = 65;
    colvals[2398] = 67;
    colvals[2399] = 68;
    colvals[2400] = 69;
    colvals[2401] = 70;
    colvals[2402] = 71;
    colvals[2403] = 72;
    colvals[2404] = 73;
    colvals[2405] = 74;
    colvals[2406] = 75;
    colvals[2407] = 76;
    colvals[2408] = 77;
    colvals[2409] = 78;
    colvals[2410] = 79;
    colvals[2411] = 80;
    colvals[2412] = 81;
    colvals[2413] = 82;
    colvals[2414] = 86;
    colvals[2415] = 87;
    colvals[2416] = 88;
    colvals[2417] = 89;
    colvals[2418] = 90;
    colvals[2419] = 91;
    colvals[2420] = 92;
    colvals[2421] = 93;
    colvals[2422] = 94;
    colvals[2423] = 95;
    colvals[2424] = 96;
    colvals[2425] = 97;
    colvals[2426] = 98;
    colvals[2427] = 99;
    colvals[2428] = 101;
    colvals[2429] = 102;
    colvals[2430] = 103;
    colvals[2431] = 104;
    colvals[2432] = 105;
    colvals[2433] = 106;
    colvals[2434] = 108;
    colvals[2435] = 109;
    colvals[2436] = 110;
    colvals[2437] = 111;
    colvals[2438] = 112;
    colvals[2439] = 113;
    colvals[2440] = 22;
    colvals[2441] = 27;
    colvals[2442] = 30;
    colvals[2443] = 32;
    colvals[2444] = 33;
    colvals[2445] = 36;
    colvals[2446] = 38;
    colvals[2447] = 42;
    colvals[2448] = 43;
    colvals[2449] = 45;
    colvals[2450] = 47;
    colvals[2451] = 48;
    colvals[2452] = 49;
    colvals[2453] = 50;
    colvals[2454] = 53;
    colvals[2455] = 57;
    colvals[2456] = 59;
    colvals[2457] = 60;
    colvals[2458] = 62;
    colvals[2459] = 63;
    colvals[2460] = 64;
    colvals[2461] = 65;
    colvals[2462] = 68;
    colvals[2463] = 69;
    colvals[2464] = 70;
    colvals[2465] = 72;
    colvals[2466] = 73;
    colvals[2467] = 74;
    colvals[2468] = 75;
    colvals[2469] = 77;
    colvals[2470] = 78;
    colvals[2471] = 80;
    colvals[2472] = 81;
    colvals[2473] = 82;
    colvals[2474] = 85;
    colvals[2475] = 86;
    colvals[2476] = 87;
    colvals[2477] = 88;
    colvals[2478] = 89;
    colvals[2479] = 90;
    colvals[2480] = 92;
    colvals[2481] = 93;
    colvals[2482] = 94;
    colvals[2483] = 95;
    colvals[2484] = 96;
    colvals[2485] = 97;
    colvals[2486] = 98;
    colvals[2487] = 99;
    colvals[2488] = 101;
    colvals[2489] = 102;
    colvals[2490] = 103;
    colvals[2491] = 104;
    colvals[2492] = 105;
    colvals[2493] = 107;
    colvals[2494] = 108;
    colvals[2495] = 109;
    colvals[2496] = 110;
    colvals[2497] = 111;
    colvals[2498] = 112;
    colvals[2499] = 113;
    colvals[2500] = 26;
    colvals[2501] = 27;
    colvals[2502] = 32;
    colvals[2503] = 33;
    colvals[2504] = 38;
    colvals[2505] = 39;
    colvals[2506] = 43;
    colvals[2507] = 46;
    colvals[2508] = 47;
    colvals[2509] = 48;
    colvals[2510] = 51;
    colvals[2511] = 52;
    colvals[2512] = 53;
    colvals[2513] = 56;
    colvals[2514] = 57;
    colvals[2515] = 62;
    colvals[2516] = 63;
    colvals[2517] = 67;
    colvals[2518] = 68;
    colvals[2519] = 69;
    colvals[2520] = 70;
    colvals[2521] = 71;
    colvals[2522] = 72;
    colvals[2523] = 74;
    colvals[2524] = 75;
    colvals[2525] = 77;
    colvals[2526] = 80;
    colvals[2527] = 81;
    colvals[2528] = 88;
    colvals[2529] = 89;
    colvals[2530] = 90;
    colvals[2531] = 92;
    colvals[2532] = 95;
    colvals[2533] = 96;
    colvals[2534] = 97;
    colvals[2535] = 98;
    colvals[2536] = 99;
    colvals[2537] = 100;
    colvals[2538] = 101;
    colvals[2539] = 103;
    colvals[2540] = 104;
    colvals[2541] = 106;
    colvals[2542] = 108;
    colvals[2543] = 109;
    colvals[2544] = 110;
    colvals[2545] = 112;
    colvals[2546] = 113;
    colvals[2547] = 28;
    colvals[2548] = 35;
    colvals[2549] = 43;
    colvals[2550] = 46;
    colvals[2551] = 47;
    colvals[2552] = 49;
    colvals[2553] = 50;
    colvals[2554] = 52;
    colvals[2555] = 53;
    colvals[2556] = 55;
    colvals[2557] = 56;
    colvals[2558] = 57;
    colvals[2559] = 59;
    colvals[2560] = 60;
    colvals[2561] = 62;
    colvals[2562] = 63;
    colvals[2563] = 64;
    colvals[2564] = 65;
    colvals[2565] = 67;
    colvals[2566] = 68;
    colvals[2567] = 69;
    colvals[2568] = 70;
    colvals[2569] = 72;
    colvals[2570] = 73;
    colvals[2571] = 74;
    colvals[2572] = 75;
    colvals[2573] = 76;
    colvals[2574] = 77;
    colvals[2575] = 78;
    colvals[2576] = 79;
    colvals[2577] = 80;
    colvals[2578] = 81;
    colvals[2579] = 82;
    colvals[2580] = 83;
    colvals[2581] = 85;
    colvals[2582] = 87;
    colvals[2583] = 88;
    colvals[2584] = 89;
    colvals[2585] = 90;
    colvals[2586] = 91;
    colvals[2587] = 92;
    colvals[2588] = 93;
    colvals[2589] = 95;
    colvals[2590] = 96;
    colvals[2591] = 97;
    colvals[2592] = 98;
    colvals[2593] = 99;
    colvals[2594] = 103;
    colvals[2595] = 104;
    colvals[2596] = 105;
    colvals[2597] = 106;
    colvals[2598] = 107;
    colvals[2599] = 108;
    colvals[2600] = 109;
    colvals[2601] = 110;
    colvals[2602] = 111;
    colvals[2603] = 112;
    colvals[2604] = 20;
    colvals[2605] = 34;
    colvals[2606] = 35;
    colvals[2607] = 37;
    colvals[2608] = 43;
    colvals[2609] = 45;
    colvals[2610] = 47;
    colvals[2611] = 50;
    colvals[2612] = 51;
    colvals[2613] = 52;
    colvals[2614] = 53;
    colvals[2615] = 55;
    colvals[2616] = 56;
    colvals[2617] = 57;
    colvals[2618] = 59;
    colvals[2619] = 60;
    colvals[2620] = 61;
    colvals[2621] = 62;
    colvals[2622] = 63;
    colvals[2623] = 64;
    colvals[2624] = 65;
    colvals[2625] = 66;
    colvals[2626] = 67;
    colvals[2627] = 68;
    colvals[2628] = 69;
    colvals[2629] = 70;
    colvals[2630] = 71;
    colvals[2631] = 73;
    colvals[2632] = 74;
    colvals[2633] = 75;
    colvals[2634] = 76;
    colvals[2635] = 77;
    colvals[2636] = 80;
    colvals[2637] = 81;
    colvals[2638] = 82;
    colvals[2639] = 85;
    colvals[2640] = 87;
    colvals[2641] = 88;
    colvals[2642] = 89;
    colvals[2643] = 90;
    colvals[2644] = 91;
    colvals[2645] = 92;
    colvals[2646] = 93;
    colvals[2647] = 95;
    colvals[2648] = 96;
    colvals[2649] = 97;
    colvals[2650] = 98;
    colvals[2651] = 99;
    colvals[2652] = 101;
    colvals[2653] = 103;
    colvals[2654] = 104;
    colvals[2655] = 106;
    colvals[2656] = 107;
    colvals[2657] = 108;
    colvals[2658] = 109;
    colvals[2659] = 110;
    colvals[2660] = 112;
    colvals[2661] = 113;
    colvals[2662] = 23;
    colvals[2663] = 32;
    colvals[2664] = 33;
    colvals[2665] = 36;
    colvals[2666] = 37;
    colvals[2667] = 38;
    colvals[2668] = 39;
    colvals[2669] = 40;
    colvals[2670] = 41;
    colvals[2671] = 42;
    colvals[2672] = 43;
    colvals[2673] = 45;
    colvals[2674] = 46;
    colvals[2675] = 47;
    colvals[2676] = 48;
    colvals[2677] = 49;
    colvals[2678] = 50;
    colvals[2679] = 51;
    colvals[2680] = 55;
    colvals[2681] = 56;
    colvals[2682] = 57;
    colvals[2683] = 59;
    colvals[2684] = 60;
    colvals[2685] = 61;
    colvals[2686] = 62;
    colvals[2687] = 63;
    colvals[2688] = 65;
    colvals[2689] = 67;
    colvals[2690] = 68;
    colvals[2691] = 69;
    colvals[2692] = 70;
    colvals[2693] = 71;
    colvals[2694] = 72;
    colvals[2695] = 73;
    colvals[2696] = 74;
    colvals[2697] = 76;
    colvals[2698] = 77;
    colvals[2699] = 78;
    colvals[2700] = 79;
    colvals[2701] = 80;
    colvals[2702] = 81;
    colvals[2703] = 82;
    colvals[2704] = 83;
    colvals[2705] = 84;
    colvals[2706] = 85;
    colvals[2707] = 86;
    colvals[2708] = 87;
    colvals[2709] = 88;
    colvals[2710] = 89;
    colvals[2711] = 90;
    colvals[2712] = 91;
    colvals[2713] = 92;
    colvals[2714] = 93;
    colvals[2715] = 94;
    colvals[2716] = 95;
    colvals[2717] = 96;
    colvals[2718] = 97;
    colvals[2719] = 98;
    colvals[2720] = 99;
    colvals[2721] = 101;
    colvals[2722] = 102;
    colvals[2723] = 103;
    colvals[2724] = 104;
    colvals[2725] = 105;
    colvals[2726] = 106;
    colvals[2727] = 108;
    colvals[2728] = 109;
    colvals[2729] = 110;
    colvals[2730] = 111;
    colvals[2731] = 112;
    colvals[2732] = 113;
    colvals[2733] = 22;
    colvals[2734] = 25;
    colvals[2735] = 28;
    colvals[2736] = 29;
    colvals[2737] = 30;
    colvals[2738] = 34;
    colvals[2739] = 35;
    colvals[2740] = 36;
    colvals[2741] = 40;
    colvals[2742] = 41;
    colvals[2743] = 43;
    colvals[2744] = 44;
    colvals[2745] = 45;
    colvals[2746] = 48;
    colvals[2747] = 49;
    colvals[2748] = 50;
    colvals[2749] = 52;
    colvals[2750] = 53;
    colvals[2751] = 54;
    colvals[2752] = 55;
    colvals[2753] = 56;
    colvals[2754] = 57;
    colvals[2755] = 58;
    colvals[2756] = 59;
    colvals[2757] = 60;
    colvals[2758] = 61;
    colvals[2759] = 62;
    colvals[2760] = 64;
    colvals[2761] = 65;
    colvals[2762] = 66;
    colvals[2763] = 67;
    colvals[2764] = 68;
    colvals[2765] = 69;
    colvals[2766] = 70;
    colvals[2767] = 71;
    colvals[2768] = 73;
    colvals[2769] = 74;
    colvals[2770] = 75;
    colvals[2771] = 76;
    colvals[2772] = 77;
    colvals[2773] = 78;
    colvals[2774] = 79;
    colvals[2775] = 80;
    colvals[2776] = 81;
    colvals[2777] = 82;
    colvals[2778] = 83;
    colvals[2779] = 84;
    colvals[2780] = 85;
    colvals[2781] = 87;
    colvals[2782] = 89;
    colvals[2783] = 90;
    colvals[2784] = 91;
    colvals[2785] = 92;
    colvals[2786] = 93;
    colvals[2787] = 95;
    colvals[2788] = 96;
    colvals[2789] = 97;
    colvals[2790] = 98;
    colvals[2791] = 99;
    colvals[2792] = 100;
    colvals[2793] = 101;
    colvals[2794] = 102;
    colvals[2795] = 103;
    colvals[2796] = 104;
    colvals[2797] = 105;
    colvals[2798] = 106;
    colvals[2799] = 107;
    colvals[2800] = 108;
    colvals[2801] = 109;
    colvals[2802] = 110;
    colvals[2803] = 111;
    colvals[2804] = 112;
    colvals[2805] = 113;
    colvals[2806] = 4;
    colvals[2807] = 27;
    colvals[2808] = 28;
    colvals[2809] = 32;
    colvals[2810] = 33;
    colvals[2811] = 35;
    colvals[2812] = 37;
    colvals[2813] = 38;
    colvals[2814] = 39;
    colvals[2815] = 42;
    colvals[2816] = 43;
    colvals[2817] = 46;
    colvals[2818] = 47;
    colvals[2819] = 49;
    colvals[2820] = 50;
    colvals[2821] = 51;
    colvals[2822] = 55;
    colvals[2823] = 56;
    colvals[2824] = 57;
    colvals[2825] = 59;
    colvals[2826] = 60;
    colvals[2827] = 62;
    colvals[2828] = 63;
    colvals[2829] = 64;
    colvals[2830] = 65;
    colvals[2831] = 66;
    colvals[2832] = 67;
    colvals[2833] = 68;
    colvals[2834] = 69;
    colvals[2835] = 70;
    colvals[2836] = 71;
    colvals[2837] = 72;
    colvals[2838] = 73;
    colvals[2839] = 74;
    colvals[2840] = 75;
    colvals[2841] = 77;
    colvals[2842] = 78;
    colvals[2843] = 79;
    colvals[2844] = 80;
    colvals[2845] = 81;
    colvals[2846] = 82;
    colvals[2847] = 87;
    colvals[2848] = 88;
    colvals[2849] = 89;
    colvals[2850] = 90;
    colvals[2851] = 91;
    colvals[2852] = 92;
    colvals[2853] = 93;
    colvals[2854] = 94;
    colvals[2855] = 95;
    colvals[2856] = 96;
    colvals[2857] = 97;
    colvals[2858] = 98;
    colvals[2859] = 99;
    colvals[2860] = 101;
    colvals[2861] = 102;
    colvals[2862] = 103;
    colvals[2863] = 104;
    colvals[2864] = 105;
    colvals[2865] = 106;
    colvals[2866] = 107;
    colvals[2867] = 108;
    colvals[2868] = 109;
    colvals[2869] = 110;
    colvals[2870] = 111;
    colvals[2871] = 113;
    colvals[2872] = 23;
    colvals[2873] = 25;
    colvals[2874] = 26;
    colvals[2875] = 28;
    colvals[2876] = 29;
    colvals[2877] = 34;
    colvals[2878] = 35;
    colvals[2879] = 37;
    colvals[2880] = 39;
    colvals[2881] = 40;
    colvals[2882] = 41;
    colvals[2883] = 43;
    colvals[2884] = 45;
    colvals[2885] = 46;
    colvals[2886] = 47;
    colvals[2887] = 48;
    colvals[2888] = 49;
    colvals[2889] = 51;
    colvals[2890] = 52;
    colvals[2891] = 53;
    colvals[2892] = 55;
    colvals[2893] = 56;
    colvals[2894] = 57;
    colvals[2895] = 60;
    colvals[2896] = 61;
    colvals[2897] = 62;
    colvals[2898] = 63;
    colvals[2899] = 65;
    colvals[2900] = 66;
    colvals[2901] = 67;
    colvals[2902] = 68;
    colvals[2903] = 69;
    colvals[2904] = 70;
    colvals[2905] = 71;
    colvals[2906] = 72;
    colvals[2907] = 73;
    colvals[2908] = 74;
    colvals[2909] = 75;
    colvals[2910] = 76;
    colvals[2911] = 77;
    colvals[2912] = 79;
    colvals[2913] = 80;
    colvals[2914] = 81;
    colvals[2915] = 82;
    colvals[2916] = 83;
    colvals[2917] = 85;
    colvals[2918] = 86;
    colvals[2919] = 87;
    colvals[2920] = 88;
    colvals[2921] = 89;
    colvals[2922] = 90;
    colvals[2923] = 91;
    colvals[2924] = 92;
    colvals[2925] = 93;
    colvals[2926] = 94;
    colvals[2927] = 95;
    colvals[2928] = 96;
    colvals[2929] = 97;
    colvals[2930] = 98;
    colvals[2931] = 99;
    colvals[2932] = 101;
    colvals[2933] = 102;
    colvals[2934] = 103;
    colvals[2935] = 104;
    colvals[2936] = 105;
    colvals[2937] = 106;
    colvals[2938] = 108;
    colvals[2939] = 109;
    colvals[2940] = 110;
    colvals[2941] = 111;
    colvals[2942] = 112;
    colvals[2943] = 113;
    colvals[2944] = 23;
    colvals[2945] = 25;
    colvals[2946] = 26;
    colvals[2947] = 28;
    colvals[2948] = 29;
    colvals[2949] = 32;
    colvals[2950] = 33;
    colvals[2951] = 34;
    colvals[2952] = 35;
    colvals[2953] = 37;
    colvals[2954] = 38;
    colvals[2955] = 39;
    colvals[2956] = 40;
    colvals[2957] = 41;
    colvals[2958] = 42;
    colvals[2959] = 43;
    colvals[2960] = 44;
    colvals[2961] = 45;
    colvals[2962] = 46;
    colvals[2963] = 47;
    colvals[2964] = 48;
    colvals[2965] = 49;
    colvals[2966] = 50;
    colvals[2967] = 51;
    colvals[2968] = 52;
    colvals[2969] = 53;
    colvals[2970] = 55;
    colvals[2971] = 56;
    colvals[2972] = 57;
    colvals[2973] = 58;
    colvals[2974] = 59;
    colvals[2975] = 60;
    colvals[2976] = 61;
    colvals[2977] = 62;
    colvals[2978] = 63;
    colvals[2979] = 64;
    colvals[2980] = 65;
    colvals[2981] = 66;
    colvals[2982] = 67;
    colvals[2983] = 68;
    colvals[2984] = 69;
    colvals[2985] = 70;
    colvals[2986] = 71;
    colvals[2987] = 72;
    colvals[2988] = 73;
    colvals[2989] = 74;
    colvals[2990] = 75;
    colvals[2991] = 76;
    colvals[2992] = 77;
    colvals[2993] = 78;
    colvals[2994] = 79;
    colvals[2995] = 80;
    colvals[2996] = 81;
    colvals[2997] = 82;
    colvals[2998] = 83;
    colvals[2999] = 85;
    colvals[3000] = 86;
    colvals[3001] = 87;
    colvals[3002] = 88;
    colvals[3003] = 89;
    colvals[3004] = 90;
    colvals[3005] = 91;
    colvals[3006] = 92;
    colvals[3007] = 93;
    colvals[3008] = 94;
    colvals[3009] = 95;
    colvals[3010] = 96;
    colvals[3011] = 97;
    colvals[3012] = 98;
    colvals[3013] = 99;
    colvals[3014] = 100;
    colvals[3015] = 101;
    colvals[3016] = 102;
    colvals[3017] = 103;
    colvals[3018] = 104;
    colvals[3019] = 105;
    colvals[3020] = 106;
    colvals[3021] = 107;
    colvals[3022] = 108;
    colvals[3023] = 109;
    colvals[3024] = 110;
    colvals[3025] = 111;
    colvals[3026] = 112;
    colvals[3027] = 113;
    
    // value of each non-zero element
    data[0] = 0.0 - k[1339] - k[1340] - k[1341] - k[1342];
    data[1] = 0.0 + k[1298];
    data[2] = 0.0 - k[1327] - k[1328] - k[1329] - k[1330];
    data[3] = 0.0 + k[1305];
    data[4] = 0.0 - k[1387] - k[1388] - k[1389] - k[1390];
    data[5] = 0.0 + k[1293];
    data[6] = 0.0 - k[1379] - k[1380] - k[1381] - k[1382];
    data[7] = 0.0 - k[1331] - k[1332] - k[1333] - k[1334];
    data[8] = 0.0 + k[1275];
    data[9] = 0.0 + k[1251];
    data[10] = 0.0 - k[1375] - k[1376] - k[1377] - k[1378];
    data[11] = 0.0 + k[1224];
    data[12] = 0.0 + k[1225];
    data[13] = 0.0 - k[1319] - k[1320] - k[1321] - k[1322];
    data[14] = 0.0 + k[1300];
    data[15] = 0.0 + k[1299];
    data[16] = 0.0 - k[1343] - k[1344] - k[1345] - k[1346];
    data[17] = 0.0 + k[1277];
    data[18] = 0.0 + k[1255];
    data[19] = 0.0 - k[1355] - k[1356] - k[1357] - k[1358];
    data[20] = 0.0 + k[1270];
    data[21] = 0.0 + k[1292];
    data[22] = 0.0 - k[1367] - k[1368] - k[1369] - k[1370];
    data[23] = 0.0 + k[1303];
    data[24] = 0.0 + k[1297];
    data[25] = 0.0 - k[1371] - k[1372] - k[1373] - k[1374];
    data[26] = 0.0 + k[1241];
    data[27] = 0.0 + k[1239];
    data[28] = 0.0 - k[1395] - k[1396] - k[1397] - k[1398];
    data[29] = 0.0 + k[1242];
    data[30] = 0.0 + k[1240];
    data[31] = 0.0 - k[1399] - k[1400] - k[1401] - k[1402];
    data[32] = 0.0 + k[1244];
    data[33] = 0.0 + k[1243];
    data[34] = 0.0 - k[1359] - k[1360] - k[1361] - k[1362];
    data[35] = 0.0 + k[1223];
    data[36] = 0.0 + k[1249];
    data[37] = 0.0 + k[1304];
    data[38] = 0.0 - k[1383] - k[1384] - k[1385] - k[1386];
    data[39] = 0.0 + k[1287];
    data[40] = 0.0 + k[1258];
    data[41] = 0.0 - k[1391] - k[1392] - k[1393] - k[1394];
    data[42] = 0.0 + k[1247];
    data[43] = 0.0 + k[1246];
    data[44] = 0.0 + k[1245];
    data[45] = 0.0 + k[1229];
    data[46] = 0.0 - k[1351] - k[1352] - k[1353] - k[1354];
    data[47] = 0.0 + k[1296];
    data[48] = 0.0 + k[1294];
    data[49] = 0.0 + k[1295];
    data[50] = 0.0 - k[1335] - k[1336] - k[1337] - k[1338];
    data[51] = 0.0 + k[1302];
    data[52] = 0.0 + k[1271];
    data[53] = 0.0 + k[1262];
    data[54] = 0.0 - k[1347] - k[1348] - k[1349] - k[1350];
    data[55] = 0.0 + k[1226];
    data[56] = 0.0 + k[1284];
    data[57] = 0.0 + k[1252];
    data[58] = 0.0 + k[1261];
    data[59] = 0.0 + k[1281];
    data[60] = 0.0 + k[1227];
    data[61] = 0.0 - k[1323] - k[1324] - k[1325] - k[1326];
    data[62] = 0.0 + k[1276];
    data[63] = 0.0 + k[1301];
    data[64] = 0.0 + k[1282];
    data[65] = 0.0 + k[1266];
    data[66] = 0.0 + k[1263];
    data[67] = 0.0 - k[1315] - k[1316] - k[1317] - k[1318];
    data[68] = 0.0 + k[1269];
    data[69] = 0.0 + k[1280];
    data[70] = 0.0 + k[1286];
    data[71] = 0.0 + k[1274];
    data[72] = 0.0 + k[1254];
    data[73] = 0.0 + k[1257];
    data[74] = 0.0 + k[1291];
    data[75] = 0.0 - k[1311] - k[1312] - k[1313] - k[1314];
    data[76] = 0.0 + k[1267];
    data[77] = 0.0 + k[1268];
    data[78] = 0.0 + k[1279];
    data[79] = 0.0 + k[1273];
    data[80] = 0.0 + k[1289];
    data[81] = 0.0 + k[1283];
    data[82] = 0.0 + k[1265];
    data[83] = 0.0 + k[1290];
    data[84] = 0.0 - k[351]*y[IDX_eM] - k[1244];
    data[85] = 0.0 + k[23]*y[IDX_CII] + k[93]*y[IDX_HII];
    data[86] = 0.0 + k[23]*y[IDX_SiC3I];
    data[87] = 0.0 + k[93]*y[IDX_SiC3I];
    data[88] = 0.0 - k[351]*y[IDX_SiC3II];
    data[89] = 0.0 + k[1339] + k[1340] + k[1341] + k[1342];
    data[90] = 0.0 - k[254] - k[973]*y[IDX_HI] - k[1013]*y[IDX_NI] - k[1060]*y[IDX_OI]
        - k[1135] - k[1298];
    data[91] = 0.0 + k[1008]*y[IDX_NI];
    data[92] = 0.0 + k[1008]*y[IDX_CH3I] - k[1013]*y[IDX_H2CNI];
    data[93] = 0.0 - k[1060]*y[IDX_H2CNI];
    data[94] = 0.0 - k[973]*y[IDX_H2CNI];
    data[95] = 0.0 - k[1307] - k[1308] - k[1309] - k[1310];
    data[96] = 0.0 + k[1288];
    data[97] = 0.0 + k[1260];
    data[98] = 0.0 + k[1259];
    data[99] = 0.0 + k[1285];
    data[100] = 0.0 + k[1278];
    data[101] = 0.0 + k[1264];
    data[102] = 0.0 + k[1272];
    data[103] = 0.0 + k[1256];
    data[104] = 0.0 + k[1253];
    data[105] = 0.0 + k[1250];
    data[106] = 0.0 - k[310]*y[IDX_eM] - k[311]*y[IDX_eM] - k[1296];
    data[107] = 0.0 + k[590]*y[IDX_H3II];
    data[108] = 0.0 + k[777]*y[IDX_O2I];
    data[109] = 0.0 + k[590]*y[IDX_HNOI];
    data[110] = 0.0 + k[777]*y[IDX_NH2II];
    data[111] = 0.0 - k[310]*y[IDX_H2NOII] - k[311]*y[IDX_H2NOII];
    data[112] = 0.0 + k[1391] + k[1392] + k[1393] + k[1394];
    data[113] = 0.0 - k[257] - k[499]*y[IDX_HII] - k[675]*y[IDX_HeII] - k[1143] -
        k[1144] - k[1247];
    data[114] = 0.0 + k[1087]*y[IDX_OI];
    data[115] = 0.0 - k[675]*y[IDX_H2SiOI];
    data[116] = 0.0 - k[499]*y[IDX_H2SiOI];
    data[117] = 0.0 + k[1087]*y[IDX_SiH3I];
    data[118] = 0.0 + k[1375] + k[1376] + k[1377] + k[1378];
    data[119] = 0.0 - k[263] - k[502]*y[IDX_HII] - k[1004]*y[IDX_CI] - k[1152] - k[1224];
    data[120] = 0.0 + k[888]*y[IDX_NOI];
    data[121] = 0.0 + k[888]*y[IDX_CH2I];
    data[122] = 0.0 - k[1004]*y[IDX_HNCOI];
    data[123] = 0.0 - k[502]*y[IDX_HNCOI];
    data[124] = 0.0 - k[5]*y[IDX_H2I] - k[335]*y[IDX_eM] - k[1226];
    data[125] = 0.0 + k[533]*y[IDX_H2I];
    data[126] = 0.0 + k[371]*y[IDX_H2OI];
    data[127] = 0.0 + k[584]*y[IDX_COI];
    data[128] = 0.0 + k[371]*y[IDX_CII];
    data[129] = 0.0 - k[335]*y[IDX_HOCII];
    data[130] = 0.0 + k[584]*y[IDX_H3II];
    data[131] = 0.0 - k[5]*y[IDX_HOCII] + k[533]*y[IDX_COII];
    data[132] = 0.0 - k[336]*y[IDX_eM] - k[537]*y[IDX_H2I] - k[618]*y[IDX_HI];
    data[133] = 0.0 + k[520]*y[IDX_HeI];
    data[134] = 0.0 + k[681]*y[IDX_HeII];
    data[135] = 0.0 + k[681]*y[IDX_HCOI];
    data[136] = 0.0 + k[520]*y[IDX_H2II] + k[1198]*y[IDX_HII];
    data[137] = 0.0 + k[1198]*y[IDX_HeI];
    data[138] = 0.0 - k[336]*y[IDX_HeHII];
    data[139] = 0.0 - k[537]*y[IDX_HeHII];
    data[140] = 0.0 - k[618]*y[IDX_HeHII];
    data[141] = 0.0 - k[350]*y[IDX_eM] - k[1242];
    data[142] = 0.0 + k[22]*y[IDX_CII] + k[92]*y[IDX_HII];
    data[143] = 0.0 + k[700]*y[IDX_HeII];
    data[144] = 0.0 + k[22]*y[IDX_SiC2I];
    data[145] = 0.0 + k[700]*y[IDX_SiC3I];
    data[146] = 0.0 + k[92]*y[IDX_SiC2I];
    data[147] = 0.0 - k[350]*y[IDX_SiC2II];
    data[148] = 0.0 - k[1363] - k[1364] - k[1365] - k[1366];
    data[149] = 0.0 + k[1248];
    data[150] = 0.0 + k[1238];
    data[151] = 0.0 + k[1236];
    data[152] = 0.0 + k[1234];
    data[153] = 0.0 + k[1233];
    data[154] = 0.0 + k[1232];
    data[155] = 0.0 + k[1237];
    data[156] = 0.0 + k[1230];
    data[157] = 0.0 + k[1235];
    data[158] = 0.0 + k[1231];
    data[159] = 0.0 + k[1228];
    data[160] = 0.0 + k[1395] + k[1396] + k[1397] + k[1398];
    data[161] = 0.0 + k[351]*y[IDX_eM];
    data[162] = 0.0 - k[22]*y[IDX_CII] - k[92]*y[IDX_HII] - k[286] - k[1081]*y[IDX_OI] -
        k[1240];
    data[163] = 0.0 + k[287] + k[1082]*y[IDX_OI] + k[1177];
    data[164] = 0.0 - k[22]*y[IDX_SiC2I];
    data[165] = 0.0 - k[92]*y[IDX_SiC2I];
    data[166] = 0.0 - k[1081]*y[IDX_SiC2I] + k[1082]*y[IDX_SiC3I];
    data[167] = 0.0 + k[351]*y[IDX_SiC3II];
    data[168] = 0.0 + k[1399] + k[1400] + k[1401] + k[1402];
    data[169] = 0.0 - k[23]*y[IDX_CII] - k[93]*y[IDX_HII] - k[287] - k[700]*y[IDX_HeII]
        - k[1082]*y[IDX_OI] - k[1177] - k[1243];
    data[170] = 0.0 - k[23]*y[IDX_SiC3I];
    data[171] = 0.0 - k[700]*y[IDX_SiC3I];
    data[172] = 0.0 - k[93]*y[IDX_SiC3I];
    data[173] = 0.0 - k[1082]*y[IDX_SiC3I];
    data[174] = 0.0 - k[360]*y[IDX_eM] - k[361]*y[IDX_eM] - k[575]*y[IDX_H2OI] - k[1248];
    data[175] = 0.0 + k[546]*y[IDX_H2I];
    data[176] = 0.0 + k[1204]*y[IDX_H2I];
    data[177] = 0.0 + k[604]*y[IDX_H3II] + k[638]*y[IDX_HCOII];
    data[178] = 0.0 + k[604]*y[IDX_SiH4I];
    data[179] = 0.0 + k[638]*y[IDX_SiH4I];
    data[180] = 0.0 - k[575]*y[IDX_SiH5II];
    data[181] = 0.0 - k[360]*y[IDX_SiH5II] - k[361]*y[IDX_SiH5II];
    data[182] = 0.0 + k[546]*y[IDX_SiH4II] + k[1204]*y[IDX_SiH3II];
    data[183] = 0.0 - k[358]*y[IDX_eM] - k[359]*y[IDX_eM] - k[489]*y[IDX_COI] -
        k[546]*y[IDX_H2I] - k[574]*y[IDX_H2OI] - k[1238];
    data[184] = 0.0 + k[97]*y[IDX_HII];
    data[185] = 0.0 + k[603]*y[IDX_H3II];
    data[186] = 0.0 + k[603]*y[IDX_SiH3I];
    data[187] = 0.0 + k[97]*y[IDX_SiH4I];
    data[188] = 0.0 - k[574]*y[IDX_SiH4II];
    data[189] = 0.0 - k[358]*y[IDX_SiH4II] - k[359]*y[IDX_SiH4II];
    data[190] = 0.0 - k[489]*y[IDX_SiH4II];
    data[191] = 0.0 - k[546]*y[IDX_SiH4II];
    data[192] = 0.0 - k[349]*y[IDX_eM] - k[744]*y[IDX_NI] - k[831]*y[IDX_OI] - k[1241];
    data[193] = 0.0 + k[24]*y[IDX_CII] + k[94]*y[IDX_HII];
    data[194] = 0.0 + k[380]*y[IDX_CII];
    data[195] = 0.0 + k[394]*y[IDX_CI];
    data[196] = 0.0 + k[381]*y[IDX_CII];
    data[197] = 0.0 + k[476]*y[IDX_CHI];
    data[198] = 0.0 + k[24]*y[IDX_SiCI] + k[380]*y[IDX_SiH2I] + k[381]*y[IDX_SiHI];
    data[199] = 0.0 + k[476]*y[IDX_SiII];
    data[200] = 0.0 - k[744]*y[IDX_SiCII];
    data[201] = 0.0 + k[394]*y[IDX_SiHII];
    data[202] = 0.0 + k[94]*y[IDX_SiCI];
    data[203] = 0.0 - k[831]*y[IDX_SiCII];
    data[204] = 0.0 - k[349]*y[IDX_SiCII];
    data[205] = 0.0 + k[1367] + k[1368] + k[1369] + k[1370];
    data[206] = 0.0 - k[281] - k[914]*y[IDX_CH3I] - k[937]*y[IDX_CHI] -
        k[938]*y[IDX_CHI] - k[954]*y[IDX_COI] - k[990]*y[IDX_HI] -
        k[991]*y[IDX_HI] - k[992]*y[IDX_HI] - k[1003]*y[IDX_HCOI] -
        k[1024]*y[IDX_NI] - k[1077]*y[IDX_OI] - k[1100]*y[IDX_OHI] - k[1170] -
        k[1171] - k[1303];
    data[207] = 0.0 + k[921]*y[IDX_O2I];
    data[208] = 0.0 + k[913]*y[IDX_O2I] - k[914]*y[IDX_O2HI];
    data[209] = 0.0 + k[549]*y[IDX_O2I];
    data[210] = 0.0 + k[1002]*y[IDX_O2I] - k[1003]*y[IDX_O2HI];
    data[211] = 0.0 - k[937]*y[IDX_O2HI] - k[938]*y[IDX_O2HI];
    data[212] = 0.0 - k[1024]*y[IDX_O2HI];
    data[213] = 0.0 - k[1100]*y[IDX_O2HI];
    data[214] = 0.0 + k[549]*y[IDX_H2COII] + k[913]*y[IDX_CH3I] + k[921]*y[IDX_CH4I] +
        k[963]*y[IDX_H2I] + k[1002]*y[IDX_HCOI];
    data[215] = 0.0 - k[1077]*y[IDX_O2HI];
    data[216] = 0.0 - k[954]*y[IDX_O2HI];
    data[217] = 0.0 + k[963]*y[IDX_O2I];
    data[218] = 0.0 - k[990]*y[IDX_O2HI] - k[991]*y[IDX_O2HI] - k[992]*y[IDX_O2HI];
    data[219] = 0.0 + k[1371] + k[1372] + k[1373] + k[1374];
    data[220] = 0.0 + k[350]*y[IDX_eM];
    data[221] = 0.0 + k[286] + k[1081]*y[IDX_OI];
    data[222] = 0.0 - k[24]*y[IDX_CII] - k[94]*y[IDX_HII] - k[288] - k[701]*y[IDX_HeII]
        - k[702]*y[IDX_HeII] - k[1027]*y[IDX_NI] - k[1083]*y[IDX_OI] -
        k[1084]*y[IDX_OI] - k[1178] - k[1239];
    data[223] = 0.0 + k[877]*y[IDX_CI];
    data[224] = 0.0 - k[24]*y[IDX_SiCI];
    data[225] = 0.0 - k[701]*y[IDX_SiCI] - k[702]*y[IDX_SiCI];
    data[226] = 0.0 - k[1027]*y[IDX_SiCI];
    data[227] = 0.0 + k[877]*y[IDX_SiHI];
    data[228] = 0.0 - k[94]*y[IDX_SiCI];
    data[229] = 0.0 + k[1081]*y[IDX_SiC2I] - k[1083]*y[IDX_SiCI] - k[1084]*y[IDX_SiCI];
    data[230] = 0.0 + k[350]*y[IDX_SiC2II];
    data[231] = 0.0 + k[1387] + k[1388] + k[1389] + k[1390];
    data[232] = 0.0 - k[276] - k[504]*y[IDX_HII] - k[595]*y[IDX_H3II] -
        k[818]*y[IDX_OII] - k[885]*y[IDX_CH2I] - k[909]*y[IDX_CH3I] -
        k[945]*y[IDX_CNI] - k[952]*y[IDX_COI] - k[986]*y[IDX_HI] -
        k[1019]*y[IDX_NI] - k[1020]*y[IDX_NI] - k[1021]*y[IDX_NI] -
        k[1041]*y[IDX_NHI] - k[1075]*y[IDX_OI] - k[1164] - k[1293];
    data[233] = 0.0 + k[1055]*y[IDX_O2I];
    data[234] = 0.0 + k[1068]*y[IDX_OI];
    data[235] = 0.0 - k[909]*y[IDX_NO2I];
    data[236] = 0.0 - k[818]*y[IDX_NO2I];
    data[237] = 0.0 - k[885]*y[IDX_NO2I];
    data[238] = 0.0 - k[1041]*y[IDX_NO2I];
    data[239] = 0.0 - k[945]*y[IDX_NO2I];
    data[240] = 0.0 - k[595]*y[IDX_NO2I];
    data[241] = 0.0 + k[1052]*y[IDX_O2I] + k[1099]*y[IDX_OHI];
    data[242] = 0.0 - k[1019]*y[IDX_NO2I] - k[1020]*y[IDX_NO2I] - k[1021]*y[IDX_NO2I];
    data[243] = 0.0 + k[1099]*y[IDX_NOI];
    data[244] = 0.0 + k[1052]*y[IDX_NOI] + k[1055]*y[IDX_OCNI];
    data[245] = 0.0 - k[504]*y[IDX_NO2I];
    data[246] = 0.0 + k[1068]*y[IDX_HNOI] - k[1075]*y[IDX_NO2I];
    data[247] = 0.0 - k[952]*y[IDX_NO2I];
    data[248] = 0.0 - k[986]*y[IDX_NO2I];
    data[249] = 0.0 - k[356]*y[IDX_eM] - k[357]*y[IDX_eM] - k[834]*y[IDX_OI] -
        k[1204]*y[IDX_H2I] - k[1236];
    data[250] = 0.0 + k[602]*y[IDX_H3II] + k[611]*y[IDX_H3OII] + k[637]*y[IDX_HCOII];
    data[251] = 0.0 + k[1203]*y[IDX_H2I];
    data[252] = 0.0 + k[446]*y[IDX_CH3II] + k[507]*y[IDX_HII];
    data[253] = 0.0 + k[26]*y[IDX_CII] + k[96]*y[IDX_HII] + k[1183];
    data[254] = 0.0 + k[446]*y[IDX_SiH4I];
    data[255] = 0.0 + k[611]*y[IDX_SiH2I];
    data[256] = 0.0 + k[26]*y[IDX_SiH3I];
    data[257] = 0.0 + k[602]*y[IDX_SiH2I];
    data[258] = 0.0 + k[96]*y[IDX_SiH3I] + k[507]*y[IDX_SiH4I];
    data[259] = 0.0 + k[637]*y[IDX_SiH2I];
    data[260] = 0.0 - k[834]*y[IDX_SiH3II];
    data[261] = 0.0 - k[356]*y[IDX_SiH3II] - k[357]*y[IDX_SiH3II];
    data[262] = 0.0 + k[1203]*y[IDX_SiHII] - k[1204]*y[IDX_SiH3II];
    data[263] = 0.0 - k[353]*y[IDX_eM] - k[354]*y[IDX_eM] - k[355]*y[IDX_eM] -
        k[833]*y[IDX_OI] - k[862]*y[IDX_O2I] - k[1234];
    data[264] = 0.0 + k[25]*y[IDX_CII] + k[95]*y[IDX_HII] + k[1180];
    data[265] = 0.0 + k[605]*y[IDX_H3II] + k[612]*y[IDX_H3OII] + k[639]*y[IDX_HCOII] +
        k[849]*y[IDX_OHII];
    data[266] = 0.0 + k[506]*y[IDX_HII] + k[706]*y[IDX_HeII];
    data[267] = 0.0 + k[1202]*y[IDX_H2I];
    data[268] = 0.0 + k[612]*y[IDX_SiHI];
    data[269] = 0.0 + k[25]*y[IDX_SiH2I];
    data[270] = 0.0 + k[849]*y[IDX_SiHI];
    data[271] = 0.0 + k[706]*y[IDX_SiH3I];
    data[272] = 0.0 + k[605]*y[IDX_SiHI];
    data[273] = 0.0 - k[862]*y[IDX_SiH2II];
    data[274] = 0.0 + k[95]*y[IDX_SiH2I] + k[506]*y[IDX_SiH3I];
    data[275] = 0.0 + k[639]*y[IDX_SiHI];
    data[276] = 0.0 - k[833]*y[IDX_SiH2II];
    data[277] = 0.0 - k[353]*y[IDX_SiH2II] - k[354]*y[IDX_SiH2II] - k[355]*y[IDX_SiH2II];
    data[278] = 0.0 + k[1202]*y[IDX_SiII];
    data[279] = 0.0 + k[1060]*y[IDX_OI];
    data[280] = 0.0 + k[945]*y[IDX_CNI];
    data[281] = 0.0 - k[283] - k[378]*y[IDX_CII] - k[697]*y[IDX_HeII] -
        k[698]*y[IDX_HeII] - k[874]*y[IDX_CI] - k[993]*y[IDX_HI] -
        k[994]*y[IDX_HI] - k[995]*y[IDX_HI] - k[1053]*y[IDX_NOI] -
        k[1054]*y[IDX_O2I] - k[1055]*y[IDX_O2I] - k[1078]*y[IDX_OI] -
        k[1079]*y[IDX_OI] - k[1172] - k[1225];
    data[282] = 0.0 - k[378]*y[IDX_OCNI];
    data[283] = 0.0 + k[1065]*y[IDX_OI];
    data[284] = 0.0 + k[945]*y[IDX_NO2I] + k[947]*y[IDX_NOI] + k[949]*y[IDX_O2I] +
        k[1091]*y[IDX_OHI];
    data[285] = 0.0 + k[1016]*y[IDX_NI];
    data[286] = 0.0 - k[697]*y[IDX_OCNI] - k[698]*y[IDX_OCNI];
    data[287] = 0.0 + k[932]*y[IDX_NOI];
    data[288] = 0.0 + k[932]*y[IDX_CHI] + k[947]*y[IDX_CNI] - k[1053]*y[IDX_OCNI];
    data[289] = 0.0 + k[1016]*y[IDX_HCOI];
    data[290] = 0.0 + k[1091]*y[IDX_CNI];
    data[291] = 0.0 + k[949]*y[IDX_CNI] - k[1054]*y[IDX_OCNI] - k[1055]*y[IDX_OCNI];
    data[292] = 0.0 - k[874]*y[IDX_OCNI];
    data[293] = 0.0 + k[1060]*y[IDX_H2CNI] + k[1065]*y[IDX_HCNI] - k[1078]*y[IDX_OCNI] -
        k[1079]*y[IDX_OCNI];
    data[294] = 0.0 - k[993]*y[IDX_OCNI] - k[994]*y[IDX_OCNI] - k[995]*y[IDX_OCNI];
    data[295] = 0.0 + k[358]*y[IDX_eM];
    data[296] = 0.0 + k[356]*y[IDX_eM];
    data[297] = 0.0 - k[25]*y[IDX_CII] - k[95]*y[IDX_HII] - k[289] - k[380]*y[IDX_CII] -
        k[505]*y[IDX_HII] - k[602]*y[IDX_H3II] - k[611]*y[IDX_H3OII] -
        k[637]*y[IDX_HCOII] - k[703]*y[IDX_HeII] - k[704]*y[IDX_HeII] -
        k[1085]*y[IDX_OI] - k[1086]*y[IDX_OI] - k[1180] - k[1181] - k[1233];
    data[298] = 0.0 + k[291] + k[1185];
    data[299] = 0.0 + k[290] + k[1182];
    data[300] = 0.0 - k[611]*y[IDX_SiH2I];
    data[301] = 0.0 - k[25]*y[IDX_SiH2I] - k[380]*y[IDX_SiH2I];
    data[302] = 0.0 - k[703]*y[IDX_SiH2I] - k[704]*y[IDX_SiH2I];
    data[303] = 0.0 - k[602]*y[IDX_SiH2I];
    data[304] = 0.0 - k[95]*y[IDX_SiH2I] - k[505]*y[IDX_SiH2I];
    data[305] = 0.0 - k[637]*y[IDX_SiH2I];
    data[306] = 0.0 - k[1085]*y[IDX_SiH2I] - k[1086]*y[IDX_SiH2I];
    data[307] = 0.0 + k[356]*y[IDX_SiH3II] + k[358]*y[IDX_SiH4II];
    data[308] = 0.0 + k[499]*y[IDX_HII] + k[675]*y[IDX_HeII];
    data[309] = 0.0 + k[834]*y[IDX_OI];
    data[310] = 0.0 + k[833]*y[IDX_OI] + k[862]*y[IDX_O2I];
    data[311] = 0.0 - k[363]*y[IDX_eM] - k[364]*y[IDX_eM] - k[1246];
    data[312] = 0.0 + k[547]*y[IDX_H2I];
    data[313] = 0.0 + k[860]*y[IDX_SiII];
    data[314] = 0.0 + k[606]*y[IDX_H3II] + k[613]*y[IDX_H3OII] + k[640]*y[IDX_HCOII] +
        k[850]*y[IDX_OHII];
    data[315] = 0.0 + k[572]*y[IDX_H2OI] + k[860]*y[IDX_CH3OHI];
    data[316] = 0.0 + k[613]*y[IDX_SiOI];
    data[317] = 0.0 + k[850]*y[IDX_SiOI];
    data[318] = 0.0 + k[675]*y[IDX_H2SiOI];
    data[319] = 0.0 + k[606]*y[IDX_SiOI];
    data[320] = 0.0 + k[862]*y[IDX_SiH2II];
    data[321] = 0.0 + k[499]*y[IDX_H2SiOI];
    data[322] = 0.0 + k[640]*y[IDX_SiOI];
    data[323] = 0.0 + k[572]*y[IDX_SiII];
    data[324] = 0.0 + k[833]*y[IDX_SiH2II] + k[834]*y[IDX_SiH3II];
    data[325] = 0.0 - k[363]*y[IDX_SiOHII] - k[364]*y[IDX_SiOHII];
    data[326] = 0.0 + k[547]*y[IDX_SiOII];
    data[327] = 0.0 + k[505]*y[IDX_HII] + k[704]*y[IDX_HeII];
    data[328] = 0.0 - k[352]*y[IDX_eM] - k[394]*y[IDX_CI] - k[477]*y[IDX_CHI] -
        k[573]*y[IDX_H2OI] - k[619]*y[IDX_HI] - k[832]*y[IDX_OI] - k[1179] -
        k[1203]*y[IDX_H2I] - k[1232];
    data[329] = 0.0 + k[708]*y[IDX_HeII];
    data[330] = 0.0 + k[98]*y[IDX_HII];
    data[331] = 0.0 + k[705]*y[IDX_HeII];
    data[332] = 0.0 + k[1209]*y[IDX_HI];
    data[333] = 0.0 + k[601]*y[IDX_H3II] + k[610]*y[IDX_H3OII] + k[848]*y[IDX_OHII] +
        k[861]*y[IDX_HCOII];
    data[334] = 0.0 + k[610]*y[IDX_SiI];
    data[335] = 0.0 + k[848]*y[IDX_SiI];
    data[336] = 0.0 + k[704]*y[IDX_SiH2I] + k[705]*y[IDX_SiH3I] + k[708]*y[IDX_SiH4I];
    data[337] = 0.0 - k[477]*y[IDX_SiHII];
    data[338] = 0.0 + k[601]*y[IDX_SiI];
    data[339] = 0.0 - k[394]*y[IDX_SiHII];
    data[340] = 0.0 + k[98]*y[IDX_SiHI] + k[505]*y[IDX_SiH2I];
    data[341] = 0.0 + k[861]*y[IDX_SiI];
    data[342] = 0.0 - k[573]*y[IDX_SiHII];
    data[343] = 0.0 - k[832]*y[IDX_SiHII];
    data[344] = 0.0 - k[352]*y[IDX_SiHII];
    data[345] = 0.0 - k[1203]*y[IDX_SiHII];
    data[346] = 0.0 - k[619]*y[IDX_SiHII] + k[1209]*y[IDX_SiII];
    data[347] = 0.0 + k[1363] + k[1364] + k[1365] + k[1366];
    data[348] = 0.0 + k[361]*y[IDX_eM] + k[575]*y[IDX_H2OI];
    data[349] = 0.0 - k[97]*y[IDX_HII] - k[291] - k[446]*y[IDX_CH3II] -
        k[507]*y[IDX_HII] - k[604]*y[IDX_H3II] - k[638]*y[IDX_HCOII] -
        k[707]*y[IDX_HeII] - k[708]*y[IDX_HeII] - k[950]*y[IDX_CNI] -
        k[1088]*y[IDX_OI] - k[1185] - k[1186] - k[1187] - k[1237];
    data[350] = 0.0 - k[446]*y[IDX_SiH4I];
    data[351] = 0.0 - k[950]*y[IDX_SiH4I];
    data[352] = 0.0 - k[707]*y[IDX_SiH4I] - k[708]*y[IDX_SiH4I];
    data[353] = 0.0 - k[604]*y[IDX_SiH4I];
    data[354] = 0.0 - k[97]*y[IDX_SiH4I] - k[507]*y[IDX_SiH4I];
    data[355] = 0.0 - k[638]*y[IDX_SiH4I];
    data[356] = 0.0 + k[575]*y[IDX_SiH5II];
    data[357] = 0.0 - k[1088]*y[IDX_SiH4I];
    data[358] = 0.0 + k[361]*y[IDX_SiH5II];
    data[359] = 0.0 + k[357]*y[IDX_eM];
    data[360] = 0.0 + k[355]*y[IDX_eM];
    data[361] = 0.0 + k[289] + k[1181];
    data[362] = 0.0 + k[1187];
    data[363] = 0.0 - k[98]*y[IDX_HII] - k[292] - k[381]*y[IDX_CII] - k[508]*y[IDX_HII]
        - k[605]*y[IDX_H3II] - k[612]*y[IDX_H3OII] - k[639]*y[IDX_HCOII] -
        k[709]*y[IDX_HeII] - k[849]*y[IDX_OHII] - k[877]*y[IDX_CI] -
        k[1089]*y[IDX_OI] - k[1188] - k[1230];
    data[364] = 0.0 + k[1184];
    data[365] = 0.0 - k[612]*y[IDX_SiHI];
    data[366] = 0.0 - k[381]*y[IDX_SiHI];
    data[367] = 0.0 - k[849]*y[IDX_SiHI];
    data[368] = 0.0 - k[709]*y[IDX_SiHI];
    data[369] = 0.0 - k[605]*y[IDX_SiHI];
    data[370] = 0.0 - k[877]*y[IDX_SiHI];
    data[371] = 0.0 - k[98]*y[IDX_SiHI] - k[508]*y[IDX_SiHI];
    data[372] = 0.0 - k[639]*y[IDX_SiHI];
    data[373] = 0.0 - k[1089]*y[IDX_SiHI];
    data[374] = 0.0 + k[355]*y[IDX_SiH2II] + k[357]*y[IDX_SiH3II];
    data[375] = 0.0 + k[360]*y[IDX_eM];
    data[376] = 0.0 + k[359]*y[IDX_eM] + k[489]*y[IDX_COI] + k[574]*y[IDX_H2OI];
    data[377] = 0.0 + k[950]*y[IDX_CNI] + k[1088]*y[IDX_OI] + k[1186];
    data[378] = 0.0 - k[26]*y[IDX_CII] - k[96]*y[IDX_HII] - k[290] - k[506]*y[IDX_HII] -
        k[603]*y[IDX_H3II] - k[705]*y[IDX_HeII] - k[706]*y[IDX_HeII] -
        k[1087]*y[IDX_OI] - k[1182] - k[1183] - k[1184] - k[1235];
    data[379] = 0.0 - k[26]*y[IDX_SiH3I];
    data[380] = 0.0 + k[950]*y[IDX_SiH4I];
    data[381] = 0.0 - k[705]*y[IDX_SiH3I] - k[706]*y[IDX_SiH3I];
    data[382] = 0.0 - k[603]*y[IDX_SiH3I];
    data[383] = 0.0 - k[96]*y[IDX_SiH3I] - k[506]*y[IDX_SiH3I];
    data[384] = 0.0 + k[574]*y[IDX_SiH4II];
    data[385] = 0.0 - k[1087]*y[IDX_SiH3I] + k[1088]*y[IDX_SiH4I];
    data[386] = 0.0 + k[359]*y[IDX_SiH4II] + k[360]*y[IDX_SiH5II];
    data[387] = 0.0 + k[489]*y[IDX_SiH4II];
    data[388] = 0.0 + k[831]*y[IDX_OI];
    data[389] = 0.0 + k[832]*y[IDX_OI];
    data[390] = 0.0 - k[138]*y[IDX_HCOI] - k[154]*y[IDX_MgI] - k[206]*y[IDX_NOI] -
        k[362]*y[IDX_eM] - k[395]*y[IDX_CI] - k[438]*y[IDX_CH2I] -
        k[478]*y[IDX_CHI] - k[490]*y[IDX_COI] - k[547]*y[IDX_H2I] -
        k[745]*y[IDX_NI] - k[746]*y[IDX_NI] - k[835]*y[IDX_OI] - k[1189] -
        k[1245];
    data[391] = 0.0 - k[154]*y[IDX_SiOII];
    data[392] = 0.0 + k[99]*y[IDX_HII] + k[1191];
    data[393] = 0.0 + k[859]*y[IDX_OHI] + k[1212]*y[IDX_OI];
    data[394] = 0.0 - k[438]*y[IDX_SiOII];
    data[395] = 0.0 - k[138]*y[IDX_SiOII];
    data[396] = 0.0 - k[478]*y[IDX_SiOII];
    data[397] = 0.0 - k[206]*y[IDX_SiOII];
    data[398] = 0.0 - k[745]*y[IDX_SiOII] - k[746]*y[IDX_SiOII];
    data[399] = 0.0 + k[859]*y[IDX_SiII];
    data[400] = 0.0 - k[395]*y[IDX_SiOII];
    data[401] = 0.0 + k[99]*y[IDX_SiOI];
    data[402] = 0.0 + k[831]*y[IDX_SiCII] + k[832]*y[IDX_SiHII] - k[835]*y[IDX_SiOII] +
        k[1212]*y[IDX_SiII];
    data[403] = 0.0 - k[362]*y[IDX_SiOII];
    data[404] = 0.0 - k[490]*y[IDX_SiOII];
    data[405] = 0.0 - k[547]*y[IDX_SiOII];
    data[406] = 0.0 - k[331]*y[IDX_eM] - k[332]*y[IDX_eM] - k[333]*y[IDX_eM] -
        k[387]*y[IDX_CI] - k[485]*y[IDX_COI] - k[567]*y[IDX_H2OI] -
        k[824]*y[IDX_OI] - k[1287];
    data[407] = 0.0 + k[447]*y[IDX_CO2I];
    data[408] = 0.0 + k[734]*y[IDX_CO2I];
    data[409] = 0.0 + k[821]*y[IDX_CO2I];
    data[410] = 0.0 + k[652]*y[IDX_CO2I];
    data[411] = 0.0 + k[514]*y[IDX_CO2I];
    data[412] = 0.0 + k[447]*y[IDX_CH4II] + k[514]*y[IDX_H2II] + k[582]*y[IDX_H3II] +
        k[620]*y[IDX_HCNII] + k[652]*y[IDX_HNOII] + k[734]*y[IDX_N2HII] +
        k[748]*y[IDX_NHII] + k[821]*y[IDX_O2HII] + k[837]*y[IDX_OHII];
    data[413] = 0.0 + k[620]*y[IDX_CO2I];
    data[414] = 0.0 + k[748]*y[IDX_CO2I];
    data[415] = 0.0 + k[837]*y[IDX_CO2I];
    data[416] = 0.0 + k[582]*y[IDX_CO2I];
    data[417] = 0.0 + k[855]*y[IDX_HCOII];
    data[418] = 0.0 - k[387]*y[IDX_HCO2II];
    data[419] = 0.0 + k[855]*y[IDX_OHI];
    data[420] = 0.0 - k[567]*y[IDX_HCO2II];
    data[421] = 0.0 - k[824]*y[IDX_HCO2II];
    data[422] = 0.0 - k[331]*y[IDX_HCO2II] - k[332]*y[IDX_HCO2II] - k[333]*y[IDX_HCO2II];
    data[423] = 0.0 - k[485]*y[IDX_HCO2II];
    data[424] = 0.0 + k[1351] + k[1352] + k[1353] + k[1354];
    data[425] = 0.0 + k[310]*y[IDX_eM];
    data[426] = 0.0 + k[909]*y[IDX_CH3I] + k[1041]*y[IDX_NHI];
    data[427] = 0.0 - k[264] - k[503]*y[IDX_HII] - k[590]*y[IDX_H3II] -
        k[686]*y[IDX_HeII] - k[687]*y[IDX_HeII] - k[883]*y[IDX_CH2I] -
        k[906]*y[IDX_CH3I] - k[926]*y[IDX_CHI] - k[944]*y[IDX_CNI] -
        k[951]*y[IDX_COI] - k[980]*y[IDX_HI] - k[981]*y[IDX_HI] -
        k[982]*y[IDX_HI] - k[999]*y[IDX_HCOI] - k[1017]*y[IDX_NI] -
        k[1068]*y[IDX_OI] - k[1069]*y[IDX_OI] - k[1070]*y[IDX_OI] -
        k[1097]*y[IDX_OHI] - k[1153] - k[1294];
    data[428] = 0.0 + k[204]*y[IDX_NOI];
    data[429] = 0.0 - k[906]*y[IDX_HNOI] + k[909]*y[IDX_NO2I];
    data[430] = 0.0 + k[1072]*y[IDX_OI];
    data[431] = 0.0 - k[883]*y[IDX_HNOI];
    data[432] = 0.0 + k[1041]*y[IDX_NO2I] + k[1044]*y[IDX_O2I] + k[1049]*y[IDX_OHI];
    data[433] = 0.0 - k[944]*y[IDX_HNOI];
    data[434] = 0.0 - k[999]*y[IDX_HNOI] + k[1000]*y[IDX_NOI];
    data[435] = 0.0 - k[686]*y[IDX_HNOI] - k[687]*y[IDX_HNOI];
    data[436] = 0.0 - k[926]*y[IDX_HNOI];
    data[437] = 0.0 - k[590]*y[IDX_HNOI];
    data[438] = 0.0 + k[204]*y[IDX_HNOII] + k[1000]*y[IDX_HCOI];
    data[439] = 0.0 - k[1017]*y[IDX_HNOI];
    data[440] = 0.0 + k[1049]*y[IDX_NHI] - k[1097]*y[IDX_HNOI];
    data[441] = 0.0 + k[1044]*y[IDX_NHI];
    data[442] = 0.0 - k[503]*y[IDX_HNOI];
    data[443] = 0.0 - k[1068]*y[IDX_HNOI] - k[1069]*y[IDX_HNOI] - k[1070]*y[IDX_HNOI] +
        k[1072]*y[IDX_NH2I];
    data[444] = 0.0 + k[310]*y[IDX_H2NOII];
    data[445] = 0.0 - k[951]*y[IDX_HNOI];
    data[446] = 0.0 - k[980]*y[IDX_HNOI] - k[981]*y[IDX_HNOI] - k[982]*y[IDX_HNOI];
    data[447] = 0.0 + k[1359] + k[1360] + k[1361] + k[1362];
    data[448] = 0.0 - k[247] - k[248] - k[365]*y[IDX_CII] - k[366]*y[IDX_CII] -
        k[396]*y[IDX_CHII] - k[397]*y[IDX_CHII] - k[439]*y[IDX_CH3II] -
        k[492]*y[IDX_HII] - k[493]*y[IDX_HII] - k[494]*y[IDX_HII] -
        k[579]*y[IDX_H3II] - k[656]*y[IDX_HeII] - k[657]*y[IDX_HeII] -
        k[712]*y[IDX_NII] - k[713]*y[IDX_NII] - k[714]*y[IDX_NII] -
        k[715]*y[IDX_NII] - k[808]*y[IDX_OII] - k[809]*y[IDX_OII] -
        k[820]*y[IDX_O2II] - k[860]*y[IDX_SiII] - k[1119] - k[1120] - k[1121] -
        k[1223];
    data[449] = 0.0 - k[860]*y[IDX_CH3OHI];
    data[450] = 0.0 - k[712]*y[IDX_CH3OHI] - k[713]*y[IDX_CH3OHI] - k[714]*y[IDX_CH3OHI]
        - k[715]*y[IDX_CH3OHI];
    data[451] = 0.0 - k[808]*y[IDX_CH3OHI] - k[809]*y[IDX_CH3OHI];
    data[452] = 0.0 - k[820]*y[IDX_CH3OHI];
    data[453] = 0.0 - k[439]*y[IDX_CH3OHI];
    data[454] = 0.0 - k[365]*y[IDX_CH3OHI] - k[366]*y[IDX_CH3OHI];
    data[455] = 0.0 - k[396]*y[IDX_CH3OHI] - k[397]*y[IDX_CH3OHI];
    data[456] = 0.0 - k[656]*y[IDX_CH3OHI] - k[657]*y[IDX_CH3OHI];
    data[457] = 0.0 - k[579]*y[IDX_CH3OHI];
    data[458] = 0.0 - k[492]*y[IDX_CH3OHI] - k[493]*y[IDX_CH3OHI] - k[494]*y[IDX_CH3OHI];
    data[459] = 0.0 + k[1319] + k[1320] + k[1321] + k[1322];
    data[460] = 0.0 - k[154]*y[IDX_MgI];
    data[461] = 0.0 - k[18]*y[IDX_CII] - k[32]*y[IDX_CHII] - k[47]*y[IDX_CH3II] -
        k[83]*y[IDX_HII] - k[119]*y[IDX_H2OII] - k[148]*y[IDX_H2COII] -
        k[149]*y[IDX_HCOII] - k[150]*y[IDX_N2II] - k[151]*y[IDX_NOII] -
        k[152]*y[IDX_O2II] - k[153]*y[IDX_SiII] - k[154]*y[IDX_SiOII] -
        k[163]*y[IDX_NII] - k[190]*y[IDX_NH3II] - k[266] - k[591]*y[IDX_H3II] -
        k[1154] - k[1300];
    data[462] = 0.0 + k[1219]*y[IDX_eM];
    data[463] = 0.0 - k[153]*y[IDX_MgI];
    data[464] = 0.0 - k[150]*y[IDX_MgI];
    data[465] = 0.0 - k[163]*y[IDX_MgI];
    data[466] = 0.0 - k[152]*y[IDX_MgI];
    data[467] = 0.0 - k[47]*y[IDX_MgI];
    data[468] = 0.0 - k[119]*y[IDX_MgI];
    data[469] = 0.0 - k[190]*y[IDX_MgI];
    data[470] = 0.0 - k[151]*y[IDX_MgI];
    data[471] = 0.0 - k[18]*y[IDX_MgI];
    data[472] = 0.0 - k[32]*y[IDX_MgI];
    data[473] = 0.0 - k[148]*y[IDX_MgI];
    data[474] = 0.0 - k[591]*y[IDX_MgI];
    data[475] = 0.0 - k[83]*y[IDX_MgI];
    data[476] = 0.0 - k[149]*y[IDX_MgI];
    data[477] = 0.0 + k[1219]*y[IDX_MgII];
    data[478] = 0.0 + k[154]*y[IDX_MgI];
    data[479] = 0.0 + k[18]*y[IDX_CII] + k[32]*y[IDX_CHII] + k[47]*y[IDX_CH3II] +
        k[83]*y[IDX_HII] + k[119]*y[IDX_H2OII] + k[148]*y[IDX_H2COII] +
        k[149]*y[IDX_HCOII] + k[150]*y[IDX_N2II] + k[151]*y[IDX_NOII] +
        k[152]*y[IDX_O2II] + k[153]*y[IDX_SiII] + k[154]*y[IDX_SiOII] +
        k[163]*y[IDX_NII] + k[190]*y[IDX_NH3II] + k[266] + k[591]*y[IDX_H3II] +
        k[1154];
    data[480] = 0.0 - k[1219]*y[IDX_eM] - k[1299];
    data[481] = 0.0 + k[153]*y[IDX_MgI];
    data[482] = 0.0 + k[150]*y[IDX_MgI];
    data[483] = 0.0 + k[163]*y[IDX_MgI];
    data[484] = 0.0 + k[152]*y[IDX_MgI];
    data[485] = 0.0 + k[47]*y[IDX_MgI];
    data[486] = 0.0 + k[119]*y[IDX_MgI];
    data[487] = 0.0 + k[190]*y[IDX_MgI];
    data[488] = 0.0 + k[151]*y[IDX_MgI];
    data[489] = 0.0 + k[18]*y[IDX_MgI];
    data[490] = 0.0 + k[32]*y[IDX_MgI];
    data[491] = 0.0 + k[148]*y[IDX_MgI];
    data[492] = 0.0 + k[591]*y[IDX_MgI];
    data[493] = 0.0 + k[83]*y[IDX_MgI];
    data[494] = 0.0 + k[149]*y[IDX_MgI];
    data[495] = 0.0 - k[1219]*y[IDX_MgII];
    data[496] = 0.0 - k[49]*y[IDX_H2COI] - k[50]*y[IDX_NH3I] - k[51]*y[IDX_O2I] -
        k[301]*y[IDX_eM] - k[302]*y[IDX_eM] - k[447]*y[IDX_CO2I] -
        k[448]*y[IDX_COI] - k[449]*y[IDX_H2COI] - k[450]*y[IDX_H2OI] -
        k[617]*y[IDX_HI] - k[822]*y[IDX_OI] - k[1122] - k[1123] - k[1288];
    data[497] = 0.0 + k[52]*y[IDX_COII] + k[77]*y[IDX_HII] + k[101]*y[IDX_H2II] +
        k[140]*y[IDX_HeII] + k[156]*y[IDX_NII] + k[207]*y[IDX_OII] + k[1126];
    data[498] = 0.0 + k[52]*y[IDX_CH4I];
    data[499] = 0.0 + k[101]*y[IDX_CH4I];
    data[500] = 0.0 - k[50]*y[IDX_CH4II];
    data[501] = 0.0 + k[578]*y[IDX_H3II];
    data[502] = 0.0 - k[447]*y[IDX_CH4II];
    data[503] = 0.0 + k[156]*y[IDX_CH4I];
    data[504] = 0.0 + k[207]*y[IDX_CH4I];
    data[505] = 0.0 + k[441]*y[IDX_HCOI];
    data[506] = 0.0 - k[49]*y[IDX_CH4II] - k[449]*y[IDX_CH4II];
    data[507] = 0.0 + k[441]*y[IDX_CH3II];
    data[508] = 0.0 + k[140]*y[IDX_CH4I];
    data[509] = 0.0 + k[578]*y[IDX_CH3I];
    data[510] = 0.0 - k[51]*y[IDX_CH4II];
    data[511] = 0.0 + k[77]*y[IDX_CH4I];
    data[512] = 0.0 - k[450]*y[IDX_CH4II];
    data[513] = 0.0 - k[822]*y[IDX_CH4II];
    data[514] = 0.0 - k[301]*y[IDX_CH4II] - k[302]*y[IDX_CH4II];
    data[515] = 0.0 - k[448]*y[IDX_CH4II];
    data[516] = 0.0 - k[617]*y[IDX_CH4II];
    data[517] = 0.0 + k[1379] + k[1380] + k[1381] + k[1382];
    data[518] = 0.0 + k[257] + k[1143] + k[1144];
    data[519] = 0.0 + k[1084]*y[IDX_OI];
    data[520] = 0.0 + k[1085]*y[IDX_OI] + k[1086]*y[IDX_OI];
    data[521] = 0.0 + k[364]*y[IDX_eM];
    data[522] = 0.0 + k[1089]*y[IDX_OI];
    data[523] = 0.0 + k[138]*y[IDX_HCOI] + k[154]*y[IDX_MgI] + k[206]*y[IDX_NOI];
    data[524] = 0.0 + k[154]*y[IDX_SiOII];
    data[525] = 0.0 - k[99]*y[IDX_HII] - k[293] - k[382]*y[IDX_CII] - k[606]*y[IDX_H3II]
        - k[613]*y[IDX_H3OII] - k[640]*y[IDX_HCOII] - k[710]*y[IDX_HeII] -
        k[711]*y[IDX_HeII] - k[850]*y[IDX_OHII] - k[1190] - k[1191] - k[1229];
    data[526] = 0.0 + k[1102]*y[IDX_OHI] + k[1103]*y[IDX_CO2I] + k[1104]*y[IDX_COI] +
        k[1105]*y[IDX_NOI] + k[1106]*y[IDX_O2I] + k[1213]*y[IDX_OI];
    data[527] = 0.0 + k[1103]*y[IDX_SiI];
    data[528] = 0.0 - k[613]*y[IDX_SiOI];
    data[529] = 0.0 - k[382]*y[IDX_SiOI];
    data[530] = 0.0 - k[850]*y[IDX_SiOI];
    data[531] = 0.0 + k[138]*y[IDX_SiOII];
    data[532] = 0.0 - k[710]*y[IDX_SiOI] - k[711]*y[IDX_SiOI];
    data[533] = 0.0 - k[606]*y[IDX_SiOI];
    data[534] = 0.0 + k[206]*y[IDX_SiOII] + k[1105]*y[IDX_SiI];
    data[535] = 0.0 + k[1102]*y[IDX_SiI];
    data[536] = 0.0 + k[1106]*y[IDX_SiI];
    data[537] = 0.0 - k[99]*y[IDX_SiOI];
    data[538] = 0.0 - k[640]*y[IDX_SiOI];
    data[539] = 0.0 + k[1084]*y[IDX_SiCI] + k[1085]*y[IDX_SiH2I] + k[1086]*y[IDX_SiH2I]
        + k[1089]*y[IDX_SiHI] + k[1213]*y[IDX_SiI];
    data[540] = 0.0 + k[364]*y[IDX_SiOHII];
    data[541] = 0.0 + k[1104]*y[IDX_SiI];
    data[542] = 0.0 + k[697]*y[IDX_HeII];
    data[543] = 0.0 - k[27]*y[IDX_CI] - k[37]*y[IDX_CH2I] - k[53]*y[IDX_CHI] -
        k[63]*y[IDX_COI] - k[64]*y[IDX_H2COI] - k[65]*y[IDX_HCNI] -
        k[66]*y[IDX_HCOI] - k[67]*y[IDX_NOI] - k[68]*y[IDX_O2I] -
        k[126]*y[IDX_HI] - k[183]*y[IDX_NH2I] - k[199]*y[IDX_NHI] -
        k[216]*y[IDX_OI] - k[225]*y[IDX_OHI] - k[303]*y[IDX_eM] -
        k[479]*y[IDX_H2COI] - k[480]*y[IDX_HCOI] - k[481]*y[IDX_O2I] -
        k[531]*y[IDX_H2I] - k[560]*y[IDX_H2OI] - k[561]*y[IDX_H2OI] -
        k[737]*y[IDX_NI] - k[1276];
    data[544] = 0.0 + k[683]*y[IDX_HeII];
    data[545] = 0.0 + k[69]*y[IDX_CNI];
    data[546] = 0.0 + k[103]*y[IDX_CNI];
    data[547] = 0.0 + k[157]*y[IDX_CNI] + k[468]*y[IDX_CHI];
    data[548] = 0.0 - k[183]*y[IDX_CNII];
    data[549] = 0.0 + k[375]*y[IDX_NHI] + k[1192]*y[IDX_NI];
    data[550] = 0.0 - k[65]*y[IDX_CNII] + k[676]*y[IDX_HeII];
    data[551] = 0.0 + k[408]*y[IDX_NI] + k[410]*y[IDX_NHI];
    data[552] = 0.0 - k[37]*y[IDX_CNII];
    data[553] = 0.0 - k[199]*y[IDX_CNII] + k[375]*y[IDX_CII] + k[410]*y[IDX_CHII];
    data[554] = 0.0 + k[69]*y[IDX_N2II] + k[103]*y[IDX_H2II] + k[157]*y[IDX_NII];
    data[555] = 0.0 - k[64]*y[IDX_CNII] - k[479]*y[IDX_CNII];
    data[556] = 0.0 - k[66]*y[IDX_CNII] - k[480]*y[IDX_CNII];
    data[557] = 0.0 + k[676]*y[IDX_HCNI] + k[683]*y[IDX_HNCI] + k[697]*y[IDX_OCNI];
    data[558] = 0.0 - k[53]*y[IDX_CNII] + k[468]*y[IDX_NII];
    data[559] = 0.0 - k[67]*y[IDX_CNII];
    data[560] = 0.0 + k[408]*y[IDX_CHII] - k[737]*y[IDX_CNII] + k[1192]*y[IDX_CII];
    data[561] = 0.0 - k[225]*y[IDX_CNII];
    data[562] = 0.0 - k[68]*y[IDX_CNII] - k[481]*y[IDX_CNII];
    data[563] = 0.0 - k[27]*y[IDX_CNII];
    data[564] = 0.0 - k[560]*y[IDX_CNII] - k[561]*y[IDX_CNII];
    data[565] = 0.0 - k[216]*y[IDX_CNII];
    data[566] = 0.0 - k[303]*y[IDX_CNII];
    data[567] = 0.0 - k[63]*y[IDX_CNII];
    data[568] = 0.0 - k[531]*y[IDX_CNII];
    data[569] = 0.0 - k[126]*y[IDX_CNII];
    data[570] = 0.0 - k[327]*y[IDX_eM] - k[328]*y[IDX_eM] - k[329]*y[IDX_eM] -
        k[427]*y[IDX_CH2I] - k[428]*y[IDX_CH2I] - k[464]*y[IDX_CHI] -
        k[465]*y[IDX_CHI] - k[633]*y[IDX_H2COI] - k[634]*y[IDX_H2COI] -
        k[785]*y[IDX_NH2I] - k[786]*y[IDX_NH2I] - k[1301];
    data[571] = 0.0 + k[631]*y[IDX_HCNI] + k[650]*y[IDX_HNCI];
    data[572] = 0.0 + k[632]*y[IDX_HCNI] + k[651]*y[IDX_HNCI];
    data[573] = 0.0 + k[407]*y[IDX_CHII] + k[559]*y[IDX_H2OII] + k[589]*y[IDX_H3II] +
        k[609]*y[IDX_H3OII] + k[626]*y[IDX_HCNII] + k[646]*y[IDX_H2COII] +
        k[647]*y[IDX_H3COII] + k[648]*y[IDX_HCOII] + k[649]*y[IDX_HNOII] +
        k[650]*y[IDX_N2HII] + k[651]*y[IDX_O2HII] + k[760]*y[IDX_NHII] +
        k[775]*y[IDX_NH2II] + k[844]*y[IDX_OHII];
    data[574] = 0.0 + k[630]*y[IDX_HCNI] + k[649]*y[IDX_HNCI];
    data[575] = 0.0 + k[628]*y[IDX_HCNI] + k[647]*y[IDX_HNCI];
    data[576] = 0.0 + k[454]*y[IDX_HCNII] + k[718]*y[IDX_NII];
    data[577] = 0.0 + k[793]*y[IDX_HCNII];
    data[578] = 0.0 + k[718]*y[IDX_CH4I];
    data[579] = 0.0 + k[454]*y[IDX_CH4I] + k[535]*y[IDX_H2I] + k[623]*y[IDX_HCNI] +
        k[625]*y[IDX_HCOI] + k[626]*y[IDX_HNCI] + k[793]*y[IDX_NH3I];
    data[580] = 0.0 + k[773]*y[IDX_HCNI] + k[775]*y[IDX_HNCI];
    data[581] = 0.0 + k[758]*y[IDX_HCNI] + k[760]*y[IDX_HNCI];
    data[582] = 0.0 + k[794]*y[IDX_NHI];
    data[583] = 0.0 - k[785]*y[IDX_HCNHII] - k[786]*y[IDX_HCNHII];
    data[584] = 0.0 + k[556]*y[IDX_HCNI] + k[559]*y[IDX_HNCI];
    data[585] = 0.0 + k[608]*y[IDX_HCNI] + k[609]*y[IDX_HNCI];
    data[586] = 0.0 + k[405]*y[IDX_CHII] + k[556]*y[IDX_H2OII] + k[587]*y[IDX_H3II] +
        k[608]*y[IDX_H3OII] + k[623]*y[IDX_HCNII] + k[627]*y[IDX_H2COII] +
        k[628]*y[IDX_H3COII] + k[629]*y[IDX_HCOII] + k[630]*y[IDX_HNOII] +
        k[631]*y[IDX_N2HII] + k[632]*y[IDX_O2HII] + k[758]*y[IDX_NHII] +
        k[773]*y[IDX_NH2II] + k[841]*y[IDX_OHII];
    data[587] = 0.0 + k[405]*y[IDX_HCNI] + k[407]*y[IDX_HNCI];
    data[588] = 0.0 - k[427]*y[IDX_HCNHII] - k[428]*y[IDX_HCNHII];
    data[589] = 0.0 + k[627]*y[IDX_HCNI] + k[646]*y[IDX_HNCI];
    data[590] = 0.0 + k[794]*y[IDX_CH3II];
    data[591] = 0.0 + k[841]*y[IDX_HCNI] + k[844]*y[IDX_HNCI];
    data[592] = 0.0 - k[633]*y[IDX_HCNHII] - k[634]*y[IDX_HCNHII];
    data[593] = 0.0 + k[625]*y[IDX_HCNII];
    data[594] = 0.0 - k[464]*y[IDX_HCNHII] - k[465]*y[IDX_HCNHII];
    data[595] = 0.0 + k[587]*y[IDX_HCNI] + k[589]*y[IDX_HNCI];
    data[596] = 0.0 + k[629]*y[IDX_HCNI] + k[648]*y[IDX_HNCI];
    data[597] = 0.0 - k[327]*y[IDX_HCNHII] - k[328]*y[IDX_HCNHII] - k[329]*y[IDX_HCNHII];
    data[598] = 0.0 + k[535]*y[IDX_HCNII];
    data[599] = 0.0 - k[338]*y[IDX_eM] - k[339]*y[IDX_eM] - k[389]*y[IDX_CI] -
        k[431]*y[IDX_CH2I] - k[469]*y[IDX_CHI] - k[487]*y[IDX_COI] -
        k[570]*y[IDX_H2OI] - k[631]*y[IDX_HCNI] - k[643]*y[IDX_HCOI] -
        k[650]*y[IDX_HNCI] - k[734]*y[IDX_CO2I] - k[735]*y[IDX_H2COI] -
        k[789]*y[IDX_NH2I] - k[801]*y[IDX_NHI] - k[826]*y[IDX_OI] -
        k[857]*y[IDX_OHI] - k[1302];
    data[600] = 0.0 + k[733]*y[IDX_N2I];
    data[601] = 0.0 - k[650]*y[IDX_N2HII];
    data[602] = 0.0 + k[732]*y[IDX_N2I];
    data[603] = 0.0 + k[539]*y[IDX_H2I] + k[569]*y[IDX_H2OI] + k[731]*y[IDX_HCOI];
    data[604] = 0.0 + k[521]*y[IDX_N2I];
    data[605] = 0.0 + k[724]*y[IDX_NII];
    data[606] = 0.0 - k[734]*y[IDX_N2HII];
    data[607] = 0.0 + k[724]*y[IDX_NH3I];
    data[608] = 0.0 + k[741]*y[IDX_NI];
    data[609] = 0.0 + k[761]*y[IDX_N2I] + k[764]*y[IDX_NOI];
    data[610] = 0.0 - k[789]*y[IDX_N2HII];
    data[611] = 0.0 + k[521]*y[IDX_H2II] + k[592]*y[IDX_H3II] + k[732]*y[IDX_HNOII] +
        k[733]*y[IDX_O2HII] + k[761]*y[IDX_NHII] + k[845]*y[IDX_OHII];
    data[612] = 0.0 - k[631]*y[IDX_N2HII];
    data[613] = 0.0 - k[431]*y[IDX_N2HII];
    data[614] = 0.0 - k[801]*y[IDX_N2HII];
    data[615] = 0.0 + k[845]*y[IDX_N2I];
    data[616] = 0.0 - k[735]*y[IDX_N2HII];
    data[617] = 0.0 - k[643]*y[IDX_N2HII] + k[731]*y[IDX_N2II];
    data[618] = 0.0 - k[469]*y[IDX_N2HII];
    data[619] = 0.0 + k[592]*y[IDX_N2I];
    data[620] = 0.0 + k[764]*y[IDX_NHII];
    data[621] = 0.0 + k[741]*y[IDX_NH2II];
    data[622] = 0.0 - k[857]*y[IDX_N2HII];
    data[623] = 0.0 - k[389]*y[IDX_N2HII];
    data[624] = 0.0 + k[569]*y[IDX_N2II] - k[570]*y[IDX_N2HII];
    data[625] = 0.0 - k[826]*y[IDX_N2HII];
    data[626] = 0.0 - k[338]*y[IDX_N2HII] - k[339]*y[IDX_N2HII];
    data[627] = 0.0 - k[487]*y[IDX_N2HII];
    data[628] = 0.0 + k[539]*y[IDX_N2II];
    data[629] = 0.0 - k[347]*y[IDX_eM] - k[392]*y[IDX_CI] - k[436]*y[IDX_CH2I] -
        k[474]*y[IDX_CHI] - k[483]*y[IDX_CNI] - k[488]*y[IDX_COI] -
        k[544]*y[IDX_H2I] - k[552]*y[IDX_H2COI] - k[571]*y[IDX_H2OI] -
        k[632]*y[IDX_HCNI] - k[645]*y[IDX_HCOI] - k[651]*y[IDX_HNCI] -
        k[733]*y[IDX_N2I] - k[790]*y[IDX_NH2I] - k[805]*y[IDX_NHI] -
        k[807]*y[IDX_NOI] - k[821]*y[IDX_CO2I] - k[829]*y[IDX_OI] -
        k[858]*y[IDX_OHI] - k[1297];
    data[630] = 0.0 - k[651]*y[IDX_O2HII];
    data[631] = 0.0 + k[525]*y[IDX_O2I];
    data[632] = 0.0 - k[821]*y[IDX_O2HII];
    data[633] = 0.0 + k[766]*y[IDX_O2I];
    data[634] = 0.0 + k[644]*y[IDX_HCOI];
    data[635] = 0.0 - k[790]*y[IDX_O2HII];
    data[636] = 0.0 - k[733]*y[IDX_O2HII];
    data[637] = 0.0 - k[632]*y[IDX_O2HII];
    data[638] = 0.0 - k[436]*y[IDX_O2HII];
    data[639] = 0.0 - k[805]*y[IDX_O2HII];
    data[640] = 0.0 - k[483]*y[IDX_O2HII];
    data[641] = 0.0 - k[552]*y[IDX_O2HII];
    data[642] = 0.0 + k[644]*y[IDX_O2II] - k[645]*y[IDX_O2HII];
    data[643] = 0.0 - k[474]*y[IDX_O2HII];
    data[644] = 0.0 + k[597]*y[IDX_O2I];
    data[645] = 0.0 - k[807]*y[IDX_O2HII];
    data[646] = 0.0 - k[858]*y[IDX_O2HII];
    data[647] = 0.0 + k[525]*y[IDX_H2II] + k[597]*y[IDX_H3II] + k[766]*y[IDX_NHII];
    data[648] = 0.0 - k[392]*y[IDX_O2HII];
    data[649] = 0.0 - k[571]*y[IDX_O2HII];
    data[650] = 0.0 - k[829]*y[IDX_O2HII];
    data[651] = 0.0 - k[347]*y[IDX_O2HII];
    data[652] = 0.0 - k[488]*y[IDX_O2HII];
    data[653] = 0.0 - k[544]*y[IDX_O2HII];
    data[654] = 0.0 + k[744]*y[IDX_NI];
    data[655] = 0.0 + k[701]*y[IDX_HeII];
    data[656] = 0.0 + k[703]*y[IDX_HeII];
    data[657] = 0.0 + k[619]*y[IDX_HI] + k[1179];
    data[658] = 0.0 + k[707]*y[IDX_HeII];
    data[659] = 0.0 + k[508]*y[IDX_HII] + k[709]*y[IDX_HeII];
    data[660] = 0.0 + k[395]*y[IDX_CI] + k[438]*y[IDX_CH2I] + k[490]*y[IDX_COI] +
        k[746]*y[IDX_NI] + k[835]*y[IDX_OI] + k[1189];
    data[661] = 0.0 - k[860]*y[IDX_SiII];
    data[662] = 0.0 - k[153]*y[IDX_SiII];
    data[663] = 0.0 + k[382]*y[IDX_CII] + k[710]*y[IDX_HeII];
    data[664] = 0.0 - k[153]*y[IDX_MgI] - k[476]*y[IDX_CHI] - k[572]*y[IDX_H2OI] -
        k[859]*y[IDX_OHI] - k[860]*y[IDX_CH3OHI] - k[1202]*y[IDX_H2I] -
        k[1209]*y[IDX_HI] - k[1212]*y[IDX_OI] - k[1222]*y[IDX_eM] - k[1231];
    data[665] = 0.0 + k[21]*y[IDX_CII] + k[35]*y[IDX_CHII] + k[91]*y[IDX_HII] +
        k[122]*y[IDX_H2OII] + k[147]*y[IDX_HeII] + k[192]*y[IDX_NH3II] +
        k[228]*y[IDX_H2COII] + k[229]*y[IDX_NOII] + k[230]*y[IDX_O2II] + k[285]
        + k[1176];
    data[666] = 0.0 + k[230]*y[IDX_SiI];
    data[667] = 0.0 + k[122]*y[IDX_SiI];
    data[668] = 0.0 + k[192]*y[IDX_SiI];
    data[669] = 0.0 + k[229]*y[IDX_SiI];
    data[670] = 0.0 + k[21]*y[IDX_SiI] + k[382]*y[IDX_SiOI];
    data[671] = 0.0 + k[35]*y[IDX_SiI];
    data[672] = 0.0 + k[438]*y[IDX_SiOII];
    data[673] = 0.0 + k[228]*y[IDX_SiI];
    data[674] = 0.0 + k[147]*y[IDX_SiI] + k[701]*y[IDX_SiCI] + k[703]*y[IDX_SiH2I] +
        k[707]*y[IDX_SiH4I] + k[709]*y[IDX_SiHI] + k[710]*y[IDX_SiOI];
    data[675] = 0.0 - k[476]*y[IDX_SiII];
    data[676] = 0.0 + k[744]*y[IDX_SiCII] + k[746]*y[IDX_SiOII];
    data[677] = 0.0 - k[859]*y[IDX_SiII];
    data[678] = 0.0 + k[395]*y[IDX_SiOII];
    data[679] = 0.0 + k[91]*y[IDX_SiI] + k[508]*y[IDX_SiHI];
    data[680] = 0.0 - k[572]*y[IDX_SiII];
    data[681] = 0.0 + k[835]*y[IDX_SiOII] - k[1212]*y[IDX_SiII];
    data[682] = 0.0 - k[1222]*y[IDX_SiII];
    data[683] = 0.0 + k[490]*y[IDX_SiOII];
    data[684] = 0.0 - k[1202]*y[IDX_SiII];
    data[685] = 0.0 + k[619]*y[IDX_SiHII] - k[1209]*y[IDX_SiII];
    data[686] = 0.0 + k[349]*y[IDX_eM];
    data[687] = 0.0 + k[288] + k[702]*y[IDX_HeII] + k[1027]*y[IDX_NI] +
        k[1083]*y[IDX_OI] + k[1178];
    data[688] = 0.0 + k[353]*y[IDX_eM] + k[354]*y[IDX_eM];
    data[689] = 0.0 + k[363]*y[IDX_eM];
    data[690] = 0.0 + k[352]*y[IDX_eM] + k[477]*y[IDX_CHI] + k[573]*y[IDX_H2OI];
    data[691] = 0.0 + k[292] + k[1188];
    data[692] = 0.0 + k[362]*y[IDX_eM] + k[478]*y[IDX_CHI] + k[745]*y[IDX_NI];
    data[693] = 0.0 + k[153]*y[IDX_SiII];
    data[694] = 0.0 + k[293] + k[711]*y[IDX_HeII] + k[1190];
    data[695] = 0.0 + k[153]*y[IDX_MgI] + k[1222]*y[IDX_eM];
    data[696] = 0.0 - k[21]*y[IDX_CII] - k[35]*y[IDX_CHII] - k[91]*y[IDX_HII] -
        k[122]*y[IDX_H2OII] - k[147]*y[IDX_HeII] - k[192]*y[IDX_NH3II] -
        k[228]*y[IDX_H2COII] - k[229]*y[IDX_NOII] - k[230]*y[IDX_O2II] - k[285]
        - k[601]*y[IDX_H3II] - k[610]*y[IDX_H3OII] - k[848]*y[IDX_OHII] -
        k[861]*y[IDX_HCOII] - k[1102]*y[IDX_OHI] - k[1103]*y[IDX_CO2I] -
        k[1104]*y[IDX_COI] - k[1105]*y[IDX_NOI] - k[1106]*y[IDX_O2I] - k[1176] -
        k[1213]*y[IDX_OI] - k[1228];
    data[697] = 0.0 - k[1103]*y[IDX_SiI];
    data[698] = 0.0 - k[230]*y[IDX_SiI];
    data[699] = 0.0 - k[122]*y[IDX_SiI];
    data[700] = 0.0 - k[192]*y[IDX_SiI];
    data[701] = 0.0 - k[229]*y[IDX_SiI];
    data[702] = 0.0 - k[610]*y[IDX_SiI];
    data[703] = 0.0 - k[21]*y[IDX_SiI];
    data[704] = 0.0 - k[35]*y[IDX_SiI];
    data[705] = 0.0 - k[228]*y[IDX_SiI];
    data[706] = 0.0 - k[848]*y[IDX_SiI];
    data[707] = 0.0 - k[147]*y[IDX_SiI] + k[702]*y[IDX_SiCI] + k[711]*y[IDX_SiOI];
    data[708] = 0.0 + k[477]*y[IDX_SiHII] + k[478]*y[IDX_SiOII];
    data[709] = 0.0 - k[601]*y[IDX_SiI];
    data[710] = 0.0 - k[1105]*y[IDX_SiI];
    data[711] = 0.0 + k[745]*y[IDX_SiOII] + k[1027]*y[IDX_SiCI];
    data[712] = 0.0 - k[1102]*y[IDX_SiI];
    data[713] = 0.0 - k[1106]*y[IDX_SiI];
    data[714] = 0.0 - k[91]*y[IDX_SiI];
    data[715] = 0.0 - k[861]*y[IDX_SiI];
    data[716] = 0.0 + k[573]*y[IDX_SiHII];
    data[717] = 0.0 + k[1083]*y[IDX_SiCI] - k[1213]*y[IDX_SiI];
    data[718] = 0.0 + k[349]*y[IDX_SiCII] + k[352]*y[IDX_SiHII] + k[353]*y[IDX_SiH2II] +
        k[354]*y[IDX_SiH2II] + k[362]*y[IDX_SiOII] + k[363]*y[IDX_SiOHII] +
        k[1222]*y[IDX_SiII];
    data[719] = 0.0 - k[1104]*y[IDX_SiI];
    data[720] = 0.0 + k[1327] + k[1328] + k[1329] + k[1330];
    data[721] = 0.0 + k[1004]*y[IDX_CI];
    data[722] = 0.0 + k[329]*y[IDX_eM] + k[428]*y[IDX_CH2I] + k[465]*y[IDX_CHI] +
        k[634]*y[IDX_H2COI] + k[786]*y[IDX_NH2I];
    data[723] = 0.0 - k[650]*y[IDX_HNCI];
    data[724] = 0.0 - k[651]*y[IDX_HNCI];
    data[725] = 0.0 - k[1]*y[IDX_HII] - k[262] - k[407]*y[IDX_CHII] -
        k[559]*y[IDX_H2OII] - k[589]*y[IDX_H3II] - k[609]*y[IDX_H3OII] -
        k[626]*y[IDX_HCNII] - k[646]*y[IDX_H2COII] - k[647]*y[IDX_H3COII] -
        k[648]*y[IDX_HCOII] - k[649]*y[IDX_HNOII] - k[650]*y[IDX_N2HII] -
        k[651]*y[IDX_O2HII] - k[683]*y[IDX_HeII] - k[684]*y[IDX_HeII] -
        k[685]*y[IDX_HeII] - k[760]*y[IDX_NHII] - k[775]*y[IDX_NH2II] -
        k[844]*y[IDX_OHII] - k[979]*y[IDX_HI] - k[1151] - k[1305];
    data[726] = 0.0 - k[649]*y[IDX_HNCI];
    data[727] = 0.0 - k[647]*y[IDX_HNCI];
    data[728] = 0.0 - k[626]*y[IDX_HNCI];
    data[729] = 0.0 - k[775]*y[IDX_HNCI];
    data[730] = 0.0 - k[760]*y[IDX_HNCI];
    data[731] = 0.0 + k[786]*y[IDX_HCNHII] + k[867]*y[IDX_CI];
    data[732] = 0.0 - k[559]*y[IDX_HNCI];
    data[733] = 0.0 - k[609]*y[IDX_HNCI];
    data[734] = 0.0 - k[407]*y[IDX_HNCI];
    data[735] = 0.0 + k[428]*y[IDX_HCNHII] + k[1006]*y[IDX_NI];
    data[736] = 0.0 - k[646]*y[IDX_HNCI];
    data[737] = 0.0 - k[844]*y[IDX_HNCI];
    data[738] = 0.0 + k[634]*y[IDX_HCNHII];
    data[739] = 0.0 - k[683]*y[IDX_HNCI] - k[684]*y[IDX_HNCI] - k[685]*y[IDX_HNCI];
    data[740] = 0.0 + k[465]*y[IDX_HCNHII];
    data[741] = 0.0 - k[589]*y[IDX_HNCI];
    data[742] = 0.0 + k[1006]*y[IDX_CH2I];
    data[743] = 0.0 + k[867]*y[IDX_NH2I] + k[1004]*y[IDX_HNCOI];
    data[744] = 0.0 - k[1]*y[IDX_HNCI];
    data[745] = 0.0 - k[648]*y[IDX_HNCI];
    data[746] = 0.0 + k[329]*y[IDX_HCNHII];
    data[747] = 0.0 - k[979]*y[IDX_HNCI];
    data[748] = 0.0 + k[807]*y[IDX_NOI];
    data[749] = 0.0 - k[649]*y[IDX_HNOII];
    data[750] = 0.0 - k[204]*y[IDX_NOI] - k[334]*y[IDX_eM] - k[388]*y[IDX_CI] -
        k[430]*y[IDX_CH2I] - k[467]*y[IDX_CHI] - k[482]*y[IDX_CNI] -
        k[486]*y[IDX_COI] - k[550]*y[IDX_H2COI] - k[568]*y[IDX_H2OI] -
        k[630]*y[IDX_HCNI] - k[642]*y[IDX_HCOI] - k[649]*y[IDX_HNCI] -
        k[652]*y[IDX_CO2I] - k[732]*y[IDX_N2I] - k[788]*y[IDX_NH2I] -
        k[800]*y[IDX_NHI] - k[856]*y[IDX_OHI] - k[1295];
    data[751] = 0.0 + k[524]*y[IDX_NOI];
    data[752] = 0.0 - k[652]*y[IDX_HNOII] + k[749]*y[IDX_NHII];
    data[753] = 0.0 + k[778]*y[IDX_O2I] + k[827]*y[IDX_OI];
    data[754] = 0.0 + k[749]*y[IDX_CO2I] + k[755]*y[IDX_H2OI];
    data[755] = 0.0 + k[804]*y[IDX_NHI];
    data[756] = 0.0 - k[788]*y[IDX_HNOII];
    data[757] = 0.0 + k[738]*y[IDX_NI];
    data[758] = 0.0 + k[828]*y[IDX_OI];
    data[759] = 0.0 - k[732]*y[IDX_HNOII];
    data[760] = 0.0 - k[630]*y[IDX_HNOII];
    data[761] = 0.0 - k[430]*y[IDX_HNOII];
    data[762] = 0.0 - k[800]*y[IDX_HNOII] + k[804]*y[IDX_O2II];
    data[763] = 0.0 + k[846]*y[IDX_NOI];
    data[764] = 0.0 - k[482]*y[IDX_HNOII];
    data[765] = 0.0 - k[550]*y[IDX_HNOII];
    data[766] = 0.0 - k[642]*y[IDX_HNOII];
    data[767] = 0.0 - k[467]*y[IDX_HNOII];
    data[768] = 0.0 + k[596]*y[IDX_NOI];
    data[769] = 0.0 - k[204]*y[IDX_HNOII] + k[524]*y[IDX_H2II] + k[596]*y[IDX_H3II] +
        k[807]*y[IDX_O2HII] + k[846]*y[IDX_OHII];
    data[770] = 0.0 + k[738]*y[IDX_H2OII];
    data[771] = 0.0 - k[856]*y[IDX_HNOII];
    data[772] = 0.0 + k[778]*y[IDX_NH2II];
    data[773] = 0.0 - k[388]*y[IDX_HNOII];
    data[774] = 0.0 - k[568]*y[IDX_HNOII] + k[755]*y[IDX_NHII];
    data[775] = 0.0 + k[827]*y[IDX_NH2II] + k[828]*y[IDX_NH3II];
    data[776] = 0.0 - k[334]*y[IDX_HNOII];
    data[777] = 0.0 - k[486]*y[IDX_HNOII];
    data[778] = 0.0 - k[150]*y[IDX_N2II];
    data[779] = 0.0 + k[737]*y[IDX_NI];
    data[780] = 0.0 - k[29]*y[IDX_CI] - k[41]*y[IDX_CH2I] - k[58]*y[IDX_CHI] -
        k[69]*y[IDX_CNI] - k[74]*y[IDX_COI] - k[125]*y[IDX_H2OI] -
        k[135]*y[IDX_HCNI] - k[150]*y[IDX_MgI] - k[170]*y[IDX_H2COI] -
        k[171]*y[IDX_HCOI] - k[172]*y[IDX_NOI] - k[173]*y[IDX_O2I] -
        k[174]*y[IDX_NI] - k[186]*y[IDX_NH2I] - k[197]*y[IDX_NH3I] -
        k[201]*y[IDX_NHI] - k[218]*y[IDX_OI] - k[227]*y[IDX_OHI] -
        k[337]*y[IDX_eM] - k[455]*y[IDX_CH4I] - k[456]*y[IDX_CH4I] -
        k[539]*y[IDX_H2I] - k[569]*y[IDX_H2OI] - k[730]*y[IDX_H2COI] -
        k[731]*y[IDX_HCOI] - k[825]*y[IDX_OI] - k[1271];
    data[781] = 0.0 - k[455]*y[IDX_N2II] - k[456]*y[IDX_N2II];
    data[782] = 0.0 - k[197]*y[IDX_N2II];
    data[783] = 0.0 + k[726]*y[IDX_NHI] + k[727]*y[IDX_NOI] + k[1210]*y[IDX_NI];
    data[784] = 0.0 + k[740]*y[IDX_NI];
    data[785] = 0.0 - k[186]*y[IDX_N2II];
    data[786] = 0.0 + k[144]*y[IDX_HeII];
    data[787] = 0.0 - k[135]*y[IDX_N2II];
    data[788] = 0.0 - k[41]*y[IDX_N2II];
    data[789] = 0.0 - k[201]*y[IDX_N2II] + k[726]*y[IDX_NII];
    data[790] = 0.0 - k[69]*y[IDX_N2II];
    data[791] = 0.0 - k[170]*y[IDX_N2II] - k[730]*y[IDX_N2II];
    data[792] = 0.0 - k[171]*y[IDX_N2II] - k[731]*y[IDX_N2II];
    data[793] = 0.0 + k[144]*y[IDX_N2I];
    data[794] = 0.0 - k[58]*y[IDX_N2II];
    data[795] = 0.0 - k[172]*y[IDX_N2II] + k[727]*y[IDX_NII];
    data[796] = 0.0 - k[174]*y[IDX_N2II] + k[737]*y[IDX_CNII] + k[740]*y[IDX_NHII] +
        k[1210]*y[IDX_NII];
    data[797] = 0.0 - k[227]*y[IDX_N2II];
    data[798] = 0.0 - k[173]*y[IDX_N2II];
    data[799] = 0.0 - k[29]*y[IDX_N2II];
    data[800] = 0.0 - k[125]*y[IDX_N2II] - k[569]*y[IDX_N2II];
    data[801] = 0.0 - k[218]*y[IDX_N2II] - k[825]*y[IDX_N2II];
    data[802] = 0.0 - k[337]*y[IDX_N2II];
    data[803] = 0.0 - k[74]*y[IDX_N2II];
    data[804] = 0.0 - k[539]*y[IDX_N2II];
    data[805] = 0.0 + k[365]*y[IDX_CII] + k[397]*y[IDX_CHII] + k[439]*y[IDX_CH3II] +
        k[493]*y[IDX_HII] + k[713]*y[IDX_NII] + k[809]*y[IDX_OII] +
        k[820]*y[IDX_O2II] + k[1120];
    data[806] = 0.0 + k[449]*y[IDX_H2COI];
    data[807] = 0.0 + k[633]*y[IDX_H2COI] + k[634]*y[IDX_H2COI];
    data[808] = 0.0 + k[735]*y[IDX_H2COI];
    data[809] = 0.0 + k[552]*y[IDX_H2COI];
    data[810] = 0.0 - k[647]*y[IDX_H3COII];
    data[811] = 0.0 + k[550]*y[IDX_H2COI];
    data[812] = 0.0 - k[317]*y[IDX_eM] - k[318]*y[IDX_eM] - k[319]*y[IDX_eM] -
        k[320]*y[IDX_eM] - k[321]*y[IDX_eM] - k[461]*y[IDX_CHI] -
        k[564]*y[IDX_H2OI] - k[628]*y[IDX_HCNI] - k[647]*y[IDX_HNCI] -
        k[782]*y[IDX_NH2I] - k[1249];
    data[813] = 0.0 + k[452]*y[IDX_H2COII];
    data[814] = 0.0 + k[713]*y[IDX_CH3OHI];
    data[815] = 0.0 + k[809]*y[IDX_CH3OHI];
    data[816] = 0.0 + k[622]*y[IDX_H2COI];
    data[817] = 0.0 + k[769]*y[IDX_H2COI];
    data[818] = 0.0 + k[752]*y[IDX_H2COI];
    data[819] = 0.0 + k[820]*y[IDX_CH3OHI];
    data[820] = 0.0 + k[439]*y[IDX_CH3OHI] + k[442]*y[IDX_O2I];
    data[821] = 0.0 - k[782]*y[IDX_H3COII];
    data[822] = 0.0 + k[418]*y[IDX_H2OI];
    data[823] = 0.0 + k[554]*y[IDX_H2COI];
    data[824] = 0.0 + k[607]*y[IDX_H2COI];
    data[825] = 0.0 + k[365]*y[IDX_CH3OHI];
    data[826] = 0.0 - k[628]*y[IDX_H3COII];
    data[827] = 0.0 + k[397]*y[IDX_CH3OHI] + k[400]*y[IDX_H2COI];
    data[828] = 0.0 + k[452]*y[IDX_CH4I] + k[548]*y[IDX_H2COI] + k[641]*y[IDX_HCOI] +
        k[796]*y[IDX_NHI];
    data[829] = 0.0 + k[796]*y[IDX_H2COII];
    data[830] = 0.0 + k[839]*y[IDX_H2COI];
    data[831] = 0.0 + k[400]*y[IDX_CHII] + k[449]*y[IDX_CH4II] + k[548]*y[IDX_H2COII] +
        k[550]*y[IDX_HNOII] + k[552]*y[IDX_O2HII] + k[554]*y[IDX_H2OII] +
        k[585]*y[IDX_H3II] + k[607]*y[IDX_H3OII] + k[622]*y[IDX_HCNII] +
        k[633]*y[IDX_HCNHII] + k[634]*y[IDX_HCNHII] + k[635]*y[IDX_HCOII] +
        k[735]*y[IDX_N2HII] + k[752]*y[IDX_NHII] + k[769]*y[IDX_NH2II] +
        k[839]*y[IDX_OHII];
    data[832] = 0.0 + k[641]*y[IDX_H2COII];
    data[833] = 0.0 - k[461]*y[IDX_H3COII];
    data[834] = 0.0 + k[585]*y[IDX_H2COI];
    data[835] = 0.0 + k[442]*y[IDX_CH3II];
    data[836] = 0.0 + k[493]*y[IDX_CH3OHI];
    data[837] = 0.0 + k[635]*y[IDX_H2COI];
    data[838] = 0.0 + k[418]*y[IDX_CH2II] - k[564]*y[IDX_H3COII];
    data[839] = 0.0 - k[317]*y[IDX_H3COII] - k[318]*y[IDX_H3COII] - k[319]*y[IDX_H3COII]
        - k[320]*y[IDX_H3COII] - k[321]*y[IDX_H3COII];
    data[840] = 0.0 + k[1307] + k[1308] + k[1309] + k[1310];
    data[841] = 0.0 + k[914]*y[IDX_CH3I];
    data[842] = 0.0 + k[446]*y[IDX_CH3II];
    data[843] = 0.0 + k[906]*y[IDX_CH3I];
    data[844] = 0.0 + k[439]*y[IDX_CH3II];
    data[845] = 0.0 + k[49]*y[IDX_H2COI] + k[50]*y[IDX_NH3I] + k[51]*y[IDX_O2I];
    data[846] = 0.0 - k[455]*y[IDX_CH4I] - k[456]*y[IDX_CH4I];
    data[847] = 0.0 - k[52]*y[IDX_COII] - k[77]*y[IDX_HII] - k[101]*y[IDX_H2II] -
        k[140]*y[IDX_HeII] - k[156]*y[IDX_NII] - k[207]*y[IDX_OII] - k[249] -
        k[451]*y[IDX_COII] - k[452]*y[IDX_H2COII] - k[453]*y[IDX_H2OII] -
        k[454]*y[IDX_HCNII] - k[455]*y[IDX_N2II] - k[456]*y[IDX_N2II] -
        k[457]*y[IDX_OHII] - k[495]*y[IDX_HII] - k[511]*y[IDX_H2II] -
        k[658]*y[IDX_HeII] - k[659]*y[IDX_HeII] - k[660]*y[IDX_HeII] -
        k[661]*y[IDX_HeII] - k[716]*y[IDX_NII] - k[717]*y[IDX_NII] -
        k[718]*y[IDX_NII] - k[810]*y[IDX_OII] - k[879]*y[IDX_CH2I] -
        k[920]*y[IDX_CNI] - k[921]*y[IDX_O2I] - k[922]*y[IDX_OHI] -
        k[969]*y[IDX_HI] - k[1028]*y[IDX_NH2I] - k[1034]*y[IDX_NHI] -
        k[1056]*y[IDX_OI] - k[1124] - k[1125] - k[1126] - k[1127] - k[1260];
    data[848] = 0.0 - k[52]*y[IDX_CH4I] - k[451]*y[IDX_CH4I];
    data[849] = 0.0 - k[101]*y[IDX_CH4I] - k[511]*y[IDX_CH4I];
    data[850] = 0.0 + k[50]*y[IDX_CH4II] + k[908]*y[IDX_CH3I];
    data[851] = 0.0 + k[901]*y[IDX_CH3I] + k[901]*y[IDX_CH3I] + k[903]*y[IDX_H2COI] +
        k[904]*y[IDX_H2OI] + k[905]*y[IDX_HCOI] + k[906]*y[IDX_HNOI] +
        k[907]*y[IDX_NH2I] + k[908]*y[IDX_NH3I] + k[914]*y[IDX_O2HI] +
        k[917]*y[IDX_OHI] + k[957]*y[IDX_H2I];
    data[852] = 0.0 - k[156]*y[IDX_CH4I] - k[716]*y[IDX_CH4I] - k[717]*y[IDX_CH4I] -
        k[718]*y[IDX_CH4I];
    data[853] = 0.0 - k[207]*y[IDX_CH4I] - k[810]*y[IDX_CH4I];
    data[854] = 0.0 - k[454]*y[IDX_CH4I];
    data[855] = 0.0 + k[439]*y[IDX_CH3OHI] + k[440]*y[IDX_H2COI] + k[446]*y[IDX_SiH4I];
    data[856] = 0.0 + k[907]*y[IDX_CH3I] - k[1028]*y[IDX_CH4I];
    data[857] = 0.0 - k[453]*y[IDX_CH4I];
    data[858] = 0.0 - k[879]*y[IDX_CH4I];
    data[859] = 0.0 - k[452]*y[IDX_CH4I];
    data[860] = 0.0 - k[1034]*y[IDX_CH4I];
    data[861] = 0.0 - k[457]*y[IDX_CH4I];
    data[862] = 0.0 - k[920]*y[IDX_CH4I];
    data[863] = 0.0 + k[49]*y[IDX_CH4II] + k[440]*y[IDX_CH3II] + k[903]*y[IDX_CH3I];
    data[864] = 0.0 + k[905]*y[IDX_CH3I];
    data[865] = 0.0 - k[140]*y[IDX_CH4I] - k[658]*y[IDX_CH4I] - k[659]*y[IDX_CH4I] -
        k[660]*y[IDX_CH4I] - k[661]*y[IDX_CH4I];
    data[866] = 0.0 + k[917]*y[IDX_CH3I] - k[922]*y[IDX_CH4I];
    data[867] = 0.0 + k[51]*y[IDX_CH4II] - k[921]*y[IDX_CH4I];
    data[868] = 0.0 - k[77]*y[IDX_CH4I] - k[495]*y[IDX_CH4I];
    data[869] = 0.0 + k[904]*y[IDX_CH3I];
    data[870] = 0.0 - k[1056]*y[IDX_CH4I];
    data[871] = 0.0 + k[957]*y[IDX_CH3I];
    data[872] = 0.0 - k[969]*y[IDX_CH4I];
    data[873] = 0.0 + k[378]*y[IDX_CII];
    data[874] = 0.0 + k[63]*y[IDX_COI];
    data[875] = 0.0 + k[74]*y[IDX_COI];
    data[876] = 0.0 - k[52]*y[IDX_COII] - k[451]*y[IDX_COII];
    data[877] = 0.0 - k[28]*y[IDX_CI] - k[38]*y[IDX_CH2I] - k[52]*y[IDX_CH4I] -
        k[54]*y[IDX_CHI] - k[70]*y[IDX_H2COI] - k[71]*y[IDX_HCOI] -
        k[72]*y[IDX_NOI] - k[73]*y[IDX_O2I] - k[123]*y[IDX_H2OI] -
        k[127]*y[IDX_HI] - k[134]*y[IDX_HCNI] - k[184]*y[IDX_NH2I] -
        k[193]*y[IDX_NH3I] - k[200]*y[IDX_NHI] - k[217]*y[IDX_OI] -
        k[226]*y[IDX_OHI] - k[304]*y[IDX_eM] - k[422]*y[IDX_CH2I] -
        k[451]*y[IDX_CH4I] - k[458]*y[IDX_CHI] - k[484]*y[IDX_H2COI] -
        k[532]*y[IDX_H2I] - k[533]*y[IDX_H2I] - k[562]*y[IDX_H2OI] -
        k[779]*y[IDX_NH2I] - k[792]*y[IDX_NH3I] - k[795]*y[IDX_NHI] -
        k[851]*y[IDX_OHI] - k[1131] - k[1275];
    data[878] = 0.0 + k[104]*y[IDX_COI];
    data[879] = 0.0 - k[193]*y[IDX_COII] - k[792]*y[IDX_COII];
    data[880] = 0.0 + k[367]*y[IDX_CII] + k[665]*y[IDX_HeII] + k[719]*y[IDX_NII];
    data[881] = 0.0 + k[158]*y[IDX_COI] + k[719]*y[IDX_CO2I];
    data[882] = 0.0 + k[208]*y[IDX_COI] + k[472]*y[IDX_CHI] + k[1195]*y[IDX_CI];
    data[883] = 0.0 + k[391]*y[IDX_CI];
    data[884] = 0.0 - k[184]*y[IDX_COII] - k[779]*y[IDX_COII];
    data[885] = 0.0 + k[367]*y[IDX_CO2I] + k[376]*y[IDX_O2I] + k[378]*y[IDX_OCNI] +
        k[379]*y[IDX_OHI] + k[1193]*y[IDX_OI];
    data[886] = 0.0 - k[134]*y[IDX_COII];
    data[887] = 0.0 + k[411]*y[IDX_O2I] + k[414]*y[IDX_OI] + k[415]*y[IDX_OHI];
    data[888] = 0.0 - k[38]*y[IDX_COII] - k[422]*y[IDX_COII];
    data[889] = 0.0 - k[200]*y[IDX_COII] - k[795]*y[IDX_COII];
    data[890] = 0.0 - k[70]*y[IDX_COII] - k[484]*y[IDX_COII] + k[497]*y[IDX_HII] +
        k[670]*y[IDX_HeII];
    data[891] = 0.0 - k[71]*y[IDX_COII] + k[500]*y[IDX_HII] + k[680]*y[IDX_HeII];
    data[892] = 0.0 + k[665]*y[IDX_CO2I] + k[670]*y[IDX_H2COI] + k[680]*y[IDX_HCOI];
    data[893] = 0.0 - k[54]*y[IDX_COII] - k[458]*y[IDX_COII] + k[472]*y[IDX_OII];
    data[894] = 0.0 - k[72]*y[IDX_COII];
    data[895] = 0.0 - k[226]*y[IDX_COII] + k[379]*y[IDX_CII] + k[415]*y[IDX_CHII] -
        k[851]*y[IDX_COII];
    data[896] = 0.0 - k[73]*y[IDX_COII] + k[376]*y[IDX_CII] + k[411]*y[IDX_CHII];
    data[897] = 0.0 - k[28]*y[IDX_COII] + k[391]*y[IDX_O2II] + k[1195]*y[IDX_OII];
    data[898] = 0.0 + k[497]*y[IDX_H2COI] + k[500]*y[IDX_HCOI];
    data[899] = 0.0 + k[1148];
    data[900] = 0.0 - k[123]*y[IDX_COII] - k[562]*y[IDX_COII];
    data[901] = 0.0 - k[217]*y[IDX_COII] + k[414]*y[IDX_CHII] + k[1193]*y[IDX_CII];
    data[902] = 0.0 - k[304]*y[IDX_COII];
    data[903] = 0.0 + k[63]*y[IDX_CNII] + k[74]*y[IDX_N2II] + k[104]*y[IDX_H2II] +
        k[158]*y[IDX_NII] + k[208]*y[IDX_OII] + k[232];
    data[904] = 0.0 - k[532]*y[IDX_COII] - k[533]*y[IDX_COII];
    data[905] = 0.0 - k[127]*y[IDX_COII];
    data[906] = 0.0 + k[618]*y[IDX_HI];
    data[907] = 0.0 - k[101]*y[IDX_H2II] - k[511]*y[IDX_H2II];
    data[908] = 0.0 - k[100]*y[IDX_CH2I] - k[101]*y[IDX_CH4I] - k[102]*y[IDX_CHI] -
        k[103]*y[IDX_CNI] - k[104]*y[IDX_COI] - k[105]*y[IDX_H2COI] -
        k[106]*y[IDX_H2OI] - k[107]*y[IDX_HCNI] - k[108]*y[IDX_HCOI] -
        k[109]*y[IDX_NH2I] - k[110]*y[IDX_NH3I] - k[111]*y[IDX_NHI] -
        k[112]*y[IDX_NOI] - k[113]*y[IDX_O2I] - k[114]*y[IDX_OHI] -
        k[128]*y[IDX_HI] - k[305]*y[IDX_eM] - k[509]*y[IDX_CI] -
        k[510]*y[IDX_CH2I] - k[511]*y[IDX_CH4I] - k[512]*y[IDX_CHI] -
        k[513]*y[IDX_CNI] - k[514]*y[IDX_CO2I] - k[515]*y[IDX_COI] -
        k[516]*y[IDX_H2I] - k[517]*y[IDX_H2COI] - k[518]*y[IDX_H2OI] -
        k[519]*y[IDX_HCOI] - k[520]*y[IDX_HeI] - k[521]*y[IDX_N2I] -
        k[522]*y[IDX_NI] - k[523]*y[IDX_NHI] - k[524]*y[IDX_NOI] -
        k[525]*y[IDX_O2I] - k[526]*y[IDX_OI] - k[527]*y[IDX_OHI] - k[1134];
    data[909] = 0.0 - k[110]*y[IDX_H2II];
    data[910] = 0.0 - k[514]*y[IDX_H2II];
    data[911] = 0.0 - k[109]*y[IDX_H2II];
    data[912] = 0.0 - k[521]*y[IDX_H2II];
    data[913] = 0.0 - k[107]*y[IDX_H2II];
    data[914] = 0.0 - k[100]*y[IDX_H2II] - k[510]*y[IDX_H2II];
    data[915] = 0.0 - k[111]*y[IDX_H2II] - k[523]*y[IDX_H2II];
    data[916] = 0.0 - k[103]*y[IDX_H2II] - k[513]*y[IDX_H2II];
    data[917] = 0.0 - k[105]*y[IDX_H2II] - k[517]*y[IDX_H2II];
    data[918] = 0.0 - k[108]*y[IDX_H2II] + k[501]*y[IDX_HII] - k[519]*y[IDX_H2II];
    data[919] = 0.0 + k[115]*y[IDX_H2I];
    data[920] = 0.0 - k[102]*y[IDX_H2II] - k[512]*y[IDX_H2II];
    data[921] = 0.0 + k[1145];
    data[922] = 0.0 - k[520]*y[IDX_H2II];
    data[923] = 0.0 - k[112]*y[IDX_H2II] - k[524]*y[IDX_H2II];
    data[924] = 0.0 - k[522]*y[IDX_H2II];
    data[925] = 0.0 - k[114]*y[IDX_H2II] - k[527]*y[IDX_H2II];
    data[926] = 0.0 - k[113]*y[IDX_H2II] - k[525]*y[IDX_H2II];
    data[927] = 0.0 - k[509]*y[IDX_H2II];
    data[928] = 0.0 + k[501]*y[IDX_HCOI] + k[1197]*y[IDX_HI];
    data[929] = 0.0 - k[106]*y[IDX_H2II] - k[518]*y[IDX_H2II];
    data[930] = 0.0 - k[526]*y[IDX_H2II];
    data[931] = 0.0 - k[305]*y[IDX_H2II];
    data[932] = 0.0 - k[104]*y[IDX_H2II] - k[515]*y[IDX_H2II];
    data[933] = 0.0 + k[115]*y[IDX_HeII] + k[234] - k[516]*y[IDX_H2II];
    data[934] = 0.0 - k[128]*y[IDX_H2II] + k[618]*y[IDX_HeHII] + k[1197]*y[IDX_HII];
    data[935] = 0.0 + k[1311] + k[1312] + k[1313] + k[1314];
    data[936] = 0.0 + k[190]*y[IDX_NH3II];
    data[937] = 0.0 - k[50]*y[IDX_NH3I];
    data[938] = 0.0 + k[192]*y[IDX_NH3II];
    data[939] = 0.0 - k[197]*y[IDX_NH3I];
    data[940] = 0.0 + k[1028]*y[IDX_NH2I];
    data[941] = 0.0 - k[193]*y[IDX_NH3I] - k[792]*y[IDX_NH3I];
    data[942] = 0.0 - k[110]*y[IDX_NH3I];
    data[943] = 0.0 - k[19]*y[IDX_CII] - k[33]*y[IDX_CHII] - k[50]*y[IDX_CH4II] -
        k[85]*y[IDX_HII] - k[110]*y[IDX_H2II] - k[145]*y[IDX_HeII] -
        k[165]*y[IDX_NII] - k[177]*y[IDX_NHII] - k[181]*y[IDX_NH2II] -
        k[193]*y[IDX_COII] - k[194]*y[IDX_H2COII] - k[195]*y[IDX_H2OII] -
        k[196]*y[IDX_HCNII] - k[197]*y[IDX_N2II] - k[198]*y[IDX_O2II] -
        k[213]*y[IDX_OII] - k[222]*y[IDX_OHII] - k[271] - k[272] - k[273] -
        k[374]*y[IDX_CII] - k[691]*y[IDX_HeII] - k[692]*y[IDX_HeII] -
        k[724]*y[IDX_NII] - k[725]*y[IDX_NII] - k[792]*y[IDX_COII] -
        k[793]*y[IDX_HCNII] - k[908]*y[IDX_CH3I] - k[984]*y[IDX_HI] -
        k[1033]*y[IDX_CNI] - k[1037]*y[IDX_NHI] - k[1074]*y[IDX_OI] -
        k[1098]*y[IDX_OHI] - k[1159] - k[1160] - k[1161] - k[1267];
    data[944] = 0.0 - k[908]*y[IDX_NH3I];
    data[945] = 0.0 - k[165]*y[IDX_NH3I] - k[724]*y[IDX_NH3I] - k[725]*y[IDX_NH3I];
    data[946] = 0.0 - k[213]*y[IDX_NH3I];
    data[947] = 0.0 - k[196]*y[IDX_NH3I] - k[793]*y[IDX_NH3I];
    data[948] = 0.0 - k[181]*y[IDX_NH3I];
    data[949] = 0.0 - k[177]*y[IDX_NH3I];
    data[950] = 0.0 - k[198]*y[IDX_NH3I];
    data[951] = 0.0 + k[961]*y[IDX_H2I] + k[1028]*y[IDX_CH4I] + k[1032]*y[IDX_OHI];
    data[952] = 0.0 - k[195]*y[IDX_NH3I];
    data[953] = 0.0 + k[189]*y[IDX_HCOI] + k[190]*y[IDX_MgI] + k[191]*y[IDX_NOI] +
        k[192]*y[IDX_SiI];
    data[954] = 0.0 - k[19]*y[IDX_NH3I] - k[374]*y[IDX_NH3I];
    data[955] = 0.0 - k[33]*y[IDX_NH3I];
    data[956] = 0.0 - k[194]*y[IDX_NH3I];
    data[957] = 0.0 - k[1037]*y[IDX_NH3I];
    data[958] = 0.0 - k[222]*y[IDX_NH3I];
    data[959] = 0.0 - k[1033]*y[IDX_NH3I];
    data[960] = 0.0 + k[189]*y[IDX_NH3II];
    data[961] = 0.0 - k[145]*y[IDX_NH3I] - k[691]*y[IDX_NH3I] - k[692]*y[IDX_NH3I];
    data[962] = 0.0 + k[191]*y[IDX_NH3II];
    data[963] = 0.0 + k[1032]*y[IDX_NH2I] - k[1098]*y[IDX_NH3I];
    data[964] = 0.0 - k[85]*y[IDX_NH3I];
    data[965] = 0.0 - k[1074]*y[IDX_NH3I];
    data[966] = 0.0 + k[961]*y[IDX_NH2I];
    data[967] = 0.0 - k[984]*y[IDX_NH3I];
    data[968] = 0.0 - k[914]*y[IDX_CH3I];
    data[969] = 0.0 - k[909]*y[IDX_CH3I];
    data[970] = 0.0 + k[883]*y[IDX_CH2I] - k[906]*y[IDX_CH3I];
    data[971] = 0.0 + k[248] + k[656]*y[IDX_HeII] + k[714]*y[IDX_NII] +
        k[860]*y[IDX_SiII] + k[1121];
    data[972] = 0.0 + k[47]*y[IDX_CH3II];
    data[973] = 0.0 + k[302]*y[IDX_eM] + k[447]*y[IDX_CO2I] + k[448]*y[IDX_COI] +
        k[449]*y[IDX_H2COI] + k[450]*y[IDX_H2OI];
    data[974] = 0.0 + k[860]*y[IDX_CH3OHI];
    data[975] = 0.0 + k[451]*y[IDX_COII] + k[452]*y[IDX_H2COII] + k[453]*y[IDX_H2OII] +
        k[454]*y[IDX_HCNII] + k[661]*y[IDX_HeII] + k[879]*y[IDX_CH2I] +
        k[879]*y[IDX_CH2I] + k[920]*y[IDX_CNI] + k[921]*y[IDX_O2I] +
        k[922]*y[IDX_OHI] + k[969]*y[IDX_HI] + k[1028]*y[IDX_NH2I] +
        k[1034]*y[IDX_NHI] + k[1056]*y[IDX_OI] + k[1125];
    data[976] = 0.0 + k[451]*y[IDX_CH4I];
    data[977] = 0.0 - k[908]*y[IDX_CH3I];
    data[978] = 0.0 - k[76]*y[IDX_HII] - k[244] - k[245] - k[246] - k[578]*y[IDX_H3II] -
        k[655]*y[IDX_HeII] - k[901]*y[IDX_CH3I] - k[901]*y[IDX_CH3I] -
        k[901]*y[IDX_CH3I] - k[901]*y[IDX_CH3I] - k[902]*y[IDX_CNI] -
        k[903]*y[IDX_H2COI] - k[904]*y[IDX_H2OI] - k[905]*y[IDX_HCOI] -
        k[906]*y[IDX_HNOI] - k[907]*y[IDX_NH2I] - k[908]*y[IDX_NH3I] -
        k[909]*y[IDX_NO2I] - k[910]*y[IDX_NOI] - k[911]*y[IDX_O2I] -
        k[912]*y[IDX_O2I] - k[913]*y[IDX_O2I] - k[914]*y[IDX_O2HI] -
        k[915]*y[IDX_OI] - k[916]*y[IDX_OI] - k[917]*y[IDX_OHI] -
        k[918]*y[IDX_OHI] - k[919]*y[IDX_OHI] - k[957]*y[IDX_H2I] -
        k[968]*y[IDX_HI] - k[1008]*y[IDX_NI] - k[1009]*y[IDX_NI] -
        k[1010]*y[IDX_NI] - k[1116] - k[1117] - k[1118] - k[1259];
    data[979] = 0.0 + k[447]*y[IDX_CH4II];
    data[980] = 0.0 + k[714]*y[IDX_CH3OHI];
    data[981] = 0.0 + k[454]*y[IDX_CH4I];
    data[982] = 0.0 + k[46]*y[IDX_HCOI] + k[47]*y[IDX_MgI] + k[48]*y[IDX_NOI] +
        k[1215]*y[IDX_eM];
    data[983] = 0.0 - k[907]*y[IDX_CH3I] + k[1028]*y[IDX_CH4I];
    data[984] = 0.0 + k[417]*y[IDX_H2COI];
    data[985] = 0.0 + k[453]*y[IDX_CH4I];
    data[986] = 0.0 + k[878]*y[IDX_CH2I] + k[878]*y[IDX_CH2I] + k[879]*y[IDX_CH4I] +
        k[879]*y[IDX_CH4I] + k[881]*y[IDX_H2COI] + k[882]*y[IDX_HCOI] +
        k[883]*y[IDX_HNOI] + k[900]*y[IDX_OHI] + k[956]*y[IDX_H2I];
    data[987] = 0.0 + k[452]*y[IDX_CH4I];
    data[988] = 0.0 + k[1034]*y[IDX_CH4I];
    data[989] = 0.0 - k[902]*y[IDX_CH3I] + k[920]*y[IDX_CH4I];
    data[990] = 0.0 + k[417]*y[IDX_CH2II] + k[449]*y[IDX_CH4II] + k[881]*y[IDX_CH2I] -
        k[903]*y[IDX_CH3I];
    data[991] = 0.0 + k[46]*y[IDX_CH3II] + k[882]*y[IDX_CH2I] - k[905]*y[IDX_CH3I];
    data[992] = 0.0 - k[655]*y[IDX_CH3I] + k[656]*y[IDX_CH3OHI] + k[661]*y[IDX_CH4I];
    data[993] = 0.0 + k[1201]*y[IDX_H2I];
    data[994] = 0.0 - k[578]*y[IDX_CH3I];
    data[995] = 0.0 + k[48]*y[IDX_CH3II] - k[910]*y[IDX_CH3I];
    data[996] = 0.0 - k[1008]*y[IDX_CH3I] - k[1009]*y[IDX_CH3I] - k[1010]*y[IDX_CH3I];
    data[997] = 0.0 + k[900]*y[IDX_CH2I] - k[917]*y[IDX_CH3I] - k[918]*y[IDX_CH3I] -
        k[919]*y[IDX_CH3I] + k[922]*y[IDX_CH4I];
    data[998] = 0.0 - k[911]*y[IDX_CH3I] - k[912]*y[IDX_CH3I] - k[913]*y[IDX_CH3I] +
        k[921]*y[IDX_CH4I];
    data[999] = 0.0 - k[76]*y[IDX_CH3I];
    data[1000] = 0.0 + k[450]*y[IDX_CH4II] - k[904]*y[IDX_CH3I];
    data[1001] = 0.0 - k[915]*y[IDX_CH3I] - k[916]*y[IDX_CH3I] + k[1056]*y[IDX_CH4I];
    data[1002] = 0.0 + k[302]*y[IDX_CH4II] + k[1215]*y[IDX_CH3II];
    data[1003] = 0.0 + k[448]*y[IDX_CH4II];
    data[1004] = 0.0 + k[956]*y[IDX_CH2I] - k[957]*y[IDX_CH3I] + k[1201]*y[IDX_CHI];
    data[1005] = 0.0 - k[968]*y[IDX_CH3I] + k[969]*y[IDX_CH4I];
    data[1006] = 0.0 + k[1383] + k[1384] + k[1385] + k[1386];
    data[1007] = 0.0 + k[954]*y[IDX_COI];
    data[1008] = 0.0 + k[952]*y[IDX_COI];
    data[1009] = 0.0 + k[1053]*y[IDX_NOI] + k[1054]*y[IDX_O2I];
    data[1010] = 0.0 + k[490]*y[IDX_COI];
    data[1011] = 0.0 + k[331]*y[IDX_eM] + k[387]*y[IDX_CI] + k[485]*y[IDX_COI] +
        k[567]*y[IDX_H2OI];
    data[1012] = 0.0 + k[951]*y[IDX_COI];
    data[1013] = 0.0 - k[447]*y[IDX_CO2I];
    data[1014] = 0.0 - k[734]*y[IDX_CO2I];
    data[1015] = 0.0 - k[821]*y[IDX_CO2I];
    data[1016] = 0.0 - k[1103]*y[IDX_CO2I];
    data[1017] = 0.0 - k[652]*y[IDX_CO2I];
    data[1018] = 0.0 - k[514]*y[IDX_CO2I];
    data[1019] = 0.0 - k[252] - k[367]*y[IDX_CII] - k[398]*y[IDX_CHII] -
        k[416]*y[IDX_CH2II] - k[447]*y[IDX_CH4II] - k[496]*y[IDX_HII] -
        k[514]*y[IDX_H2II] - k[582]*y[IDX_H3II] - k[620]*y[IDX_HCNII] -
        k[652]*y[IDX_HNOII] - k[665]*y[IDX_HeII] - k[666]*y[IDX_HeII] -
        k[667]*y[IDX_HeII] - k[668]*y[IDX_HeII] - k[719]*y[IDX_NII] -
        k[734]*y[IDX_N2HII] - k[748]*y[IDX_NHII] - k[749]*y[IDX_NHII] -
        k[750]*y[IDX_NHII] - k[812]*y[IDX_OII] - k[821]*y[IDX_O2HII] -
        k[837]*y[IDX_OHII] - k[923]*y[IDX_CHI] - k[971]*y[IDX_HI] -
        k[1012]*y[IDX_NI] - k[1059]*y[IDX_OI] - k[1103]*y[IDX_SiI] - k[1132] -
        k[1258];
    data[1020] = 0.0 - k[719]*y[IDX_CO2I];
    data[1021] = 0.0 - k[812]*y[IDX_CO2I];
    data[1022] = 0.0 - k[620]*y[IDX_CO2I];
    data[1023] = 0.0 - k[748]*y[IDX_CO2I] - k[749]*y[IDX_CO2I] - k[750]*y[IDX_CO2I];
    data[1024] = 0.0 - k[416]*y[IDX_CO2I];
    data[1025] = 0.0 - k[367]*y[IDX_CO2I];
    data[1026] = 0.0 - k[398]*y[IDX_CO2I];
    data[1027] = 0.0 + k[889]*y[IDX_O2I] + k[890]*y[IDX_O2I];
    data[1028] = 0.0 - k[837]*y[IDX_CO2I];
    data[1029] = 0.0 + k[1001]*y[IDX_O2I] + k[1066]*y[IDX_OI];
    data[1030] = 0.0 - k[665]*y[IDX_CO2I] - k[666]*y[IDX_CO2I] - k[667]*y[IDX_CO2I] -
        k[668]*y[IDX_CO2I];
    data[1031] = 0.0 - k[923]*y[IDX_CO2I] + k[933]*y[IDX_O2I];
    data[1032] = 0.0 - k[582]*y[IDX_CO2I];
    data[1033] = 0.0 + k[1053]*y[IDX_OCNI];
    data[1034] = 0.0 - k[1012]*y[IDX_CO2I];
    data[1035] = 0.0 + k[1092]*y[IDX_COI];
    data[1036] = 0.0 + k[889]*y[IDX_CH2I] + k[890]*y[IDX_CH2I] + k[933]*y[IDX_CHI] +
        k[953]*y[IDX_COI] + k[1001]*y[IDX_HCOI] + k[1054]*y[IDX_OCNI];
    data[1037] = 0.0 + k[387]*y[IDX_HCO2II];
    data[1038] = 0.0 - k[496]*y[IDX_CO2I];
    data[1039] = 0.0 + k[567]*y[IDX_HCO2II];
    data[1040] = 0.0 - k[1059]*y[IDX_CO2I] + k[1066]*y[IDX_HCOI];
    data[1041] = 0.0 + k[331]*y[IDX_HCO2II];
    data[1042] = 0.0 + k[485]*y[IDX_HCO2II] + k[490]*y[IDX_SiOII] + k[951]*y[IDX_HNOI] +
        k[952]*y[IDX_NO2I] + k[953]*y[IDX_O2I] + k[954]*y[IDX_O2HI] +
        k[1092]*y[IDX_OHI];
    data[1043] = 0.0 - k[971]*y[IDX_CO2I];
    data[1044] = 0.0 - k[712]*y[IDX_NII] - k[713]*y[IDX_NII] - k[714]*y[IDX_NII] -
        k[715]*y[IDX_NII];
    data[1045] = 0.0 - k[163]*y[IDX_NII];
    data[1046] = 0.0 + k[174]*y[IDX_NI];
    data[1047] = 0.0 - k[156]*y[IDX_NII] - k[716]*y[IDX_NII] - k[717]*y[IDX_NII] -
        k[718]*y[IDX_NII];
    data[1048] = 0.0 - k[165]*y[IDX_NII] - k[724]*y[IDX_NII] - k[725]*y[IDX_NII];
    data[1049] = 0.0 - k[719]*y[IDX_NII];
    data[1050] = 0.0 - k[57]*y[IDX_CHI] - k[155]*y[IDX_CH2I] - k[156]*y[IDX_CH4I] -
        k[157]*y[IDX_CNI] - k[158]*y[IDX_COI] - k[159]*y[IDX_H2COI] -
        k[160]*y[IDX_H2OI] - k[161]*y[IDX_HCNI] - k[162]*y[IDX_HCOI] -
        k[163]*y[IDX_MgI] - k[164]*y[IDX_NH2I] - k[165]*y[IDX_NH3I] -
        k[166]*y[IDX_NHI] - k[167]*y[IDX_NOI] - k[168]*y[IDX_O2I] -
        k[169]*y[IDX_OHI] - k[468]*y[IDX_CHI] - k[538]*y[IDX_H2I] -
        k[712]*y[IDX_CH3OHI] - k[713]*y[IDX_CH3OHI] - k[714]*y[IDX_CH3OHI] -
        k[715]*y[IDX_CH3OHI] - k[716]*y[IDX_CH4I] - k[717]*y[IDX_CH4I] -
        k[718]*y[IDX_CH4I] - k[719]*y[IDX_CO2I] - k[720]*y[IDX_COI] -
        k[721]*y[IDX_H2COI] - k[722]*y[IDX_H2COI] - k[723]*y[IDX_HCOI] -
        k[724]*y[IDX_NH3I] - k[725]*y[IDX_NH3I] - k[726]*y[IDX_NHI] -
        k[727]*y[IDX_NOI] - k[728]*y[IDX_O2I] - k[729]*y[IDX_O2I] -
        k[1210]*y[IDX_NI] - k[1220]*y[IDX_eM] - k[1268];
    data[1051] = 0.0 - k[164]*y[IDX_NII] + k[689]*y[IDX_HeII];
    data[1052] = 0.0 + k[688]*y[IDX_HeII];
    data[1053] = 0.0 - k[161]*y[IDX_NII] + k[677]*y[IDX_HeII];
    data[1054] = 0.0 - k[155]*y[IDX_NII];
    data[1055] = 0.0 - k[166]*y[IDX_NII] + k[693]*y[IDX_HeII] - k[726]*y[IDX_NII];
    data[1056] = 0.0 - k[157]*y[IDX_NII] + k[663]*y[IDX_HeII];
    data[1057] = 0.0 - k[159]*y[IDX_NII] - k[721]*y[IDX_NII] - k[722]*y[IDX_NII];
    data[1058] = 0.0 - k[162]*y[IDX_NII] - k[723]*y[IDX_NII];
    data[1059] = 0.0 + k[663]*y[IDX_CNI] + k[677]*y[IDX_HCNI] + k[688]*y[IDX_N2I] +
        k[689]*y[IDX_NH2I] + k[693]*y[IDX_NHI] + k[695]*y[IDX_NOI];
    data[1060] = 0.0 - k[57]*y[IDX_NII] - k[468]*y[IDX_NII];
    data[1061] = 0.0 - k[167]*y[IDX_NII] + k[695]*y[IDX_HeII] - k[727]*y[IDX_NII];
    data[1062] = 0.0 + k[174]*y[IDX_N2II] + k[238] + k[268] - k[1210]*y[IDX_NII];
    data[1063] = 0.0 - k[169]*y[IDX_NII];
    data[1064] = 0.0 - k[168]*y[IDX_NII] - k[728]*y[IDX_NII] - k[729]*y[IDX_NII];
    data[1065] = 0.0 - k[160]*y[IDX_NII];
    data[1066] = 0.0 - k[1220]*y[IDX_NII];
    data[1067] = 0.0 - k[158]*y[IDX_NII] - k[720]*y[IDX_NII];
    data[1068] = 0.0 - k[538]*y[IDX_NII];
    data[1069] = 0.0 - k[818]*y[IDX_OII];
    data[1070] = 0.0 + k[698]*y[IDX_HeII];
    data[1071] = 0.0 - k[808]*y[IDX_OII] - k[809]*y[IDX_OII];
    data[1072] = 0.0 + k[711]*y[IDX_HeII];
    data[1073] = 0.0 + k[216]*y[IDX_OI];
    data[1074] = 0.0 + k[218]*y[IDX_OI];
    data[1075] = 0.0 - k[207]*y[IDX_OII] - k[810]*y[IDX_OII];
    data[1076] = 0.0 + k[217]*y[IDX_OI];
    data[1077] = 0.0 - k[213]*y[IDX_OII];
    data[1078] = 0.0 + k[666]*y[IDX_HeII] - k[812]*y[IDX_OII];
    data[1079] = 0.0 + k[729]*y[IDX_O2I];
    data[1080] = 0.0 - k[43]*y[IDX_CH2I] - k[60]*y[IDX_CHI] - k[131]*y[IDX_HI] -
        k[202]*y[IDX_NHI] - k[207]*y[IDX_CH4I] - k[208]*y[IDX_COI] -
        k[209]*y[IDX_H2COI] - k[210]*y[IDX_H2OI] - k[211]*y[IDX_HCOI] -
        k[212]*y[IDX_NH2I] - k[213]*y[IDX_NH3I] - k[214]*y[IDX_O2I] -
        k[215]*y[IDX_OHI] - k[472]*y[IDX_CHI] - k[543]*y[IDX_H2I] -
        k[803]*y[IDX_NHI] - k[808]*y[IDX_CH3OHI] - k[809]*y[IDX_CH3OHI] -
        k[810]*y[IDX_CH4I] - k[811]*y[IDX_CNI] - k[812]*y[IDX_CO2I] -
        k[813]*y[IDX_H2COI] - k[814]*y[IDX_HCNI] - k[815]*y[IDX_HCNI] -
        k[816]*y[IDX_HCOI] - k[817]*y[IDX_N2I] - k[818]*y[IDX_NO2I] -
        k[819]*y[IDX_OHI] - k[1195]*y[IDX_CI] - k[1221]*y[IDX_eM] - k[1269];
    data[1081] = 0.0 + k[1167];
    data[1082] = 0.0 - k[212]*y[IDX_OII];
    data[1083] = 0.0 - k[817]*y[IDX_OII];
    data[1084] = 0.0 + k[377]*y[IDX_O2I];
    data[1085] = 0.0 - k[814]*y[IDX_OII] - k[815]*y[IDX_OII];
    data[1086] = 0.0 + k[413]*y[IDX_O2I];
    data[1087] = 0.0 - k[43]*y[IDX_OII];
    data[1088] = 0.0 - k[202]*y[IDX_OII] - k[803]*y[IDX_OII];
    data[1089] = 0.0 + k[1173];
    data[1090] = 0.0 - k[811]*y[IDX_OII];
    data[1091] = 0.0 - k[209]*y[IDX_OII] - k[813]*y[IDX_OII];
    data[1092] = 0.0 - k[211]*y[IDX_OII] - k[816]*y[IDX_OII];
    data[1093] = 0.0 + k[666]*y[IDX_CO2I] + k[694]*y[IDX_NOI] + k[696]*y[IDX_O2I] +
        k[698]*y[IDX_OCNI] + k[699]*y[IDX_OHI] + k[711]*y[IDX_SiOI];
    data[1094] = 0.0 - k[60]*y[IDX_OII] - k[472]*y[IDX_OII];
    data[1095] = 0.0 + k[694]*y[IDX_HeII];
    data[1096] = 0.0 - k[215]*y[IDX_OII] + k[699]*y[IDX_HeII] - k[819]*y[IDX_OII];
    data[1097] = 0.0 - k[214]*y[IDX_OII] + k[377]*y[IDX_CII] + k[413]*y[IDX_CHII] +
        k[696]*y[IDX_HeII] + k[729]*y[IDX_NII];
    data[1098] = 0.0 - k[1195]*y[IDX_OII];
    data[1099] = 0.0 + k[89]*y[IDX_OI];
    data[1100] = 0.0 - k[210]*y[IDX_OII];
    data[1101] = 0.0 + k[89]*y[IDX_HII] + k[216]*y[IDX_CNII] + k[217]*y[IDX_COII] +
        k[218]*y[IDX_N2II] + k[239] + k[282];
    data[1102] = 0.0 - k[1221]*y[IDX_OII];
    data[1103] = 0.0 - k[208]*y[IDX_OII];
    data[1104] = 0.0 - k[543]*y[IDX_OII];
    data[1105] = 0.0 - k[131]*y[IDX_OII];
    data[1106] = 0.0 + k[65]*y[IDX_HCNI] + k[480]*y[IDX_HCOI] + k[531]*y[IDX_H2I] +
        k[560]*y[IDX_H2OI];
    data[1107] = 0.0 + k[483]*y[IDX_CNI];
    data[1108] = 0.0 - k[626]*y[IDX_HCNII];
    data[1109] = 0.0 + k[482]*y[IDX_CNI];
    data[1110] = 0.0 + k[135]*y[IDX_HCNI];
    data[1111] = 0.0 - k[454]*y[IDX_HCNII] + k[717]*y[IDX_NII];
    data[1112] = 0.0 + k[134]*y[IDX_HCNI];
    data[1113] = 0.0 + k[107]*y[IDX_HCNI] + k[513]*y[IDX_CNI];
    data[1114] = 0.0 - k[196]*y[IDX_HCNII] + k[374]*y[IDX_CII] - k[793]*y[IDX_HCNII];
    data[1115] = 0.0 - k[620]*y[IDX_HCNII];
    data[1116] = 0.0 + k[161]*y[IDX_HCNI] + k[717]*y[IDX_CH4I];
    data[1117] = 0.0 - k[124]*y[IDX_H2OI] - k[129]*y[IDX_HI] - k[132]*y[IDX_NOI] -
        k[133]*y[IDX_O2I] - k[196]*y[IDX_NH3I] - k[326]*y[IDX_eM] -
        k[385]*y[IDX_CI] - k[426]*y[IDX_CH2I] - k[454]*y[IDX_CH4I] -
        k[463]*y[IDX_CHI] - k[535]*y[IDX_H2I] - k[565]*y[IDX_H2OI] -
        k[620]*y[IDX_CO2I] - k[621]*y[IDX_COI] - k[622]*y[IDX_H2COI] -
        k[623]*y[IDX_HCNI] - k[624]*y[IDX_HCOI] - k[625]*y[IDX_HCOI] -
        k[626]*y[IDX_HNCI] - k[784]*y[IDX_NH2I] - k[793]*y[IDX_NH3I] -
        k[798]*y[IDX_NHI] - k[853]*y[IDX_OHI] - k[1282];
    data[1118] = 0.0 + k[747]*y[IDX_CNI];
    data[1119] = 0.0 + k[373]*y[IDX_CII] + k[409]*y[IDX_CHII] - k[784]*y[IDX_HCNII];
    data[1120] = 0.0 + k[736]*y[IDX_NI];
    data[1121] = 0.0 + k[373]*y[IDX_NH2I] + k[374]*y[IDX_NH3I];
    data[1122] = 0.0 + k[65]*y[IDX_CNII] + k[81]*y[IDX_HII] + k[107]*y[IDX_H2II] +
        k[134]*y[IDX_COII] + k[135]*y[IDX_N2II] + k[161]*y[IDX_NII] -
        k[623]*y[IDX_HCNII];
    data[1123] = 0.0 + k[409]*y[IDX_NH2I];
    data[1124] = 0.0 - k[426]*y[IDX_HCNII];
    data[1125] = 0.0 - k[798]*y[IDX_HCNII];
    data[1126] = 0.0 + k[836]*y[IDX_CNI];
    data[1127] = 0.0 + k[482]*y[IDX_HNOII] + k[483]*y[IDX_O2HII] + k[513]*y[IDX_H2II] +
        k[581]*y[IDX_H3II] + k[747]*y[IDX_NHII] + k[836]*y[IDX_OHII];
    data[1128] = 0.0 - k[622]*y[IDX_HCNII];
    data[1129] = 0.0 + k[480]*y[IDX_CNII] - k[624]*y[IDX_HCNII] - k[625]*y[IDX_HCNII];
    data[1130] = 0.0 - k[463]*y[IDX_HCNII];
    data[1131] = 0.0 + k[581]*y[IDX_CNI];
    data[1132] = 0.0 - k[132]*y[IDX_HCNII];
    data[1133] = 0.0 + k[736]*y[IDX_CH2II];
    data[1134] = 0.0 - k[853]*y[IDX_HCNII];
    data[1135] = 0.0 - k[133]*y[IDX_HCNII];
    data[1136] = 0.0 - k[385]*y[IDX_HCNII];
    data[1137] = 0.0 + k[81]*y[IDX_HCNI];
    data[1138] = 0.0 - k[124]*y[IDX_HCNII] + k[560]*y[IDX_CNII] - k[565]*y[IDX_HCNII];
    data[1139] = 0.0 - k[326]*y[IDX_HCNII];
    data[1140] = 0.0 - k[621]*y[IDX_HCNII];
    data[1141] = 0.0 + k[531]*y[IDX_CNII] - k[535]*y[IDX_HCNII];
    data[1142] = 0.0 - k[129]*y[IDX_HCNII];
    data[1143] = 0.0 + k[502]*y[IDX_HII];
    data[1144] = 0.0 + k[183]*y[IDX_NH2I];
    data[1145] = 0.0 + k[801]*y[IDX_NHI];
    data[1146] = 0.0 + k[805]*y[IDX_NHI];
    data[1147] = 0.0 - k[775]*y[IDX_NH2II];
    data[1148] = 0.0 + k[800]*y[IDX_NHI];
    data[1149] = 0.0 + k[186]*y[IDX_NH2I];
    data[1150] = 0.0 + k[184]*y[IDX_NH2I];
    data[1151] = 0.0 + k[109]*y[IDX_NH2I] + k[523]*y[IDX_NHI];
    data[1152] = 0.0 - k[181]*y[IDX_NH2II] + k[692]*y[IDX_HeII] + k[725]*y[IDX_NII];
    data[1153] = 0.0 + k[164]*y[IDX_NH2I] + k[725]*y[IDX_NH3I];
    data[1154] = 0.0 + k[212]*y[IDX_NH2I];
    data[1155] = 0.0 + k[798]*y[IDX_NHI];
    data[1156] = 0.0 - k[42]*y[IDX_CH2I] - k[59]*y[IDX_CHI] - k[180]*y[IDX_HCOI] -
        k[181]*y[IDX_NH3I] - k[182]*y[IDX_NOI] - k[341]*y[IDX_eM] -
        k[342]*y[IDX_eM] - k[433]*y[IDX_CH2I] - k[471]*y[IDX_CHI] -
        k[542]*y[IDX_H2I] - k[741]*y[IDX_NI] - k[769]*y[IDX_H2COI] -
        k[770]*y[IDX_H2COI] - k[771]*y[IDX_H2OI] - k[772]*y[IDX_H2OI] -
        k[773]*y[IDX_HCNI] - k[774]*y[IDX_HCOI] - k[775]*y[IDX_HNCI] -
        k[776]*y[IDX_NH2I] - k[777]*y[IDX_O2I] - k[778]*y[IDX_O2I] -
        k[802]*y[IDX_NHI] - k[827]*y[IDX_OI] - k[1279];
    data[1157] = 0.0 + k[541]*y[IDX_H2I] + k[757]*y[IDX_H2OI] + k[763]*y[IDX_NHI];
    data[1158] = 0.0 + k[187]*y[IDX_NH2I];
    data[1159] = 0.0 + k[84]*y[IDX_HII] + k[109]*y[IDX_H2II] + k[164]*y[IDX_NII] +
        k[183]*y[IDX_CNII] + k[184]*y[IDX_COII] + k[185]*y[IDX_H2OII] +
        k[186]*y[IDX_N2II] + k[187]*y[IDX_O2II] + k[188]*y[IDX_OHII] +
        k[212]*y[IDX_OII] + k[269] - k[776]*y[IDX_NH2II] + k[1157];
    data[1160] = 0.0 + k[185]*y[IDX_NH2I];
    data[1161] = 0.0 - k[773]*y[IDX_NH2II];
    data[1162] = 0.0 - k[42]*y[IDX_NH2II] - k[433]*y[IDX_NH2II];
    data[1163] = 0.0 + k[523]*y[IDX_H2II] + k[594]*y[IDX_H3II] + k[763]*y[IDX_NHII] +
        k[798]*y[IDX_HCNII] + k[799]*y[IDX_HCOII] + k[800]*y[IDX_HNOII] +
        k[801]*y[IDX_N2HII] - k[802]*y[IDX_NH2II] + k[805]*y[IDX_O2HII] +
        k[806]*y[IDX_OHII];
    data[1164] = 0.0 + k[188]*y[IDX_NH2I] + k[806]*y[IDX_NHI];
    data[1165] = 0.0 - k[769]*y[IDX_NH2II] - k[770]*y[IDX_NH2II];
    data[1166] = 0.0 - k[180]*y[IDX_NH2II] - k[774]*y[IDX_NH2II];
    data[1167] = 0.0 + k[692]*y[IDX_NH3I];
    data[1168] = 0.0 - k[59]*y[IDX_NH2II] - k[471]*y[IDX_NH2II];
    data[1169] = 0.0 + k[594]*y[IDX_NHI];
    data[1170] = 0.0 - k[182]*y[IDX_NH2II];
    data[1171] = 0.0 - k[741]*y[IDX_NH2II];
    data[1172] = 0.0 - k[777]*y[IDX_NH2II] - k[778]*y[IDX_NH2II];
    data[1173] = 0.0 + k[84]*y[IDX_NH2I] + k[502]*y[IDX_HNCOI];
    data[1174] = 0.0 + k[799]*y[IDX_NHI];
    data[1175] = 0.0 + k[757]*y[IDX_NHII] - k[771]*y[IDX_NH2II] - k[772]*y[IDX_NH2II];
    data[1176] = 0.0 - k[827]*y[IDX_NH2II];
    data[1177] = 0.0 - k[341]*y[IDX_NH2II] - k[342]*y[IDX_NH2II];
    data[1178] = 0.0 + k[541]*y[IDX_NHII] - k[542]*y[IDX_NH2II];
    data[1179] = 0.0 + k[199]*y[IDX_NHI];
    data[1180] = 0.0 + k[685]*y[IDX_HeII] - k[760]*y[IDX_NHII];
    data[1181] = 0.0 + k[201]*y[IDX_NHI];
    data[1182] = 0.0 + k[200]*y[IDX_NHI];
    data[1183] = 0.0 + k[111]*y[IDX_NHI] + k[522]*y[IDX_NI];
    data[1184] = 0.0 - k[177]*y[IDX_NHII] + k[691]*y[IDX_HeII];
    data[1185] = 0.0 - k[748]*y[IDX_NHII] - k[749]*y[IDX_NHII] - k[750]*y[IDX_NHII];
    data[1186] = 0.0 + k[166]*y[IDX_NHI] + k[538]*y[IDX_H2I] + k[723]*y[IDX_HCOI];
    data[1187] = 0.0 + k[202]*y[IDX_NHI];
    data[1188] = 0.0 - k[175]*y[IDX_H2COI] - k[176]*y[IDX_H2OI] - k[177]*y[IDX_NH3I] -
        k[178]*y[IDX_NOI] - k[179]*y[IDX_O2I] - k[340]*y[IDX_eM] -
        k[390]*y[IDX_CI] - k[432]*y[IDX_CH2I] - k[470]*y[IDX_CHI] -
        k[540]*y[IDX_H2I] - k[541]*y[IDX_H2I] - k[740]*y[IDX_NI] -
        k[747]*y[IDX_CNI] - k[748]*y[IDX_CO2I] - k[749]*y[IDX_CO2I] -
        k[750]*y[IDX_CO2I] - k[751]*y[IDX_COI] - k[752]*y[IDX_H2COI] -
        k[753]*y[IDX_H2COI] - k[754]*y[IDX_H2OI] - k[755]*y[IDX_H2OI] -
        k[756]*y[IDX_H2OI] - k[757]*y[IDX_H2OI] - k[758]*y[IDX_HCNI] -
        k[759]*y[IDX_HCOI] - k[760]*y[IDX_HNCI] - k[761]*y[IDX_N2I] -
        k[762]*y[IDX_NH2I] - k[763]*y[IDX_NHI] - k[764]*y[IDX_NOI] -
        k[765]*y[IDX_O2I] - k[766]*y[IDX_O2I] - k[767]*y[IDX_OI] -
        k[768]*y[IDX_OHI] - k[1156] - k[1273];
    data[1189] = 0.0 + k[690]*y[IDX_HeII] - k[762]*y[IDX_NHII];
    data[1190] = 0.0 - k[761]*y[IDX_NHII];
    data[1191] = 0.0 - k[758]*y[IDX_NHII];
    data[1192] = 0.0 - k[432]*y[IDX_NHII];
    data[1193] = 0.0 + k[86]*y[IDX_HII] + k[111]*y[IDX_H2II] + k[166]*y[IDX_NII] +
        k[199]*y[IDX_CNII] + k[200]*y[IDX_COII] + k[201]*y[IDX_N2II] +
        k[202]*y[IDX_OII] + k[275] - k[763]*y[IDX_NHII] + k[1163];
    data[1194] = 0.0 - k[747]*y[IDX_NHII];
    data[1195] = 0.0 - k[175]*y[IDX_NHII] - k[752]*y[IDX_NHII] - k[753]*y[IDX_NHII];
    data[1196] = 0.0 + k[723]*y[IDX_NII] - k[759]*y[IDX_NHII];
    data[1197] = 0.0 + k[685]*y[IDX_HNCI] + k[690]*y[IDX_NH2I] + k[691]*y[IDX_NH3I];
    data[1198] = 0.0 - k[470]*y[IDX_NHII];
    data[1199] = 0.0 - k[178]*y[IDX_NHII] - k[764]*y[IDX_NHII];
    data[1200] = 0.0 + k[522]*y[IDX_H2II] - k[740]*y[IDX_NHII];
    data[1201] = 0.0 - k[768]*y[IDX_NHII];
    data[1202] = 0.0 - k[179]*y[IDX_NHII] - k[765]*y[IDX_NHII] - k[766]*y[IDX_NHII];
    data[1203] = 0.0 - k[390]*y[IDX_NHII];
    data[1204] = 0.0 + k[86]*y[IDX_NHI];
    data[1205] = 0.0 - k[176]*y[IDX_NHII] - k[754]*y[IDX_NHII] - k[755]*y[IDX_NHII] -
        k[756]*y[IDX_NHII] - k[757]*y[IDX_NHII];
    data[1206] = 0.0 - k[767]*y[IDX_NHII];
    data[1207] = 0.0 - k[340]*y[IDX_NHII];
    data[1208] = 0.0 - k[751]*y[IDX_NHII];
    data[1209] = 0.0 + k[538]*y[IDX_NII] - k[540]*y[IDX_NHII] - k[541]*y[IDX_NHII];
    data[1210] = 0.0 - k[820]*y[IDX_O2II];
    data[1211] = 0.0 - k[152]*y[IDX_O2II];
    data[1212] = 0.0 + k[51]*y[IDX_O2I];
    data[1213] = 0.0 + k[68]*y[IDX_O2I];
    data[1214] = 0.0 - k[230]*y[IDX_O2II];
    data[1215] = 0.0 + k[173]*y[IDX_O2I];
    data[1216] = 0.0 + k[73]*y[IDX_O2I];
    data[1217] = 0.0 + k[113]*y[IDX_O2I];
    data[1218] = 0.0 - k[198]*y[IDX_O2II];
    data[1219] = 0.0 + k[667]*y[IDX_HeII] + k[812]*y[IDX_OII];
    data[1220] = 0.0 + k[168]*y[IDX_O2I];
    data[1221] = 0.0 + k[214]*y[IDX_O2I] + k[812]*y[IDX_CO2I] + k[819]*y[IDX_OHI];
    data[1222] = 0.0 + k[133]*y[IDX_O2I];
    data[1223] = 0.0 + k[179]*y[IDX_O2I];
    data[1224] = 0.0 - k[30]*y[IDX_CI] - k[44]*y[IDX_CH2I] - k[61]*y[IDX_CHI] -
        k[116]*y[IDX_H2COI] - k[137]*y[IDX_HCOI] - k[152]*y[IDX_MgI] -
        k[187]*y[IDX_NH2I] - k[198]*y[IDX_NH3I] - k[205]*y[IDX_NOI] -
        k[230]*y[IDX_SiI] - k[346]*y[IDX_eM] - k[391]*y[IDX_CI] -
        k[435]*y[IDX_CH2I] - k[473]*y[IDX_CHI] - k[551]*y[IDX_H2COI] -
        k[644]*y[IDX_HCOI] - k[742]*y[IDX_NI] - k[804]*y[IDX_NHI] -
        k[820]*y[IDX_CH3OHI] - k[1167] - k[1270];
    data[1225] = 0.0 - k[187]*y[IDX_O2II];
    data[1226] = 0.0 + k[121]*y[IDX_O2I] + k[823]*y[IDX_OI];
    data[1227] = 0.0 - k[44]*y[IDX_O2II] - k[435]*y[IDX_O2II];
    data[1228] = 0.0 - k[804]*y[IDX_O2II];
    data[1229] = 0.0 + k[224]*y[IDX_O2I] + k[830]*y[IDX_OI];
    data[1230] = 0.0 - k[116]*y[IDX_O2II] - k[551]*y[IDX_O2II];
    data[1231] = 0.0 - k[137]*y[IDX_O2II] - k[644]*y[IDX_O2II];
    data[1232] = 0.0 + k[146]*y[IDX_O2I] + k[667]*y[IDX_CO2I];
    data[1233] = 0.0 - k[61]*y[IDX_O2II] - k[473]*y[IDX_O2II];
    data[1234] = 0.0 - k[205]*y[IDX_O2II];
    data[1235] = 0.0 - k[742]*y[IDX_O2II];
    data[1236] = 0.0 + k[819]*y[IDX_OII];
    data[1237] = 0.0 + k[51]*y[IDX_CH4II] + k[68]*y[IDX_CNII] + k[73]*y[IDX_COII] +
        k[88]*y[IDX_HII] + k[113]*y[IDX_H2II] + k[121]*y[IDX_H2OII] +
        k[133]*y[IDX_HCNII] + k[146]*y[IDX_HeII] + k[168]*y[IDX_NII] +
        k[173]*y[IDX_N2II] + k[179]*y[IDX_NHII] + k[214]*y[IDX_OII] +
        k[224]*y[IDX_OHII] + k[279] + k[1168];
    data[1238] = 0.0 - k[30]*y[IDX_O2II] - k[391]*y[IDX_O2II];
    data[1239] = 0.0 + k[88]*y[IDX_O2I];
    data[1240] = 0.0 + k[823]*y[IDX_H2OII] + k[830]*y[IDX_OHII];
    data[1241] = 0.0 - k[346]*y[IDX_O2II];
    data[1242] = 0.0 - k[446]*y[IDX_CH3II];
    data[1243] = 0.0 + k[366]*y[IDX_CII] + k[396]*y[IDX_CHII] - k[439]*y[IDX_CH3II] +
        k[492]*y[IDX_HII] + k[579]*y[IDX_H3II] + k[657]*y[IDX_HeII] +
        k[715]*y[IDX_NII];
    data[1244] = 0.0 - k[47]*y[IDX_CH3II];
    data[1245] = 0.0 + k[617]*y[IDX_HI] + k[822]*y[IDX_OI] + k[1123];
    data[1246] = 0.0 + k[427]*y[IDX_CH2I] + k[428]*y[IDX_CH2I];
    data[1247] = 0.0 + k[431]*y[IDX_CH2I];
    data[1248] = 0.0 + k[436]*y[IDX_CH2I];
    data[1249] = 0.0 + k[430]*y[IDX_CH2I];
    data[1250] = 0.0 + k[456]*y[IDX_CH4I];
    data[1251] = 0.0 + k[456]*y[IDX_N2II] + k[495]*y[IDX_HII] + k[511]*y[IDX_H2II] +
        k[660]*y[IDX_HeII] + k[716]*y[IDX_NII] + k[810]*y[IDX_OII];
    data[1252] = 0.0 + k[510]*y[IDX_CH2I] + k[511]*y[IDX_CH4I];
    data[1253] = 0.0 + k[76]*y[IDX_HII] + k[245] + k[1117];
    data[1254] = 0.0 + k[715]*y[IDX_CH3OHI] + k[716]*y[IDX_CH4I];
    data[1255] = 0.0 + k[810]*y[IDX_CH4I];
    data[1256] = 0.0 + k[426]*y[IDX_CH2I];
    data[1257] = 0.0 + k[433]*y[IDX_CH2I];
    data[1258] = 0.0 + k[432]*y[IDX_CH2I];
    data[1259] = 0.0 - k[46]*y[IDX_HCOI] - k[47]*y[IDX_MgI] - k[48]*y[IDX_NOI] -
        k[298]*y[IDX_eM] - k[299]*y[IDX_eM] - k[300]*y[IDX_eM] -
        k[439]*y[IDX_CH3OHI] - k[440]*y[IDX_H2COI] - k[441]*y[IDX_HCOI] -
        k[442]*y[IDX_O2I] - k[443]*y[IDX_OI] - k[444]*y[IDX_OI] -
        k[445]*y[IDX_OHI] - k[446]*y[IDX_SiH4I] - k[616]*y[IDX_HI] -
        k[794]*y[IDX_NHI] - k[1114] - k[1115] - k[1215]*y[IDX_eM] - k[1285];
    data[1260] = 0.0 + k[419]*y[IDX_HCOI] + k[530]*y[IDX_H2I];
    data[1261] = 0.0 + k[424]*y[IDX_CH2I];
    data[1262] = 0.0 + k[434]*y[IDX_CH2I];
    data[1263] = 0.0 + k[425]*y[IDX_CH2I];
    data[1264] = 0.0 + k[366]*y[IDX_CH3OHI];
    data[1265] = 0.0 + k[396]*y[IDX_CH3OHI] + k[399]*y[IDX_H2COI];
    data[1266] = 0.0 + k[423]*y[IDX_H2COII] + k[424]*y[IDX_H2OII] + k[425]*y[IDX_H3OII] +
        k[426]*y[IDX_HCNII] + k[427]*y[IDX_HCNHII] + k[428]*y[IDX_HCNHII] +
        k[429]*y[IDX_HCOII] + k[430]*y[IDX_HNOII] + k[431]*y[IDX_N2HII] +
        k[432]*y[IDX_NHII] + k[433]*y[IDX_NH2II] + k[434]*y[IDX_NH3II] +
        k[436]*y[IDX_O2HII] + k[437]*y[IDX_OHII] + k[510]*y[IDX_H2II] +
        k[577]*y[IDX_H3II];
    data[1267] = 0.0 + k[423]*y[IDX_CH2I];
    data[1268] = 0.0 - k[794]*y[IDX_CH3II];
    data[1269] = 0.0 + k[437]*y[IDX_CH2I];
    data[1270] = 0.0 + k[399]*y[IDX_CHII] - k[440]*y[IDX_CH3II];
    data[1271] = 0.0 - k[46]*y[IDX_CH3II] + k[419]*y[IDX_CH2II] - k[441]*y[IDX_CH3II];
    data[1272] = 0.0 + k[657]*y[IDX_CH3OHI] + k[660]*y[IDX_CH4I];
    data[1273] = 0.0 + k[577]*y[IDX_CH2I] + k[579]*y[IDX_CH3OHI];
    data[1274] = 0.0 - k[48]*y[IDX_CH3II];
    data[1275] = 0.0 - k[445]*y[IDX_CH3II];
    data[1276] = 0.0 - k[442]*y[IDX_CH3II];
    data[1277] = 0.0 + k[76]*y[IDX_CH3I] + k[492]*y[IDX_CH3OHI] + k[495]*y[IDX_CH4I];
    data[1278] = 0.0 + k[429]*y[IDX_CH2I];
    data[1279] = 0.0 - k[443]*y[IDX_CH3II] - k[444]*y[IDX_CH3II] + k[822]*y[IDX_CH4II];
    data[1280] = 0.0 - k[298]*y[IDX_CH3II] - k[299]*y[IDX_CH3II] - k[300]*y[IDX_CH3II] -
        k[1215]*y[IDX_CH3II];
    data[1281] = 0.0 + k[530]*y[IDX_CH2II];
    data[1282] = 0.0 - k[616]*y[IDX_CH3II] + k[617]*y[IDX_CH4II];
    data[1283] = 0.0 + k[980]*y[IDX_HI];
    data[1284] = 0.0 - k[183]*y[IDX_NH2I];
    data[1285] = 0.0 - k[785]*y[IDX_NH2I] - k[786]*y[IDX_NH2I];
    data[1286] = 0.0 - k[789]*y[IDX_NH2I];
    data[1287] = 0.0 - k[790]*y[IDX_NH2I];
    data[1288] = 0.0 - k[788]*y[IDX_NH2I];
    data[1289] = 0.0 - k[186]*y[IDX_NH2I];
    data[1290] = 0.0 - k[782]*y[IDX_NH2I];
    data[1291] = 0.0 - k[1028]*y[IDX_NH2I] + k[1034]*y[IDX_NHI];
    data[1292] = 0.0 - k[184]*y[IDX_NH2I] - k[779]*y[IDX_NH2I] + k[792]*y[IDX_NH3I];
    data[1293] = 0.0 - k[109]*y[IDX_NH2I];
    data[1294] = 0.0 + k[181]*y[IDX_NH2II] + k[271] + k[792]*y[IDX_COII] +
        k[793]*y[IDX_HCNII] + k[908]*y[IDX_CH3I] + k[984]*y[IDX_HI] +
        k[1033]*y[IDX_CNI] + k[1037]*y[IDX_NHI] + k[1037]*y[IDX_NHI] +
        k[1074]*y[IDX_OI] + k[1098]*y[IDX_OHI] + k[1159];
    data[1295] = 0.0 - k[907]*y[IDX_NH2I] + k[908]*y[IDX_NH3I];
    data[1296] = 0.0 - k[164]*y[IDX_NH2I];
    data[1297] = 0.0 - k[212]*y[IDX_NH2I];
    data[1298] = 0.0 - k[784]*y[IDX_NH2I] + k[793]*y[IDX_NH3I];
    data[1299] = 0.0 + k[42]*y[IDX_CH2I] + k[59]*y[IDX_CHI] + k[180]*y[IDX_HCOI] +
        k[181]*y[IDX_NH3I] + k[182]*y[IDX_NOI] - k[776]*y[IDX_NH2I];
    data[1300] = 0.0 + k[753]*y[IDX_H2COI] - k[762]*y[IDX_NH2I];
    data[1301] = 0.0 - k[187]*y[IDX_NH2I];
    data[1302] = 0.0 - k[84]*y[IDX_HII] - k[109]*y[IDX_H2II] - k[164]*y[IDX_NII] -
        k[183]*y[IDX_CNII] - k[184]*y[IDX_COII] - k[185]*y[IDX_H2OII] -
        k[186]*y[IDX_N2II] - k[187]*y[IDX_O2II] - k[188]*y[IDX_OHII] -
        k[212]*y[IDX_OII] - k[269] - k[270] - k[373]*y[IDX_CII] -
        k[409]*y[IDX_CHII] - k[593]*y[IDX_H3II] - k[689]*y[IDX_HeII] -
        k[690]*y[IDX_HeII] - k[762]*y[IDX_NHII] - k[776]*y[IDX_NH2II] -
        k[779]*y[IDX_COII] - k[780]*y[IDX_H2COII] - k[781]*y[IDX_H2OII] -
        k[782]*y[IDX_H3COII] - k[783]*y[IDX_H3OII] - k[784]*y[IDX_HCNII] -
        k[785]*y[IDX_HCNHII] - k[786]*y[IDX_HCNHII] - k[787]*y[IDX_HCOII] -
        k[788]*y[IDX_HNOII] - k[789]*y[IDX_N2HII] - k[790]*y[IDX_O2HII] -
        k[791]*y[IDX_OHII] - k[866]*y[IDX_CI] - k[867]*y[IDX_CI] -
        k[868]*y[IDX_CI] - k[907]*y[IDX_CH3I] - k[961]*y[IDX_H2I] -
        k[983]*y[IDX_HI] - k[1028]*y[IDX_CH4I] - k[1029]*y[IDX_NOI] -
        k[1030]*y[IDX_NOI] - k[1031]*y[IDX_OHI] - k[1032]*y[IDX_OHI] -
        k[1072]*y[IDX_OI] - k[1073]*y[IDX_OI] - k[1157] - k[1158] - k[1289];
    data[1303] = 0.0 - k[185]*y[IDX_NH2I] - k[781]*y[IDX_NH2I];
    data[1304] = 0.0 + k[343]*y[IDX_eM] + k[434]*y[IDX_CH2I];
    data[1305] = 0.0 - k[783]*y[IDX_NH2I];
    data[1306] = 0.0 - k[373]*y[IDX_NH2I];
    data[1307] = 0.0 + k[1095]*y[IDX_OHI];
    data[1308] = 0.0 - k[409]*y[IDX_NH2I];
    data[1309] = 0.0 + k[42]*y[IDX_NH2II] + k[434]*y[IDX_NH3II];
    data[1310] = 0.0 - k[780]*y[IDX_NH2I];
    data[1311] = 0.0 + k[962]*y[IDX_H2I] + k[1034]*y[IDX_CH4I] + k[1036]*y[IDX_H2OI] +
        k[1037]*y[IDX_NH3I] + k[1037]*y[IDX_NH3I] + k[1040]*y[IDX_NHI] +
        k[1040]*y[IDX_NHI] + k[1050]*y[IDX_OHI];
    data[1312] = 0.0 - k[188]*y[IDX_NH2I] - k[791]*y[IDX_NH2I];
    data[1313] = 0.0 + k[1033]*y[IDX_NH3I];
    data[1314] = 0.0 + k[753]*y[IDX_NHII];
    data[1315] = 0.0 + k[180]*y[IDX_NH2II];
    data[1316] = 0.0 - k[689]*y[IDX_NH2I] - k[690]*y[IDX_NH2I];
    data[1317] = 0.0 + k[59]*y[IDX_NH2II];
    data[1318] = 0.0 - k[593]*y[IDX_NH2I];
    data[1319] = 0.0 + k[182]*y[IDX_NH2II] - k[1029]*y[IDX_NH2I] - k[1030]*y[IDX_NH2I];
    data[1320] = 0.0 - k[1031]*y[IDX_NH2I] - k[1032]*y[IDX_NH2I] + k[1050]*y[IDX_NHI] +
        k[1095]*y[IDX_HCNI] + k[1098]*y[IDX_NH3I];
    data[1321] = 0.0 - k[866]*y[IDX_NH2I] - k[867]*y[IDX_NH2I] - k[868]*y[IDX_NH2I];
    data[1322] = 0.0 - k[84]*y[IDX_NH2I];
    data[1323] = 0.0 - k[787]*y[IDX_NH2I];
    data[1324] = 0.0 + k[1036]*y[IDX_NHI];
    data[1325] = 0.0 - k[1072]*y[IDX_NH2I] - k[1073]*y[IDX_NH2I] + k[1074]*y[IDX_NH3I];
    data[1326] = 0.0 + k[343]*y[IDX_NH3II];
    data[1327] = 0.0 - k[961]*y[IDX_NH2I] + k[962]*y[IDX_NHI];
    data[1328] = 0.0 + k[980]*y[IDX_HNOI] - k[983]*y[IDX_NH2I] + k[984]*y[IDX_NH3I];
    data[1329] = 0.0 + k[477]*y[IDX_CHI];
    data[1330] = 0.0 + k[1122];
    data[1331] = 0.0 + k[37]*y[IDX_CH2I];
    data[1332] = 0.0 + k[464]*y[IDX_CHI] + k[465]*y[IDX_CHI];
    data[1333] = 0.0 + k[469]*y[IDX_CHI];
    data[1334] = 0.0 + k[474]*y[IDX_CHI];
    data[1335] = 0.0 + k[467]*y[IDX_CHI];
    data[1336] = 0.0 + k[41]*y[IDX_CH2I] + k[455]*y[IDX_CH4I];
    data[1337] = 0.0 + k[461]*y[IDX_CHI];
    data[1338] = 0.0 + k[455]*y[IDX_N2II] + k[659]*y[IDX_HeII];
    data[1339] = 0.0 + k[38]*y[IDX_CH2I];
    data[1340] = 0.0 + k[100]*y[IDX_CH2I] + k[512]*y[IDX_CHI];
    data[1341] = 0.0 - k[416]*y[IDX_CH2II];
    data[1342] = 0.0 + k[155]*y[IDX_CH2I];
    data[1343] = 0.0 + k[43]*y[IDX_CH2I];
    data[1344] = 0.0 + k[463]*y[IDX_CHI];
    data[1345] = 0.0 + k[42]*y[IDX_CH2I] + k[471]*y[IDX_CHI];
    data[1346] = 0.0 + k[470]*y[IDX_CHI];
    data[1347] = 0.0 + k[44]*y[IDX_CH2I];
    data[1348] = 0.0 + k[616]*y[IDX_HI] + k[1115];
    data[1349] = 0.0 - k[36]*y[IDX_NOI] - k[295]*y[IDX_eM] - k[296]*y[IDX_eM] -
        k[297]*y[IDX_eM] - k[416]*y[IDX_CO2I] - k[417]*y[IDX_H2COI] -
        k[418]*y[IDX_H2OI] - k[419]*y[IDX_HCOI] - k[420]*y[IDX_O2I] -
        k[421]*y[IDX_OI] - k[530]*y[IDX_H2I] - k[615]*y[IDX_HI] -
        k[736]*y[IDX_NI] - k[1109] - k[1110] - k[1111] - k[1278];
    data[1350] = 0.0 + k[40]*y[IDX_CH2I] + k[460]*y[IDX_CHI];
    data[1351] = 0.0 + k[462]*y[IDX_CHI];
    data[1352] = 0.0 + k[14]*y[IDX_CH2I] + k[368]*y[IDX_H2COI] + k[1199]*y[IDX_H2I];
    data[1353] = 0.0 + k[406]*y[IDX_HCOI] + k[529]*y[IDX_H2I];
    data[1354] = 0.0 + k[14]*y[IDX_CII] + k[37]*y[IDX_CNII] + k[38]*y[IDX_COII] +
        k[39]*y[IDX_H2COII] + k[40]*y[IDX_H2OII] + k[41]*y[IDX_N2II] +
        k[42]*y[IDX_NH2II] + k[43]*y[IDX_OII] + k[44]*y[IDX_O2II] +
        k[45]*y[IDX_OHII] + k[75]*y[IDX_HII] + k[100]*y[IDX_H2II] +
        k[155]*y[IDX_NII] + k[242] + k[1112];
    data[1355] = 0.0 + k[39]*y[IDX_CH2I] + k[459]*y[IDX_CHI];
    data[1356] = 0.0 + k[45]*y[IDX_CH2I] + k[475]*y[IDX_CHI];
    data[1357] = 0.0 + k[368]*y[IDX_CII] - k[417]*y[IDX_CH2II] + k[672]*y[IDX_HeII];
    data[1358] = 0.0 + k[406]*y[IDX_CHII] - k[419]*y[IDX_CH2II];
    data[1359] = 0.0 + k[659]*y[IDX_CH4I] + k[672]*y[IDX_H2COI];
    data[1360] = 0.0 + k[459]*y[IDX_H2COII] + k[460]*y[IDX_H2OII] + k[461]*y[IDX_H3COII]
        + k[462]*y[IDX_H3OII] + k[463]*y[IDX_HCNII] + k[464]*y[IDX_HCNHII] +
        k[465]*y[IDX_HCNHII] + k[466]*y[IDX_HCOII] + k[467]*y[IDX_HNOII] +
        k[469]*y[IDX_N2HII] + k[470]*y[IDX_NHII] + k[471]*y[IDX_NH2II] +
        k[474]*y[IDX_O2HII] + k[475]*y[IDX_OHII] + k[477]*y[IDX_SiHII] +
        k[512]*y[IDX_H2II] + k[580]*y[IDX_H3II];
    data[1361] = 0.0 + k[580]*y[IDX_CHI];
    data[1362] = 0.0 - k[36]*y[IDX_CH2II];
    data[1363] = 0.0 - k[736]*y[IDX_CH2II];
    data[1364] = 0.0 - k[420]*y[IDX_CH2II];
    data[1365] = 0.0 + k[75]*y[IDX_CH2I];
    data[1366] = 0.0 + k[466]*y[IDX_CHI];
    data[1367] = 0.0 - k[418]*y[IDX_CH2II];
    data[1368] = 0.0 - k[421]*y[IDX_CH2II];
    data[1369] = 0.0 - k[295]*y[IDX_CH2II] - k[296]*y[IDX_CH2II] - k[297]*y[IDX_CH2II];
    data[1370] = 0.0 + k[529]*y[IDX_CHII] - k[530]*y[IDX_CH2II] + k[1199]*y[IDX_CII];
    data[1371] = 0.0 - k[615]*y[IDX_CH2II] + k[616]*y[IDX_CH3II];
    data[1372] = 0.0 - k[119]*y[IDX_H2OII];
    data[1373] = 0.0 + k[857]*y[IDX_OHI];
    data[1374] = 0.0 + k[858]*y[IDX_OHI];
    data[1375] = 0.0 - k[122]*y[IDX_H2OII];
    data[1376] = 0.0 - k[559]*y[IDX_H2OII];
    data[1377] = 0.0 + k[856]*y[IDX_OHI];
    data[1378] = 0.0 + k[125]*y[IDX_H2OI];
    data[1379] = 0.0 - k[453]*y[IDX_H2OII];
    data[1380] = 0.0 + k[123]*y[IDX_H2OI];
    data[1381] = 0.0 + k[106]*y[IDX_H2OI] + k[527]*y[IDX_OHI];
    data[1382] = 0.0 - k[195]*y[IDX_H2OII];
    data[1383] = 0.0 + k[160]*y[IDX_H2OI];
    data[1384] = 0.0 + k[210]*y[IDX_H2OI];
    data[1385] = 0.0 + k[124]*y[IDX_H2OI] + k[853]*y[IDX_OHI];
    data[1386] = 0.0 + k[176]*y[IDX_H2OI] + k[768]*y[IDX_OHI];
    data[1387] = 0.0 - k[185]*y[IDX_H2OII] - k[781]*y[IDX_H2OII];
    data[1388] = 0.0 - k[40]*y[IDX_CH2I] - k[56]*y[IDX_CHI] - k[117]*y[IDX_H2COI] -
        k[118]*y[IDX_HCOI] - k[119]*y[IDX_MgI] - k[120]*y[IDX_NOI] -
        k[121]*y[IDX_O2I] - k[122]*y[IDX_SiI] - k[185]*y[IDX_NH2I] -
        k[195]*y[IDX_NH3I] - k[312]*y[IDX_eM] - k[313]*y[IDX_eM] -
        k[314]*y[IDX_eM] - k[383]*y[IDX_CI] - k[424]*y[IDX_CH2I] -
        k[453]*y[IDX_CH4I] - k[460]*y[IDX_CHI] - k[534]*y[IDX_H2I] -
        k[553]*y[IDX_COI] - k[554]*y[IDX_H2COI] - k[555]*y[IDX_H2OI] -
        k[556]*y[IDX_HCNI] - k[557]*y[IDX_HCOI] - k[558]*y[IDX_HCOI] -
        k[559]*y[IDX_HNCI] - k[738]*y[IDX_NI] - k[739]*y[IDX_NI] -
        k[781]*y[IDX_NH2I] - k[797]*y[IDX_NHI] - k[823]*y[IDX_OI] -
        k[852]*y[IDX_OHI] - k[1140] - k[1280];
    data[1389] = 0.0 - k[556]*y[IDX_H2OII];
    data[1390] = 0.0 - k[40]*y[IDX_H2OII] - k[424]*y[IDX_H2OII];
    data[1391] = 0.0 - k[797]*y[IDX_H2OII];
    data[1392] = 0.0 + k[220]*y[IDX_H2OI] + k[545]*y[IDX_H2I] + k[842]*y[IDX_HCOI] +
        k[847]*y[IDX_OHI];
    data[1393] = 0.0 - k[117]*y[IDX_H2OII] - k[554]*y[IDX_H2OII];
    data[1394] = 0.0 - k[118]*y[IDX_H2OII] - k[557]*y[IDX_H2OII] - k[558]*y[IDX_H2OII] +
        k[842]*y[IDX_OHII];
    data[1395] = 0.0 + k[143]*y[IDX_H2OI];
    data[1396] = 0.0 - k[56]*y[IDX_H2OII] - k[460]*y[IDX_H2OII];
    data[1397] = 0.0 + k[598]*y[IDX_OI] + k[600]*y[IDX_OHI];
    data[1398] = 0.0 - k[120]*y[IDX_H2OII];
    data[1399] = 0.0 - k[738]*y[IDX_H2OII] - k[739]*y[IDX_H2OII];
    data[1400] = 0.0 + k[527]*y[IDX_H2II] + k[600]*y[IDX_H3II] + k[768]*y[IDX_NHII] +
        k[847]*y[IDX_OHII] - k[852]*y[IDX_H2OII] + k[853]*y[IDX_HCNII] +
        k[854]*y[IDX_HCOII] + k[856]*y[IDX_HNOII] + k[857]*y[IDX_N2HII] +
        k[858]*y[IDX_O2HII];
    data[1401] = 0.0 - k[121]*y[IDX_H2OII];
    data[1402] = 0.0 - k[383]*y[IDX_H2OII];
    data[1403] = 0.0 + k[80]*y[IDX_H2OI];
    data[1404] = 0.0 + k[854]*y[IDX_OHI];
    data[1405] = 0.0 + k[80]*y[IDX_HII] + k[106]*y[IDX_H2II] + k[123]*y[IDX_COII] +
        k[124]*y[IDX_HCNII] + k[125]*y[IDX_N2II] + k[143]*y[IDX_HeII] +
        k[160]*y[IDX_NII] + k[176]*y[IDX_NHII] + k[210]*y[IDX_OII] +
        k[220]*y[IDX_OHII] - k[555]*y[IDX_H2OII] + k[1141];
    data[1406] = 0.0 + k[598]*y[IDX_H3II] - k[823]*y[IDX_H2OII];
    data[1407] = 0.0 - k[312]*y[IDX_H2OII] - k[313]*y[IDX_H2OII] - k[314]*y[IDX_H2OII];
    data[1408] = 0.0 - k[553]*y[IDX_H2OII];
    data[1409] = 0.0 - k[534]*y[IDX_H2OII] + k[545]*y[IDX_OHII];
    data[1410] = 0.0 - k[190]*y[IDX_NH3II];
    data[1411] = 0.0 + k[50]*y[IDX_NH3I];
    data[1412] = 0.0 + k[785]*y[IDX_NH2I] + k[786]*y[IDX_NH2I];
    data[1413] = 0.0 + k[789]*y[IDX_NH2I];
    data[1414] = 0.0 + k[790]*y[IDX_NH2I];
    data[1415] = 0.0 - k[192]*y[IDX_NH3II];
    data[1416] = 0.0 + k[788]*y[IDX_NH2I];
    data[1417] = 0.0 + k[197]*y[IDX_NH3I];
    data[1418] = 0.0 + k[782]*y[IDX_NH2I];
    data[1419] = 0.0 + k[193]*y[IDX_NH3I];
    data[1420] = 0.0 + k[110]*y[IDX_NH3I];
    data[1421] = 0.0 + k[19]*y[IDX_CII] + k[33]*y[IDX_CHII] + k[50]*y[IDX_CH4II] +
        k[85]*y[IDX_HII] + k[110]*y[IDX_H2II] + k[145]*y[IDX_HeII] +
        k[165]*y[IDX_NII] + k[177]*y[IDX_NHII] + k[181]*y[IDX_NH2II] +
        k[193]*y[IDX_COII] + k[194]*y[IDX_H2COII] + k[195]*y[IDX_H2OII] +
        k[196]*y[IDX_HCNII] + k[197]*y[IDX_N2II] + k[198]*y[IDX_O2II] +
        k[213]*y[IDX_OII] + k[222]*y[IDX_OHII] + k[272] + k[1160];
    data[1422] = 0.0 + k[165]*y[IDX_NH3I];
    data[1423] = 0.0 + k[213]*y[IDX_NH3I];
    data[1424] = 0.0 + k[196]*y[IDX_NH3I] + k[784]*y[IDX_NH2I];
    data[1425] = 0.0 + k[181]*y[IDX_NH3I] + k[542]*y[IDX_H2I] + k[770]*y[IDX_H2COI] +
        k[772]*y[IDX_H2OI] + k[776]*y[IDX_NH2I] + k[802]*y[IDX_NHI];
    data[1426] = 0.0 + k[177]*y[IDX_NH3I] + k[756]*y[IDX_H2OI] + k[762]*y[IDX_NH2I];
    data[1427] = 0.0 + k[198]*y[IDX_NH3I];
    data[1428] = 0.0 + k[593]*y[IDX_H3II] + k[762]*y[IDX_NHII] + k[776]*y[IDX_NH2II] +
        k[780]*y[IDX_H2COII] + k[781]*y[IDX_H2OII] + k[782]*y[IDX_H3COII] +
        k[783]*y[IDX_H3OII] + k[784]*y[IDX_HCNII] + k[785]*y[IDX_HCNHII] +
        k[786]*y[IDX_HCNHII] + k[787]*y[IDX_HCOII] + k[788]*y[IDX_HNOII] +
        k[789]*y[IDX_N2HII] + k[790]*y[IDX_O2HII] + k[791]*y[IDX_OHII];
    data[1429] = 0.0 + k[195]*y[IDX_NH3I] + k[781]*y[IDX_NH2I];
    data[1430] = 0.0 - k[189]*y[IDX_HCOI] - k[190]*y[IDX_MgI] - k[191]*y[IDX_NOI] -
        k[192]*y[IDX_SiI] - k[343]*y[IDX_eM] - k[344]*y[IDX_eM] -
        k[434]*y[IDX_CH2I] - k[828]*y[IDX_OI] - k[1283];
    data[1431] = 0.0 + k[783]*y[IDX_NH2I];
    data[1432] = 0.0 + k[19]*y[IDX_NH3I];
    data[1433] = 0.0 + k[33]*y[IDX_NH3I];
    data[1434] = 0.0 - k[434]*y[IDX_NH3II];
    data[1435] = 0.0 + k[194]*y[IDX_NH3I] + k[780]*y[IDX_NH2I];
    data[1436] = 0.0 + k[802]*y[IDX_NH2II];
    data[1437] = 0.0 + k[222]*y[IDX_NH3I] + k[791]*y[IDX_NH2I];
    data[1438] = 0.0 + k[770]*y[IDX_NH2II];
    data[1439] = 0.0 - k[189]*y[IDX_NH3II];
    data[1440] = 0.0 + k[145]*y[IDX_NH3I];
    data[1441] = 0.0 + k[593]*y[IDX_NH2I];
    data[1442] = 0.0 - k[191]*y[IDX_NH3II];
    data[1443] = 0.0 + k[85]*y[IDX_NH3I];
    data[1444] = 0.0 + k[787]*y[IDX_NH2I];
    data[1445] = 0.0 + k[756]*y[IDX_NHII] + k[772]*y[IDX_NH2II];
    data[1446] = 0.0 - k[828]*y[IDX_NH3II];
    data[1447] = 0.0 - k[343]*y[IDX_NH3II] - k[344]*y[IDX_NH3II];
    data[1448] = 0.0 + k[542]*y[IDX_NH2II];
    data[1449] = 0.0 + k[504]*y[IDX_HII] + k[595]*y[IDX_H3II] + k[818]*y[IDX_OII];
    data[1450] = 0.0 + k[206]*y[IDX_NOI] + k[745]*y[IDX_NI];
    data[1451] = 0.0 + k[503]*y[IDX_HII] + k[686]*y[IDX_HeII];
    data[1452] = 0.0 + k[714]*y[IDX_NII];
    data[1453] = 0.0 - k[151]*y[IDX_NOII];
    data[1454] = 0.0 + k[67]*y[IDX_NOI] + k[481]*y[IDX_O2I];
    data[1455] = 0.0 - k[229]*y[IDX_NOII];
    data[1456] = 0.0 + k[204]*y[IDX_NOI];
    data[1457] = 0.0 + k[172]*y[IDX_NOI] + k[825]*y[IDX_OI];
    data[1458] = 0.0 + k[72]*y[IDX_NOI];
    data[1459] = 0.0 + k[112]*y[IDX_NOI];
    data[1460] = 0.0 + k[750]*y[IDX_NHII];
    data[1461] = 0.0 + k[167]*y[IDX_NOI] + k[714]*y[IDX_CH3OHI] + k[720]*y[IDX_COI] +
        k[722]*y[IDX_H2COI] + k[728]*y[IDX_O2I];
    data[1462] = 0.0 + k[803]*y[IDX_NHI] + k[811]*y[IDX_CNI] + k[815]*y[IDX_HCNI] +
        k[817]*y[IDX_N2I] + k[818]*y[IDX_NO2I];
    data[1463] = 0.0 + k[132]*y[IDX_NOI];
    data[1464] = 0.0 + k[182]*y[IDX_NOI];
    data[1465] = 0.0 + k[178]*y[IDX_NOI] + k[750]*y[IDX_CO2I] + k[765]*y[IDX_O2I];
    data[1466] = 0.0 + k[205]*y[IDX_NOI] + k[742]*y[IDX_NI];
    data[1467] = 0.0 + k[48]*y[IDX_NOI];
    data[1468] = 0.0 + k[36]*y[IDX_NOI];
    data[1469] = 0.0 + k[120]*y[IDX_NOI] + k[739]*y[IDX_NI];
    data[1470] = 0.0 + k[191]*y[IDX_NOI];
    data[1471] = 0.0 - k[151]*y[IDX_MgI] - k[229]*y[IDX_SiI] - k[345]*y[IDX_eM] - k[1277];
    data[1472] = 0.0 + k[817]*y[IDX_OII];
    data[1473] = 0.0 + k[20]*y[IDX_NOI];
    data[1474] = 0.0 + k[815]*y[IDX_OII];
    data[1475] = 0.0 + k[34]*y[IDX_NOI];
    data[1476] = 0.0 + k[203]*y[IDX_NOI];
    data[1477] = 0.0 + k[803]*y[IDX_OII];
    data[1478] = 0.0 + k[223]*y[IDX_NOI] + k[743]*y[IDX_NI];
    data[1479] = 0.0 + k[811]*y[IDX_OII];
    data[1480] = 0.0 + k[722]*y[IDX_NII];
    data[1481] = 0.0 + k[686]*y[IDX_HNOI];
    data[1482] = 0.0 + k[595]*y[IDX_NO2I];
    data[1483] = 0.0 + k[20]*y[IDX_CII] + k[34]*y[IDX_CHII] + k[36]*y[IDX_CH2II] +
        k[48]*y[IDX_CH3II] + k[67]*y[IDX_CNII] + k[72]*y[IDX_COII] +
        k[87]*y[IDX_HII] + k[112]*y[IDX_H2II] + k[120]*y[IDX_H2OII] +
        k[132]*y[IDX_HCNII] + k[167]*y[IDX_NII] + k[172]*y[IDX_N2II] +
        k[178]*y[IDX_NHII] + k[182]*y[IDX_NH2II] + k[191]*y[IDX_NH3II] +
        k[203]*y[IDX_H2COII] + k[204]*y[IDX_HNOII] + k[205]*y[IDX_O2II] +
        k[206]*y[IDX_SiOII] + k[223]*y[IDX_OHII] + k[277] + k[1165];
    data[1484] = 0.0 + k[739]*y[IDX_H2OII] + k[742]*y[IDX_O2II] + k[743]*y[IDX_OHII] +
        k[745]*y[IDX_SiOII];
    data[1485] = 0.0 + k[481]*y[IDX_CNII] + k[728]*y[IDX_NII] + k[765]*y[IDX_NHII];
    data[1486] = 0.0 + k[87]*y[IDX_NOI] + k[503]*y[IDX_HNOI] + k[504]*y[IDX_NO2I];
    data[1487] = 0.0 + k[825]*y[IDX_N2II];
    data[1488] = 0.0 - k[345]*y[IDX_NOII];
    data[1489] = 0.0 + k[720]*y[IDX_NII];
    data[1490] = 0.0 + k[575]*y[IDX_H2OI];
    data[1491] = 0.0 + k[574]*y[IDX_H2OI];
    data[1492] = 0.0 - k[611]*y[IDX_H3OII];
    data[1493] = 0.0 + k[573]*y[IDX_H2OI];
    data[1494] = 0.0 - k[612]*y[IDX_H3OII];
    data[1495] = 0.0 + k[567]*y[IDX_H2OI];
    data[1496] = 0.0 + k[450]*y[IDX_H2OI];
    data[1497] = 0.0 - k[613]*y[IDX_H3OII];
    data[1498] = 0.0 + k[570]*y[IDX_H2OI];
    data[1499] = 0.0 + k[571]*y[IDX_H2OI];
    data[1500] = 0.0 - k[610]*y[IDX_H3OII];
    data[1501] = 0.0 - k[609]*y[IDX_H3OII];
    data[1502] = 0.0 + k[568]*y[IDX_H2OI];
    data[1503] = 0.0 + k[564]*y[IDX_H2OI];
    data[1504] = 0.0 + k[453]*y[IDX_H2OII] + k[457]*y[IDX_OHII];
    data[1505] = 0.0 + k[518]*y[IDX_H2OI];
    data[1506] = 0.0 + k[565]*y[IDX_H2OI];
    data[1507] = 0.0 + k[771]*y[IDX_H2OI];
    data[1508] = 0.0 + k[754]*y[IDX_H2OI];
    data[1509] = 0.0 - k[783]*y[IDX_H3OII];
    data[1510] = 0.0 + k[453]*y[IDX_CH4I] + k[534]*y[IDX_H2I] + k[555]*y[IDX_H2OI] +
        k[557]*y[IDX_HCOI] + k[797]*y[IDX_NHI] + k[852]*y[IDX_OHI];
    data[1511] = 0.0 - k[322]*y[IDX_eM] - k[323]*y[IDX_eM] - k[324]*y[IDX_eM] -
        k[325]*y[IDX_eM] - k[384]*y[IDX_CI] - k[425]*y[IDX_CH2I] -
        k[462]*y[IDX_CHI] - k[607]*y[IDX_H2COI] - k[608]*y[IDX_HCNI] -
        k[609]*y[IDX_HNCI] - k[610]*y[IDX_SiI] - k[611]*y[IDX_SiH2I] -
        k[612]*y[IDX_SiHI] - k[613]*y[IDX_SiOI] - k[783]*y[IDX_NH2I] - k[1286];
    data[1512] = 0.0 - k[608]*y[IDX_H3OII];
    data[1513] = 0.0 + k[403]*y[IDX_H2OI];
    data[1514] = 0.0 - k[425]*y[IDX_H3OII];
    data[1515] = 0.0 + k[563]*y[IDX_H2OI];
    data[1516] = 0.0 + k[797]*y[IDX_H2OII];
    data[1517] = 0.0 + k[457]*y[IDX_CH4I] + k[840]*y[IDX_H2OI];
    data[1518] = 0.0 - k[607]*y[IDX_H3OII];
    data[1519] = 0.0 + k[557]*y[IDX_H2OII];
    data[1520] = 0.0 - k[462]*y[IDX_H3OII];
    data[1521] = 0.0 + k[586]*y[IDX_H2OI];
    data[1522] = 0.0 + k[852]*y[IDX_H2OII];
    data[1523] = 0.0 - k[384]*y[IDX_H3OII];
    data[1524] = 0.0 + k[566]*y[IDX_H2OI];
    data[1525] = 0.0 + k[403]*y[IDX_CHII] + k[450]*y[IDX_CH4II] + k[518]*y[IDX_H2II] +
        k[555]*y[IDX_H2OII] + k[563]*y[IDX_H2COII] + k[564]*y[IDX_H3COII] +
        k[565]*y[IDX_HCNII] + k[566]*y[IDX_HCOII] + k[567]*y[IDX_HCO2II] +
        k[568]*y[IDX_HNOII] + k[570]*y[IDX_N2HII] + k[571]*y[IDX_O2HII] +
        k[573]*y[IDX_SiHII] + k[574]*y[IDX_SiH4II] + k[575]*y[IDX_SiH5II] +
        k[586]*y[IDX_H3II] + k[754]*y[IDX_NHII] + k[771]*y[IDX_NH2II] +
        k[840]*y[IDX_OHII];
    data[1526] = 0.0 - k[322]*y[IDX_H3OII] - k[323]*y[IDX_H3OII] - k[324]*y[IDX_H3OII] -
        k[325]*y[IDX_H3OII];
    data[1527] = 0.0 + k[534]*y[IDX_H2OII];
    data[1528] = 0.0 + k[1335] + k[1336] + k[1337] + k[1338];
    data[1529] = 0.0 + k[1019]*y[IDX_NI] + k[1021]*y[IDX_NI];
    data[1530] = 0.0 + k[1053]*y[IDX_NOI];
    data[1531] = 0.0 + k[150]*y[IDX_N2II];
    data[1532] = 0.0 + k[338]*y[IDX_eM] + k[389]*y[IDX_CI] + k[431]*y[IDX_CH2I] +
        k[469]*y[IDX_CHI] + k[487]*y[IDX_COI] + k[570]*y[IDX_H2OI] +
        k[631]*y[IDX_HCNI] + k[643]*y[IDX_HCOI] + k[650]*y[IDX_HNCI] +
        k[734]*y[IDX_CO2I] + k[735]*y[IDX_H2COI] + k[789]*y[IDX_NH2I] +
        k[801]*y[IDX_NHI] + k[826]*y[IDX_OI] + k[857]*y[IDX_OHI];
    data[1533] = 0.0 - k[733]*y[IDX_N2I];
    data[1534] = 0.0 + k[650]*y[IDX_N2HII];
    data[1535] = 0.0 - k[732]*y[IDX_N2I];
    data[1536] = 0.0 + k[29]*y[IDX_CI] + k[41]*y[IDX_CH2I] + k[58]*y[IDX_CHI] +
        k[69]*y[IDX_CNI] + k[74]*y[IDX_COI] + k[125]*y[IDX_H2OI] +
        k[135]*y[IDX_HCNI] + k[150]*y[IDX_MgI] + k[170]*y[IDX_H2COI] +
        k[171]*y[IDX_HCOI] + k[172]*y[IDX_NOI] + k[173]*y[IDX_O2I] +
        k[174]*y[IDX_NI] + k[186]*y[IDX_NH2I] + k[197]*y[IDX_NH3I] +
        k[201]*y[IDX_NHI] + k[218]*y[IDX_OI] + k[227]*y[IDX_OHI] +
        k[455]*y[IDX_CH4I] + k[456]*y[IDX_CH4I] + k[730]*y[IDX_H2COI];
    data[1537] = 0.0 + k[455]*y[IDX_N2II] + k[456]*y[IDX_N2II];
    data[1538] = 0.0 - k[521]*y[IDX_N2I];
    data[1539] = 0.0 + k[197]*y[IDX_N2II];
    data[1540] = 0.0 + k[734]*y[IDX_N2HII];
    data[1541] = 0.0 - k[817]*y[IDX_N2I];
    data[1542] = 0.0 - k[761]*y[IDX_N2I];
    data[1543] = 0.0 + k[186]*y[IDX_N2II] + k[789]*y[IDX_N2HII] + k[1029]*y[IDX_NOI] +
        k[1030]*y[IDX_NOI];
    data[1544] = 0.0 - k[144]*y[IDX_HeII] - k[267] - k[521]*y[IDX_H2II] -
        k[592]*y[IDX_H3II] - k[688]*y[IDX_HeII] - k[732]*y[IDX_HNOII] -
        k[733]*y[IDX_O2HII] - k[761]*y[IDX_NHII] - k[817]*y[IDX_OII] -
        k[845]*y[IDX_OHII] - k[865]*y[IDX_CI] - k[884]*y[IDX_CH2I] -
        k[927]*y[IDX_CHI] - k[1071]*y[IDX_OI] - k[1155] - k[1262];
    data[1545] = 0.0 + k[135]*y[IDX_N2II] + k[631]*y[IDX_N2HII];
    data[1546] = 0.0 + k[41]*y[IDX_N2II] + k[431]*y[IDX_N2HII] - k[884]*y[IDX_N2I];
    data[1547] = 0.0 + k[201]*y[IDX_N2II] + k[801]*y[IDX_N2HII] + k[1018]*y[IDX_NI] +
        k[1038]*y[IDX_NHI] + k[1038]*y[IDX_NHI] + k[1039]*y[IDX_NHI] +
        k[1039]*y[IDX_NHI] + k[1042]*y[IDX_NOI] + k[1043]*y[IDX_NOI];
    data[1548] = 0.0 - k[845]*y[IDX_N2I];
    data[1549] = 0.0 + k[69]*y[IDX_N2II] + k[946]*y[IDX_NOI] + k[1011]*y[IDX_NI];
    data[1550] = 0.0 + k[170]*y[IDX_N2II] + k[730]*y[IDX_N2II] + k[735]*y[IDX_N2HII];
    data[1551] = 0.0 + k[171]*y[IDX_N2II] + k[643]*y[IDX_N2HII];
    data[1552] = 0.0 - k[144]*y[IDX_N2I] - k[688]*y[IDX_N2I];
    data[1553] = 0.0 + k[58]*y[IDX_N2II] + k[469]*y[IDX_N2HII] - k[927]*y[IDX_N2I];
    data[1554] = 0.0 - k[592]*y[IDX_N2I];
    data[1555] = 0.0 + k[172]*y[IDX_N2II] + k[946]*y[IDX_CNI] + k[1022]*y[IDX_NI] +
        k[1029]*y[IDX_NH2I] + k[1030]*y[IDX_NH2I] + k[1042]*y[IDX_NHI] +
        k[1043]*y[IDX_NHI] + k[1051]*y[IDX_NOI] + k[1051]*y[IDX_NOI] +
        k[1053]*y[IDX_OCNI];
    data[1556] = 0.0 + k[174]*y[IDX_N2II] + k[1011]*y[IDX_CNI] + k[1018]*y[IDX_NHI] +
        k[1019]*y[IDX_NO2I] + k[1021]*y[IDX_NO2I] + k[1022]*y[IDX_NOI];
    data[1557] = 0.0 + k[227]*y[IDX_N2II] + k[857]*y[IDX_N2HII];
    data[1558] = 0.0 + k[173]*y[IDX_N2II];
    data[1559] = 0.0 + k[29]*y[IDX_N2II] + k[389]*y[IDX_N2HII] - k[865]*y[IDX_N2I];
    data[1560] = 0.0 + k[125]*y[IDX_N2II] + k[570]*y[IDX_N2HII];
    data[1561] = 0.0 + k[218]*y[IDX_N2II] + k[826]*y[IDX_N2HII] - k[1071]*y[IDX_N2I];
    data[1562] = 0.0 + k[338]*y[IDX_N2HII];
    data[1563] = 0.0 + k[74]*y[IDX_N2II] + k[487]*y[IDX_N2HII];
    data[1564] = 0.0 - k[22]*y[IDX_CII];
    data[1565] = 0.0 - k[23]*y[IDX_CII];
    data[1566] = 0.0 - k[24]*y[IDX_CII] + k[702]*y[IDX_HeII];
    data[1567] = 0.0 - k[378]*y[IDX_CII];
    data[1568] = 0.0 - k[25]*y[IDX_CII] - k[380]*y[IDX_CII];
    data[1569] = 0.0 - k[381]*y[IDX_CII];
    data[1570] = 0.0 - k[26]*y[IDX_CII];
    data[1571] = 0.0 - k[365]*y[IDX_CII] - k[366]*y[IDX_CII];
    data[1572] = 0.0 - k[18]*y[IDX_CII];
    data[1573] = 0.0 - k[382]*y[IDX_CII];
    data[1574] = 0.0 + k[27]*y[IDX_CI];
    data[1575] = 0.0 - k[21]*y[IDX_CII];
    data[1576] = 0.0 + k[684]*y[IDX_HeII];
    data[1577] = 0.0 + k[29]*y[IDX_CI];
    data[1578] = 0.0 + k[28]*y[IDX_CI] + k[1131];
    data[1579] = 0.0 - k[19]*y[IDX_CII] - k[374]*y[IDX_CII];
    data[1580] = 0.0 - k[367]*y[IDX_CII] + k[668]*y[IDX_HeII];
    data[1581] = 0.0 + k[30]*y[IDX_CI];
    data[1582] = 0.0 - k[373]*y[IDX_CII];
    data[1583] = 0.0 + k[1109];
    data[1584] = 0.0 - k[14]*y[IDX_CH2I] - k[15]*y[IDX_CHI] - k[16]*y[IDX_H2COI] -
        k[17]*y[IDX_HCOI] - k[18]*y[IDX_MgI] - k[19]*y[IDX_NH3I] -
        k[20]*y[IDX_NOI] - k[21]*y[IDX_SiI] - k[22]*y[IDX_SiC2I] -
        k[23]*y[IDX_SiC3I] - k[24]*y[IDX_SiCI] - k[25]*y[IDX_SiH2I] -
        k[26]*y[IDX_SiH3I] - k[365]*y[IDX_CH3OHI] - k[366]*y[IDX_CH3OHI] -
        k[367]*y[IDX_CO2I] - k[368]*y[IDX_H2COI] - k[369]*y[IDX_H2COI] -
        k[370]*y[IDX_H2OI] - k[371]*y[IDX_H2OI] - k[372]*y[IDX_HCOI] -
        k[373]*y[IDX_NH2I] - k[374]*y[IDX_NH3I] - k[375]*y[IDX_NHI] -
        k[376]*y[IDX_O2I] - k[377]*y[IDX_O2I] - k[378]*y[IDX_OCNI] -
        k[379]*y[IDX_OHI] - k[380]*y[IDX_SiH2I] - k[381]*y[IDX_SiHI] -
        k[382]*y[IDX_SiOI] - k[528]*y[IDX_H2I] - k[1192]*y[IDX_NI] -
        k[1193]*y[IDX_OI] - k[1199]*y[IDX_H2I] - k[1205]*y[IDX_HI] -
        k[1214]*y[IDX_eM] - k[1264];
    data[1585] = 0.0 + k[678]*y[IDX_HeII];
    data[1586] = 0.0 + k[241] + k[614]*y[IDX_HI];
    data[1587] = 0.0 - k[14]*y[IDX_CII] + k[653]*y[IDX_HeII];
    data[1588] = 0.0 - k[375]*y[IDX_CII];
    data[1589] = 0.0 + k[664]*y[IDX_HeII];
    data[1590] = 0.0 - k[16]*y[IDX_CII] - k[368]*y[IDX_CII] - k[369]*y[IDX_CII];
    data[1591] = 0.0 - k[17]*y[IDX_CII] - k[372]*y[IDX_CII];
    data[1592] = 0.0 + k[139]*y[IDX_CI] + k[653]*y[IDX_CH2I] + k[662]*y[IDX_CHI] +
        k[664]*y[IDX_CNI] + k[668]*y[IDX_CO2I] + k[669]*y[IDX_COI] +
        k[678]*y[IDX_HCNI] + k[684]*y[IDX_HNCI] + k[702]*y[IDX_SiCI];
    data[1593] = 0.0 - k[15]*y[IDX_CII] + k[662]*y[IDX_HeII];
    data[1594] = 0.0 - k[20]*y[IDX_CII];
    data[1595] = 0.0 - k[1192]*y[IDX_CII];
    data[1596] = 0.0 - k[379]*y[IDX_CII];
    data[1597] = 0.0 - k[376]*y[IDX_CII] - k[377]*y[IDX_CII];
    data[1598] = 0.0 + k[27]*y[IDX_CNII] + k[28]*y[IDX_COII] + k[29]*y[IDX_N2II] +
        k[30]*y[IDX_O2II] + k[139]*y[IDX_HeII] + k[231] + k[240] + k[1107];
    data[1599] = 0.0 - k[370]*y[IDX_CII] - k[371]*y[IDX_CII];
    data[1600] = 0.0 - k[1193]*y[IDX_CII];
    data[1601] = 0.0 - k[1214]*y[IDX_CII];
    data[1602] = 0.0 + k[669]*y[IDX_HeII];
    data[1603] = 0.0 - k[528]*y[IDX_CII] - k[1199]*y[IDX_CII];
    data[1604] = 0.0 + k[614]*y[IDX_CHII] - k[1205]*y[IDX_CII];
    data[1605] = 0.0 + k[1323] + k[1324] + k[1325] + k[1326];
    data[1606] = 0.0 + k[254] + k[973]*y[IDX_HI] + k[1013]*y[IDX_NI] + k[1135];
    data[1607] = 0.0 + k[993]*y[IDX_HI];
    data[1608] = 0.0 + k[950]*y[IDX_CNI];
    data[1609] = 0.0 + k[944]*y[IDX_CNI];
    data[1610] = 0.0 - k[65]*y[IDX_HCNI] + k[479]*y[IDX_H2COI];
    data[1611] = 0.0 + k[328]*y[IDX_eM] + k[427]*y[IDX_CH2I] + k[464]*y[IDX_CHI] +
        k[633]*y[IDX_H2COI] + k[785]*y[IDX_NH2I];
    data[1612] = 0.0 - k[631]*y[IDX_HCNI];
    data[1613] = 0.0 - k[632]*y[IDX_HCNI];
    data[1614] = 0.0 + k[1]*y[IDX_HII] + k[979]*y[IDX_HI];
    data[1615] = 0.0 - k[630]*y[IDX_HCNI];
    data[1616] = 0.0 - k[135]*y[IDX_HCNI];
    data[1617] = 0.0 - k[628]*y[IDX_HCNI];
    data[1618] = 0.0 + k[920]*y[IDX_CNI];
    data[1619] = 0.0 - k[134]*y[IDX_HCNI];
    data[1620] = 0.0 - k[107]*y[IDX_HCNI];
    data[1621] = 0.0 + k[196]*y[IDX_HCNII] + k[1033]*y[IDX_CNI];
    data[1622] = 0.0 + k[902]*y[IDX_CNI] + k[910]*y[IDX_NOI] + k[1009]*y[IDX_NI] +
        k[1010]*y[IDX_NI];
    data[1623] = 0.0 - k[161]*y[IDX_HCNI];
    data[1624] = 0.0 - k[814]*y[IDX_HCNI] - k[815]*y[IDX_HCNI];
    data[1625] = 0.0 + k[124]*y[IDX_H2OI] + k[129]*y[IDX_HI] + k[132]*y[IDX_NOI] +
        k[133]*y[IDX_O2I] + k[196]*y[IDX_NH3I] - k[623]*y[IDX_HCNI];
    data[1626] = 0.0 - k[773]*y[IDX_HCNI];
    data[1627] = 0.0 - k[758]*y[IDX_HCNI];
    data[1628] = 0.0 + k[785]*y[IDX_HCNHII] + k[866]*y[IDX_CI];
    data[1629] = 0.0 - k[556]*y[IDX_HCNI];
    data[1630] = 0.0 - k[608]*y[IDX_HCNI];
    data[1631] = 0.0 + k[884]*y[IDX_CH2I] + k[927]*y[IDX_CHI];
    data[1632] = 0.0 - k[65]*y[IDX_CNII] - k[81]*y[IDX_HII] - k[107]*y[IDX_H2II] -
        k[134]*y[IDX_COII] - k[135]*y[IDX_N2II] - k[161]*y[IDX_NII] - k[259] -
        k[405]*y[IDX_CHII] - k[556]*y[IDX_H2OII] - k[587]*y[IDX_H3II] -
        k[608]*y[IDX_H3OII] - k[623]*y[IDX_HCNII] - k[627]*y[IDX_H2COII] -
        k[628]*y[IDX_H3COII] - k[629]*y[IDX_HCOII] - k[630]*y[IDX_HNOII] -
        k[631]*y[IDX_N2HII] - k[632]*y[IDX_O2HII] - k[676]*y[IDX_HeII] -
        k[677]*y[IDX_HeII] - k[678]*y[IDX_HeII] - k[679]*y[IDX_HeII] -
        k[758]*y[IDX_NHII] - k[773]*y[IDX_NH2II] - k[814]*y[IDX_OII] -
        k[815]*y[IDX_OII] - k[841]*y[IDX_OHII] - k[976]*y[IDX_HI] -
        k[1063]*y[IDX_OI] - k[1064]*y[IDX_OI] - k[1065]*y[IDX_OI] -
        k[1094]*y[IDX_OHI] - k[1095]*y[IDX_OHI] - k[1147] - k[1266];
    data[1633] = 0.0 - k[405]*y[IDX_HCNI];
    data[1634] = 0.0 + k[427]*y[IDX_HCNHII] + k[880]*y[IDX_CNI] + k[884]*y[IDX_N2I] +
        k[887]*y[IDX_NOI] + k[1005]*y[IDX_NI];
    data[1635] = 0.0 - k[627]*y[IDX_HCNI];
    data[1636] = 0.0 + k[1035]*y[IDX_CNI];
    data[1637] = 0.0 - k[841]*y[IDX_HCNI];
    data[1638] = 0.0 + k[880]*y[IDX_CH2I] + k[902]*y[IDX_CH3I] + k[920]*y[IDX_CH4I] +
        k[942]*y[IDX_H2COI] + k[943]*y[IDX_HCOI] + k[944]*y[IDX_HNOI] +
        k[950]*y[IDX_SiH4I] + k[959]*y[IDX_H2I] + k[1033]*y[IDX_NH3I] +
        k[1035]*y[IDX_NHI] + k[1090]*y[IDX_OHI];
    data[1639] = 0.0 + k[479]*y[IDX_CNII] + k[633]*y[IDX_HCNHII] + k[942]*y[IDX_CNI];
    data[1640] = 0.0 + k[943]*y[IDX_CNI] + k[1015]*y[IDX_NI];
    data[1641] = 0.0 - k[676]*y[IDX_HCNI] - k[677]*y[IDX_HCNI] - k[678]*y[IDX_HCNI] -
        k[679]*y[IDX_HCNI];
    data[1642] = 0.0 + k[464]*y[IDX_HCNHII] + k[927]*y[IDX_N2I] + k[930]*y[IDX_NOI];
    data[1643] = 0.0 - k[587]*y[IDX_HCNI];
    data[1644] = 0.0 + k[132]*y[IDX_HCNII] + k[887]*y[IDX_CH2I] + k[910]*y[IDX_CH3I] +
        k[930]*y[IDX_CHI];
    data[1645] = 0.0 + k[1005]*y[IDX_CH2I] + k[1009]*y[IDX_CH3I] + k[1010]*y[IDX_CH3I] +
        k[1013]*y[IDX_H2CNI] + k[1015]*y[IDX_HCOI];
    data[1646] = 0.0 + k[1090]*y[IDX_CNI] - k[1094]*y[IDX_HCNI] - k[1095]*y[IDX_HCNI];
    data[1647] = 0.0 + k[133]*y[IDX_HCNII];
    data[1648] = 0.0 + k[866]*y[IDX_NH2I];
    data[1649] = 0.0 + k[1]*y[IDX_HNCI] - k[81]*y[IDX_HCNI];
    data[1650] = 0.0 - k[629]*y[IDX_HCNI];
    data[1651] = 0.0 + k[124]*y[IDX_HCNII];
    data[1652] = 0.0 - k[1063]*y[IDX_HCNI] - k[1064]*y[IDX_HCNI] - k[1065]*y[IDX_HCNI];
    data[1653] = 0.0 + k[328]*y[IDX_HCNHII];
    data[1654] = 0.0 + k[959]*y[IDX_CNI];
    data[1655] = 0.0 + k[129]*y[IDX_HCNII] + k[973]*y[IDX_H2CNI] - k[976]*y[IDX_HCNI] +
        k[979]*y[IDX_HNCI] + k[993]*y[IDX_OCNI];
    data[1656] = 0.0 + k[387]*y[IDX_CI];
    data[1657] = 0.0 - k[396]*y[IDX_CHII] - k[397]*y[IDX_CHII];
    data[1658] = 0.0 - k[32]*y[IDX_CHII];
    data[1659] = 0.0 + k[53]*y[IDX_CHI];
    data[1660] = 0.0 + k[389]*y[IDX_CI];
    data[1661] = 0.0 + k[392]*y[IDX_CI];
    data[1662] = 0.0 - k[35]*y[IDX_CHII];
    data[1663] = 0.0 - k[407]*y[IDX_CHII];
    data[1664] = 0.0 + k[388]*y[IDX_CI];
    data[1665] = 0.0 + k[58]*y[IDX_CHI];
    data[1666] = 0.0 + k[658]*y[IDX_HeII];
    data[1667] = 0.0 + k[54]*y[IDX_CHI];
    data[1668] = 0.0 + k[102]*y[IDX_CHI] + k[509]*y[IDX_CI];
    data[1669] = 0.0 - k[33]*y[IDX_CHII];
    data[1670] = 0.0 + k[655]*y[IDX_HeII];
    data[1671] = 0.0 - k[398]*y[IDX_CHII];
    data[1672] = 0.0 + k[57]*y[IDX_CHI];
    data[1673] = 0.0 + k[60]*y[IDX_CHI];
    data[1674] = 0.0 + k[385]*y[IDX_CI];
    data[1675] = 0.0 + k[59]*y[IDX_CHI];
    data[1676] = 0.0 + k[390]*y[IDX_CI];
    data[1677] = 0.0 + k[61]*y[IDX_CHI];
    data[1678] = 0.0 + k[1114];
    data[1679] = 0.0 - k[409]*y[IDX_CHII];
    data[1680] = 0.0 + k[615]*y[IDX_HI] + k[1110];
    data[1681] = 0.0 + k[56]*y[IDX_CHI] + k[383]*y[IDX_CI];
    data[1682] = 0.0 + k[15]*y[IDX_CHI] + k[372]*y[IDX_HCOI] + k[528]*y[IDX_H2I] +
        k[1205]*y[IDX_HI];
    data[1683] = 0.0 - k[405]*y[IDX_CHII] + k[679]*y[IDX_HeII];
    data[1684] = 0.0 - k[31]*y[IDX_HCOI] - k[32]*y[IDX_MgI] - k[33]*y[IDX_NH3I] -
        k[34]*y[IDX_NOI] - k[35]*y[IDX_SiI] - k[241] - k[294]*y[IDX_eM] -
        k[396]*y[IDX_CH3OHI] - k[397]*y[IDX_CH3OHI] - k[398]*y[IDX_CO2I] -
        k[399]*y[IDX_H2COI] - k[400]*y[IDX_H2COI] - k[401]*y[IDX_H2COI] -
        k[402]*y[IDX_H2OI] - k[403]*y[IDX_H2OI] - k[404]*y[IDX_H2OI] -
        k[405]*y[IDX_HCNI] - k[406]*y[IDX_HCOI] - k[407]*y[IDX_HNCI] -
        k[408]*y[IDX_NI] - k[409]*y[IDX_NH2I] - k[410]*y[IDX_NHI] -
        k[411]*y[IDX_O2I] - k[412]*y[IDX_O2I] - k[413]*y[IDX_O2I] -
        k[414]*y[IDX_OI] - k[415]*y[IDX_OHI] - k[529]*y[IDX_H2I] -
        k[614]*y[IDX_HI] - k[1108] - k[1272];
    data[1685] = 0.0 + k[491]*y[IDX_HII] + k[654]*y[IDX_HeII];
    data[1686] = 0.0 + k[55]*y[IDX_CHI];
    data[1687] = 0.0 - k[410]*y[IDX_CHII];
    data[1688] = 0.0 + k[62]*y[IDX_CHI] + k[393]*y[IDX_CI];
    data[1689] = 0.0 - k[399]*y[IDX_CHII] - k[400]*y[IDX_CHII] - k[401]*y[IDX_CHII];
    data[1690] = 0.0 - k[31]*y[IDX_CHII] + k[372]*y[IDX_CII] - k[406]*y[IDX_CHII] +
        k[682]*y[IDX_HeII];
    data[1691] = 0.0 + k[141]*y[IDX_CHI] + k[654]*y[IDX_CH2I] + k[655]*y[IDX_CH3I] +
        k[658]*y[IDX_CH4I] + k[679]*y[IDX_HCNI] + k[682]*y[IDX_HCOI];
    data[1692] = 0.0 + k[15]*y[IDX_CII] + k[53]*y[IDX_CNII] + k[54]*y[IDX_COII] +
        k[55]*y[IDX_H2COII] + k[56]*y[IDX_H2OII] + k[57]*y[IDX_NII] +
        k[58]*y[IDX_N2II] + k[59]*y[IDX_NH2II] + k[60]*y[IDX_OII] +
        k[61]*y[IDX_O2II] + k[62]*y[IDX_OHII] + k[78]*y[IDX_HII] +
        k[102]*y[IDX_H2II] + k[141]*y[IDX_HeII] + k[1129];
    data[1693] = 0.0 + k[576]*y[IDX_CI];
    data[1694] = 0.0 - k[34]*y[IDX_CHII];
    data[1695] = 0.0 - k[408]*y[IDX_CHII];
    data[1696] = 0.0 - k[415]*y[IDX_CHII];
    data[1697] = 0.0 - k[411]*y[IDX_CHII] - k[412]*y[IDX_CHII] - k[413]*y[IDX_CHII];
    data[1698] = 0.0 + k[383]*y[IDX_H2OII] + k[385]*y[IDX_HCNII] + k[386]*y[IDX_HCOII] +
        k[387]*y[IDX_HCO2II] + k[388]*y[IDX_HNOII] + k[389]*y[IDX_N2HII] +
        k[390]*y[IDX_NHII] + k[392]*y[IDX_O2HII] + k[393]*y[IDX_OHII] +
        k[509]*y[IDX_H2II] + k[576]*y[IDX_H3II];
    data[1699] = 0.0 + k[78]*y[IDX_CHI] + k[491]*y[IDX_CH2I];
    data[1700] = 0.0 + k[386]*y[IDX_CI];
    data[1701] = 0.0 - k[402]*y[IDX_CHII] - k[403]*y[IDX_CHII] - k[404]*y[IDX_CHII];
    data[1702] = 0.0 - k[414]*y[IDX_CHII];
    data[1703] = 0.0 - k[294]*y[IDX_CHII];
    data[1704] = 0.0 + k[528]*y[IDX_CII] - k[529]*y[IDX_CHII];
    data[1705] = 0.0 - k[614]*y[IDX_CHII] + k[615]*y[IDX_CH2II] + k[1205]*y[IDX_CII];
    data[1706] = 0.0 + k[938]*y[IDX_CHI];
    data[1707] = 0.0 - k[885]*y[IDX_CH2I];
    data[1708] = 0.0 - k[438]*y[IDX_CH2I];
    data[1709] = 0.0 - k[883]*y[IDX_CH2I] + k[926]*y[IDX_CHI];
    data[1710] = 0.0 + k[397]*y[IDX_CHII];
    data[1711] = 0.0 + k[301]*y[IDX_eM];
    data[1712] = 0.0 - k[37]*y[IDX_CH2I];
    data[1713] = 0.0 - k[427]*y[IDX_CH2I] - k[428]*y[IDX_CH2I];
    data[1714] = 0.0 - k[431]*y[IDX_CH2I];
    data[1715] = 0.0 - k[436]*y[IDX_CH2I];
    data[1716] = 0.0 - k[430]*y[IDX_CH2I];
    data[1717] = 0.0 - k[41]*y[IDX_CH2I];
    data[1718] = 0.0 + k[317]*y[IDX_eM];
    data[1719] = 0.0 + k[249] + k[457]*y[IDX_OHII] - k[879]*y[IDX_CH2I] + k[1124];
    data[1720] = 0.0 - k[38]*y[IDX_CH2I] - k[422]*y[IDX_CH2I];
    data[1721] = 0.0 - k[100]*y[IDX_CH2I] - k[510]*y[IDX_CH2I];
    data[1722] = 0.0 + k[244] + k[901]*y[IDX_CH3I] + k[901]*y[IDX_CH3I] +
        k[902]*y[IDX_CNI] + k[913]*y[IDX_O2I] + k[919]*y[IDX_OHI] +
        k[968]*y[IDX_HI] + k[1116];
    data[1723] = 0.0 - k[155]*y[IDX_CH2I] + k[722]*y[IDX_H2COI];
    data[1724] = 0.0 - k[43]*y[IDX_CH2I];
    data[1725] = 0.0 - k[426]*y[IDX_CH2I];
    data[1726] = 0.0 - k[42]*y[IDX_CH2I] - k[433]*y[IDX_CH2I];
    data[1727] = 0.0 - k[432]*y[IDX_CH2I];
    data[1728] = 0.0 - k[44]*y[IDX_CH2I] - k[435]*y[IDX_CH2I];
    data[1729] = 0.0 + k[298]*y[IDX_eM];
    data[1730] = 0.0 + k[36]*y[IDX_NOI];
    data[1731] = 0.0 - k[40]*y[IDX_CH2I] - k[424]*y[IDX_CH2I];
    data[1732] = 0.0 - k[434]*y[IDX_CH2I];
    data[1733] = 0.0 - k[425]*y[IDX_CH2I];
    data[1734] = 0.0 - k[884]*y[IDX_CH2I];
    data[1735] = 0.0 - k[14]*y[IDX_CH2I];
    data[1736] = 0.0 + k[397]*y[IDX_CH3OHI] + k[401]*y[IDX_H2COI];
    data[1737] = 0.0 - k[14]*y[IDX_CII] - k[37]*y[IDX_CNII] - k[38]*y[IDX_COII] -
        k[39]*y[IDX_H2COII] - k[40]*y[IDX_H2OII] - k[41]*y[IDX_N2II] -
        k[42]*y[IDX_NH2II] - k[43]*y[IDX_OII] - k[44]*y[IDX_O2II] -
        k[45]*y[IDX_OHII] - k[75]*y[IDX_HII] - k[100]*y[IDX_H2II] -
        k[155]*y[IDX_NII] - k[242] - k[243] - k[422]*y[IDX_COII] -
        k[423]*y[IDX_H2COII] - k[424]*y[IDX_H2OII] - k[425]*y[IDX_H3OII] -
        k[426]*y[IDX_HCNII] - k[427]*y[IDX_HCNHII] - k[428]*y[IDX_HCNHII] -
        k[429]*y[IDX_HCOII] - k[430]*y[IDX_HNOII] - k[431]*y[IDX_N2HII] -
        k[432]*y[IDX_NHII] - k[433]*y[IDX_NH2II] - k[434]*y[IDX_NH3II] -
        k[435]*y[IDX_O2II] - k[436]*y[IDX_O2HII] - k[437]*y[IDX_OHII] -
        k[438]*y[IDX_SiOII] - k[491]*y[IDX_HII] - k[510]*y[IDX_H2II] -
        k[577]*y[IDX_H3II] - k[653]*y[IDX_HeII] - k[654]*y[IDX_HeII] -
        k[863]*y[IDX_CI] - k[878]*y[IDX_CH2I] - k[878]*y[IDX_CH2I] -
        k[878]*y[IDX_CH2I] - k[878]*y[IDX_CH2I] - k[879]*y[IDX_CH4I] -
        k[880]*y[IDX_CNI] - k[881]*y[IDX_H2COI] - k[882]*y[IDX_HCOI] -
        k[883]*y[IDX_HNOI] - k[884]*y[IDX_N2I] - k[885]*y[IDX_NO2I] -
        k[886]*y[IDX_NOI] - k[887]*y[IDX_NOI] - k[888]*y[IDX_NOI] -
        k[889]*y[IDX_O2I] - k[890]*y[IDX_O2I] - k[891]*y[IDX_O2I] -
        k[892]*y[IDX_O2I] - k[893]*y[IDX_O2I] - k[894]*y[IDX_OI] -
        k[895]*y[IDX_OI] - k[896]*y[IDX_OI] - k[897]*y[IDX_OI] -
        k[898]*y[IDX_OHI] - k[899]*y[IDX_OHI] - k[900]*y[IDX_OHI] -
        k[956]*y[IDX_H2I] - k[967]*y[IDX_HI] - k[1005]*y[IDX_NI] -
        k[1006]*y[IDX_NI] - k[1007]*y[IDX_NI] - k[1112] - k[1113] - k[1256];
    data[1738] = 0.0 - k[39]*y[IDX_CH2I] + k[306]*y[IDX_eM] - k[423]*y[IDX_CH2I];
    data[1739] = 0.0 - k[45]*y[IDX_CH2I] - k[437]*y[IDX_CH2I] + k[457]*y[IDX_CH4I];
    data[1740] = 0.0 - k[880]*y[IDX_CH2I] + k[902]*y[IDX_CH3I];
    data[1741] = 0.0 + k[401]*y[IDX_CHII] + k[722]*y[IDX_NII] - k[881]*y[IDX_CH2I] +
        k[924]*y[IDX_CHI];
    data[1742] = 0.0 - k[882]*y[IDX_CH2I] + k[925]*y[IDX_CHI] + k[978]*y[IDX_HI];
    data[1743] = 0.0 - k[653]*y[IDX_CH2I] - k[654]*y[IDX_CH2I];
    data[1744] = 0.0 + k[924]*y[IDX_H2COI] + k[925]*y[IDX_HCOI] + k[926]*y[IDX_HNOI] +
        k[938]*y[IDX_O2HI] + k[958]*y[IDX_H2I];
    data[1745] = 0.0 - k[577]*y[IDX_CH2I];
    data[1746] = 0.0 + k[36]*y[IDX_CH2II] - k[886]*y[IDX_CH2I] - k[887]*y[IDX_CH2I] -
        k[888]*y[IDX_CH2I];
    data[1747] = 0.0 - k[1005]*y[IDX_CH2I] - k[1006]*y[IDX_CH2I] - k[1007]*y[IDX_CH2I];
    data[1748] = 0.0 - k[898]*y[IDX_CH2I] - k[899]*y[IDX_CH2I] - k[900]*y[IDX_CH2I] +
        k[919]*y[IDX_CH3I];
    data[1749] = 0.0 - k[889]*y[IDX_CH2I] - k[890]*y[IDX_CH2I] - k[891]*y[IDX_CH2I] -
        k[892]*y[IDX_CH2I] - k[893]*y[IDX_CH2I] + k[913]*y[IDX_CH3I];
    data[1750] = 0.0 - k[863]*y[IDX_CH2I] + k[1200]*y[IDX_H2I];
    data[1751] = 0.0 - k[75]*y[IDX_CH2I] - k[491]*y[IDX_CH2I];
    data[1752] = 0.0 - k[429]*y[IDX_CH2I];
    data[1753] = 0.0 - k[894]*y[IDX_CH2I] - k[895]*y[IDX_CH2I] - k[896]*y[IDX_CH2I] -
        k[897]*y[IDX_CH2I];
    data[1754] = 0.0 + k[298]*y[IDX_CH3II] + k[301]*y[IDX_CH4II] + k[306]*y[IDX_H2COII] +
        k[317]*y[IDX_H3COII];
    data[1755] = 0.0 - k[956]*y[IDX_CH2I] + k[958]*y[IDX_CHI] + k[1200]*y[IDX_CI];
    data[1756] = 0.0 - k[967]*y[IDX_CH2I] + k[968]*y[IDX_CH3I] + k[978]*y[IDX_HCOI];
    data[1757] = 0.0 + k[712]*y[IDX_NII] + k[808]*y[IDX_OII];
    data[1758] = 0.0 - k[148]*y[IDX_H2COII];
    data[1759] = 0.0 + k[49]*y[IDX_H2COI];
    data[1760] = 0.0 + k[64]*y[IDX_H2COI];
    data[1761] = 0.0 + k[643]*y[IDX_HCOI];
    data[1762] = 0.0 + k[645]*y[IDX_HCOI];
    data[1763] = 0.0 - k[228]*y[IDX_H2COII];
    data[1764] = 0.0 - k[646]*y[IDX_H2COII];
    data[1765] = 0.0 + k[642]*y[IDX_HCOI];
    data[1766] = 0.0 + k[170]*y[IDX_H2COI];
    data[1767] = 0.0 - k[452]*y[IDX_H2COII];
    data[1768] = 0.0 + k[70]*y[IDX_H2COI];
    data[1769] = 0.0 + k[105]*y[IDX_H2COI];
    data[1770] = 0.0 - k[194]*y[IDX_H2COII];
    data[1771] = 0.0 + k[416]*y[IDX_CH2II];
    data[1772] = 0.0 + k[159]*y[IDX_H2COI] + k[712]*y[IDX_CH3OHI];
    data[1773] = 0.0 + k[209]*y[IDX_H2COI] + k[808]*y[IDX_CH3OHI];
    data[1774] = 0.0 + k[624]*y[IDX_HCOI];
    data[1775] = 0.0 + k[774]*y[IDX_HCOI];
    data[1776] = 0.0 + k[175]*y[IDX_H2COI] + k[759]*y[IDX_HCOI];
    data[1777] = 0.0 + k[116]*y[IDX_H2COI] + k[435]*y[IDX_CH2I];
    data[1778] = 0.0 + k[443]*y[IDX_OI] + k[445]*y[IDX_OHI];
    data[1779] = 0.0 - k[780]*y[IDX_H2COII];
    data[1780] = 0.0 + k[416]*y[IDX_CO2I];
    data[1781] = 0.0 + k[117]*y[IDX_H2COI] + k[558]*y[IDX_HCOI];
    data[1782] = 0.0 + k[16]*y[IDX_H2COI];
    data[1783] = 0.0 - k[627]*y[IDX_H2COII];
    data[1784] = 0.0 + k[402]*y[IDX_H2OI];
    data[1785] = 0.0 - k[39]*y[IDX_H2COII] - k[423]*y[IDX_H2COII] + k[435]*y[IDX_O2II];
    data[1786] = 0.0 - k[39]*y[IDX_CH2I] - k[55]*y[IDX_CHI] - k[136]*y[IDX_HCOI] -
        k[148]*y[IDX_MgI] - k[194]*y[IDX_NH3I] - k[203]*y[IDX_NOI] -
        k[228]*y[IDX_SiI] - k[306]*y[IDX_eM] - k[307]*y[IDX_eM] -
        k[308]*y[IDX_eM] - k[309]*y[IDX_eM] - k[423]*y[IDX_CH2I] -
        k[452]*y[IDX_CH4I] - k[459]*y[IDX_CHI] - k[548]*y[IDX_H2COI] -
        k[549]*y[IDX_O2I] - k[563]*y[IDX_H2OI] - k[627]*y[IDX_HCNI] -
        k[641]*y[IDX_HCOI] - k[646]*y[IDX_HNCI] - k[780]*y[IDX_NH2I] -
        k[796]*y[IDX_NHI] - k[1217]*y[IDX_eM] - k[1284];
    data[1787] = 0.0 - k[796]*y[IDX_H2COII];
    data[1788] = 0.0 + k[219]*y[IDX_H2COI] + k[843]*y[IDX_HCOI];
    data[1789] = 0.0 + k[16]*y[IDX_CII] + k[49]*y[IDX_CH4II] + k[64]*y[IDX_CNII] +
        k[70]*y[IDX_COII] + k[79]*y[IDX_HII] + k[105]*y[IDX_H2II] +
        k[116]*y[IDX_O2II] + k[117]*y[IDX_H2OII] + k[142]*y[IDX_HeII] +
        k[159]*y[IDX_NII] + k[170]*y[IDX_N2II] + k[175]*y[IDX_NHII] +
        k[209]*y[IDX_OII] + k[219]*y[IDX_OHII] - k[548]*y[IDX_H2COII] + k[1138];
    data[1790] = 0.0 - k[136]*y[IDX_H2COII] + k[558]*y[IDX_H2OII] + k[588]*y[IDX_H3II] +
        k[624]*y[IDX_HCNII] + k[636]*y[IDX_HCOII] - k[641]*y[IDX_H2COII] +
        k[642]*y[IDX_HNOII] + k[643]*y[IDX_N2HII] + k[645]*y[IDX_O2HII] +
        k[759]*y[IDX_NHII] + k[774]*y[IDX_NH2II] + k[843]*y[IDX_OHII];
    data[1791] = 0.0 + k[142]*y[IDX_H2COI];
    data[1792] = 0.0 - k[55]*y[IDX_H2COII] - k[459]*y[IDX_H2COII];
    data[1793] = 0.0 + k[588]*y[IDX_HCOI];
    data[1794] = 0.0 - k[203]*y[IDX_H2COII];
    data[1795] = 0.0 + k[445]*y[IDX_CH3II];
    data[1796] = 0.0 - k[549]*y[IDX_H2COII];
    data[1797] = 0.0 + k[79]*y[IDX_H2COI];
    data[1798] = 0.0 + k[636]*y[IDX_HCOI];
    data[1799] = 0.0 + k[402]*y[IDX_CHII] - k[563]*y[IDX_H2COII];
    data[1800] = 0.0 + k[443]*y[IDX_CH3II];
    data[1801] = 0.0 - k[306]*y[IDX_H2COII] - k[307]*y[IDX_H2COII] - k[308]*y[IDX_H2COII]
        - k[309]*y[IDX_H2COII] - k[1217]*y[IDX_H2COII];
    data[1802] = 0.0 + k[1013]*y[IDX_NI];
    data[1803] = 0.0 + k[263] + k[1152];
    data[1804] = 0.0 + k[1024]*y[IDX_NI];
    data[1805] = 0.0 - k[1041]*y[IDX_NHI];
    data[1806] = 0.0 + k[994]*y[IDX_HI];
    data[1807] = 0.0 + k[951]*y[IDX_COI] + k[982]*y[IDX_HI] + k[1017]*y[IDX_NI] +
        k[1070]*y[IDX_OI];
    data[1808] = 0.0 + k[712]*y[IDX_NII] + k[713]*y[IDX_NII];
    data[1809] = 0.0 - k[199]*y[IDX_NHI] + k[561]*y[IDX_H2OI];
    data[1810] = 0.0 + k[339]*y[IDX_eM] - k[801]*y[IDX_NHI];
    data[1811] = 0.0 - k[805]*y[IDX_NHI];
    data[1812] = 0.0 + k[775]*y[IDX_NH2II];
    data[1813] = 0.0 - k[800]*y[IDX_NHI];
    data[1814] = 0.0 - k[201]*y[IDX_NHI];
    data[1815] = 0.0 - k[1034]*y[IDX_NHI];
    data[1816] = 0.0 - k[200]*y[IDX_NHI] + k[779]*y[IDX_NH2I] - k[795]*y[IDX_NHI];
    data[1817] = 0.0 - k[111]*y[IDX_NHI] - k[523]*y[IDX_NHI];
    data[1818] = 0.0 + k[177]*y[IDX_NHII] + k[273] + k[725]*y[IDX_NII] -
        k[1037]*y[IDX_NHI] + k[1161];
    data[1819] = 0.0 + k[907]*y[IDX_NH2I];
    data[1820] = 0.0 - k[166]*y[IDX_NHI] + k[712]*y[IDX_CH3OHI] + k[713]*y[IDX_CH3OHI] +
        k[721]*y[IDX_H2COI] + k[725]*y[IDX_NH3I] - k[726]*y[IDX_NHI];
    data[1821] = 0.0 - k[202]*y[IDX_NHI] - k[803]*y[IDX_NHI];
    data[1822] = 0.0 - k[798]*y[IDX_NHI];
    data[1823] = 0.0 + k[342]*y[IDX_eM] + k[433]*y[IDX_CH2I] + k[471]*y[IDX_CHI] +
        k[769]*y[IDX_H2COI] + k[771]*y[IDX_H2OI] + k[773]*y[IDX_HCNI] +
        k[774]*y[IDX_HCOI] + k[775]*y[IDX_HNCI] + k[776]*y[IDX_NH2I] -
        k[802]*y[IDX_NHI];
    data[1824] = 0.0 + k[175]*y[IDX_H2COI] + k[176]*y[IDX_H2OI] + k[177]*y[IDX_NH3I] +
        k[178]*y[IDX_NOI] + k[179]*y[IDX_O2I] - k[763]*y[IDX_NHI];
    data[1825] = 0.0 - k[804]*y[IDX_NHI];
    data[1826] = 0.0 - k[794]*y[IDX_NHI];
    data[1827] = 0.0 + k[270] + k[776]*y[IDX_NH2II] + k[779]*y[IDX_COII] +
        k[868]*y[IDX_CI] + k[907]*y[IDX_CH3I] + k[983]*y[IDX_HI] +
        k[1031]*y[IDX_OHI] + k[1073]*y[IDX_OI] + k[1158];
    data[1828] = 0.0 - k[797]*y[IDX_NHI];
    data[1829] = 0.0 + k[344]*y[IDX_eM];
    data[1830] = 0.0 + k[884]*y[IDX_CH2I];
    data[1831] = 0.0 - k[375]*y[IDX_NHI];
    data[1832] = 0.0 + k[773]*y[IDX_NH2II] + k[1064]*y[IDX_OI];
    data[1833] = 0.0 - k[410]*y[IDX_NHI];
    data[1834] = 0.0 + k[433]*y[IDX_NH2II] + k[884]*y[IDX_N2I] + k[1007]*y[IDX_NI];
    data[1835] = 0.0 - k[796]*y[IDX_NHI];
    data[1836] = 0.0 - k[86]*y[IDX_HII] - k[111]*y[IDX_H2II] - k[166]*y[IDX_NII] -
        k[199]*y[IDX_CNII] - k[200]*y[IDX_COII] - k[201]*y[IDX_N2II] -
        k[202]*y[IDX_OII] - k[274] - k[275] - k[375]*y[IDX_CII] -
        k[410]*y[IDX_CHII] - k[523]*y[IDX_H2II] - k[594]*y[IDX_H3II] -
        k[693]*y[IDX_HeII] - k[726]*y[IDX_NII] - k[763]*y[IDX_NHII] -
        k[794]*y[IDX_CH3II] - k[795]*y[IDX_COII] - k[796]*y[IDX_H2COII] -
        k[797]*y[IDX_H2OII] - k[798]*y[IDX_HCNII] - k[799]*y[IDX_HCOII] -
        k[800]*y[IDX_HNOII] - k[801]*y[IDX_N2HII] - k[802]*y[IDX_NH2II] -
        k[803]*y[IDX_OII] - k[804]*y[IDX_O2II] - k[805]*y[IDX_O2HII] -
        k[806]*y[IDX_OHII] - k[869]*y[IDX_CI] - k[870]*y[IDX_CI] -
        k[962]*y[IDX_H2I] - k[985]*y[IDX_HI] - k[1018]*y[IDX_NI] -
        k[1034]*y[IDX_CH4I] - k[1035]*y[IDX_CNI] - k[1036]*y[IDX_H2OI] -
        k[1037]*y[IDX_NH3I] - k[1038]*y[IDX_NHI] - k[1038]*y[IDX_NHI] -
        k[1038]*y[IDX_NHI] - k[1038]*y[IDX_NHI] - k[1039]*y[IDX_NHI] -
        k[1039]*y[IDX_NHI] - k[1039]*y[IDX_NHI] - k[1039]*y[IDX_NHI] -
        k[1040]*y[IDX_NHI] - k[1040]*y[IDX_NHI] - k[1040]*y[IDX_NHI] -
        k[1040]*y[IDX_NHI] - k[1041]*y[IDX_NO2I] - k[1042]*y[IDX_NOI] -
        k[1043]*y[IDX_NOI] - k[1044]*y[IDX_O2I] - k[1045]*y[IDX_O2I] -
        k[1046]*y[IDX_OI] - k[1047]*y[IDX_OI] - k[1048]*y[IDX_OHI] -
        k[1049]*y[IDX_OHI] - k[1050]*y[IDX_OHI] - k[1162] - k[1163] - k[1265];
    data[1837] = 0.0 - k[806]*y[IDX_NHI];
    data[1838] = 0.0 - k[1035]*y[IDX_NHI];
    data[1839] = 0.0 + k[175]*y[IDX_NHII] + k[721]*y[IDX_NII] + k[769]*y[IDX_NH2II];
    data[1840] = 0.0 + k[774]*y[IDX_NH2II] + k[1014]*y[IDX_NI];
    data[1841] = 0.0 - k[693]*y[IDX_NHI];
    data[1842] = 0.0 + k[471]*y[IDX_NH2II] + k[929]*y[IDX_NI];
    data[1843] = 0.0 - k[594]*y[IDX_NHI];
    data[1844] = 0.0 + k[178]*y[IDX_NHII] + k[987]*y[IDX_HI] - k[1042]*y[IDX_NHI] -
        k[1043]*y[IDX_NHI];
    data[1845] = 0.0 + k[929]*y[IDX_CHI] + k[960]*y[IDX_H2I] + k[1007]*y[IDX_CH2I] +
        k[1013]*y[IDX_H2CNI] + k[1014]*y[IDX_HCOI] + k[1017]*y[IDX_HNOI] -
        k[1018]*y[IDX_NHI] + k[1024]*y[IDX_O2HI] + k[1026]*y[IDX_OHI];
    data[1846] = 0.0 + k[1026]*y[IDX_NI] + k[1031]*y[IDX_NH2I] - k[1048]*y[IDX_NHI] -
        k[1049]*y[IDX_NHI] - k[1050]*y[IDX_NHI];
    data[1847] = 0.0 + k[179]*y[IDX_NHII] - k[1044]*y[IDX_NHI] - k[1045]*y[IDX_NHI];
    data[1848] = 0.0 + k[868]*y[IDX_NH2I] - k[869]*y[IDX_NHI] - k[870]*y[IDX_NHI];
    data[1849] = 0.0 - k[86]*y[IDX_NHI];
    data[1850] = 0.0 - k[799]*y[IDX_NHI];
    data[1851] = 0.0 + k[176]*y[IDX_NHII] + k[561]*y[IDX_CNII] + k[771]*y[IDX_NH2II] -
        k[1036]*y[IDX_NHI];
    data[1852] = 0.0 - k[1046]*y[IDX_NHI] - k[1047]*y[IDX_NHI] + k[1064]*y[IDX_HCNI] +
        k[1070]*y[IDX_HNOI] + k[1073]*y[IDX_NH2I];
    data[1853] = 0.0 + k[339]*y[IDX_N2HII] + k[342]*y[IDX_NH2II] + k[344]*y[IDX_NH3II];
    data[1854] = 0.0 + k[951]*y[IDX_HNOI];
    data[1855] = 0.0 + k[960]*y[IDX_NI] - k[962]*y[IDX_NHI];
    data[1856] = 0.0 + k[982]*y[IDX_HNOI] + k[983]*y[IDX_NH2I] - k[985]*y[IDX_NHI] +
        k[987]*y[IDX_NOI] + k[994]*y[IDX_OCNI];
    data[1857] = 0.0 - k[849]*y[IDX_OHII];
    data[1858] = 0.0 + k[656]*y[IDX_HeII];
    data[1859] = 0.0 - k[850]*y[IDX_OHII];
    data[1860] = 0.0 + k[225]*y[IDX_OHI];
    data[1861] = 0.0 + k[826]*y[IDX_OI];
    data[1862] = 0.0 + k[829]*y[IDX_OI];
    data[1863] = 0.0 - k[848]*y[IDX_OHII];
    data[1864] = 0.0 - k[844]*y[IDX_OHII];
    data[1865] = 0.0 + k[227]*y[IDX_OHI];
    data[1866] = 0.0 - k[457]*y[IDX_OHII];
    data[1867] = 0.0 + k[226]*y[IDX_OHI];
    data[1868] = 0.0 + k[114]*y[IDX_OHI] + k[526]*y[IDX_OI];
    data[1869] = 0.0 - k[222]*y[IDX_OHII];
    data[1870] = 0.0 - k[837]*y[IDX_OHII];
    data[1871] = 0.0 + k[169]*y[IDX_OHI];
    data[1872] = 0.0 + k[215]*y[IDX_OHI] + k[543]*y[IDX_H2I] + k[816]*y[IDX_HCOI];
    data[1873] = 0.0 + k[767]*y[IDX_OI];
    data[1874] = 0.0 - k[188]*y[IDX_OHII] - k[791]*y[IDX_OHII];
    data[1875] = 0.0 + k[1140];
    data[1876] = 0.0 - k[845]*y[IDX_OHII];
    data[1877] = 0.0 - k[841]*y[IDX_OHII];
    data[1878] = 0.0 - k[45]*y[IDX_OHII] - k[437]*y[IDX_OHII];
    data[1879] = 0.0 - k[806]*y[IDX_OHII];
    data[1880] = 0.0 - k[45]*y[IDX_CH2I] - k[62]*y[IDX_CHI] - k[188]*y[IDX_NH2I] -
        k[219]*y[IDX_H2COI] - k[220]*y[IDX_H2OI] - k[221]*y[IDX_HCOI] -
        k[222]*y[IDX_NH3I] - k[223]*y[IDX_NOI] - k[224]*y[IDX_O2I] -
        k[348]*y[IDX_eM] - k[393]*y[IDX_CI] - k[437]*y[IDX_CH2I] -
        k[457]*y[IDX_CH4I] - k[475]*y[IDX_CHI] - k[545]*y[IDX_H2I] -
        k[743]*y[IDX_NI] - k[791]*y[IDX_NH2I] - k[806]*y[IDX_NHI] -
        k[830]*y[IDX_OI] - k[836]*y[IDX_CNI] - k[837]*y[IDX_CO2I] -
        k[838]*y[IDX_COI] - k[839]*y[IDX_H2COI] - k[840]*y[IDX_H2OI] -
        k[841]*y[IDX_HCNI] - k[842]*y[IDX_HCOI] - k[843]*y[IDX_HCOI] -
        k[844]*y[IDX_HNCI] - k[845]*y[IDX_N2I] - k[846]*y[IDX_NOI] -
        k[847]*y[IDX_OHI] - k[848]*y[IDX_SiI] - k[849]*y[IDX_SiHI] -
        k[850]*y[IDX_SiOI] - k[1173] - k[1274];
    data[1881] = 0.0 - k[836]*y[IDX_OHII];
    data[1882] = 0.0 - k[219]*y[IDX_OHII] - k[839]*y[IDX_OHII];
    data[1883] = 0.0 - k[221]*y[IDX_OHII] + k[816]*y[IDX_OII] - k[842]*y[IDX_OHII] -
        k[843]*y[IDX_OHII];
    data[1884] = 0.0 + k[656]*y[IDX_CH3OHI] + k[673]*y[IDX_H2OI];
    data[1885] = 0.0 - k[62]*y[IDX_OHII] - k[475]*y[IDX_OHII];
    data[1886] = 0.0 + k[599]*y[IDX_OI];
    data[1887] = 0.0 - k[223]*y[IDX_OHII] - k[846]*y[IDX_OHII];
    data[1888] = 0.0 - k[743]*y[IDX_OHII];
    data[1889] = 0.0 + k[90]*y[IDX_HII] + k[114]*y[IDX_H2II] + k[169]*y[IDX_NII] +
        k[215]*y[IDX_OII] + k[225]*y[IDX_CNII] + k[226]*y[IDX_COII] +
        k[227]*y[IDX_N2II] - k[847]*y[IDX_OHII] + k[1175];
    data[1890] = 0.0 - k[224]*y[IDX_OHII];
    data[1891] = 0.0 - k[393]*y[IDX_OHII];
    data[1892] = 0.0 + k[90]*y[IDX_OHI];
    data[1893] = 0.0 - k[220]*y[IDX_OHII] + k[673]*y[IDX_HeII] - k[840]*y[IDX_OHII];
    data[1894] = 0.0 + k[526]*y[IDX_H2II] + k[599]*y[IDX_H3II] + k[767]*y[IDX_NHII] +
        k[826]*y[IDX_N2HII] + k[829]*y[IDX_O2HII] - k[830]*y[IDX_OHII];
    data[1895] = 0.0 - k[348]*y[IDX_OHII];
    data[1896] = 0.0 - k[838]*y[IDX_OHII];
    data[1897] = 0.0 + k[543]*y[IDX_OII] - k[545]*y[IDX_OHII];
    data[1898] = 0.0 + k[744]*y[IDX_NI];
    data[1899] = 0.0 + k[1027]*y[IDX_NI];
    data[1900] = 0.0 - k[945]*y[IDX_CNI];
    data[1901] = 0.0 + k[283] + k[378]*y[IDX_CII] + k[698]*y[IDX_HeII] + k[874]*y[IDX_CI]
        + k[995]*y[IDX_HI] + k[1079]*y[IDX_OI] + k[1172];
    data[1902] = 0.0 - k[950]*y[IDX_CNI];
    data[1903] = 0.0 - k[944]*y[IDX_CNI];
    data[1904] = 0.0 + k[27]*y[IDX_CI] + k[37]*y[IDX_CH2I] + k[53]*y[IDX_CHI] +
        k[63]*y[IDX_COI] + k[64]*y[IDX_H2COI] + k[65]*y[IDX_HCNI] +
        k[66]*y[IDX_HCOI] + k[67]*y[IDX_NOI] + k[68]*y[IDX_O2I] +
        k[126]*y[IDX_HI] + k[183]*y[IDX_NH2I] + k[199]*y[IDX_NHI] +
        k[216]*y[IDX_OI] + k[225]*y[IDX_OHI];
    data[1905] = 0.0 + k[327]*y[IDX_eM];
    data[1906] = 0.0 - k[483]*y[IDX_CNI];
    data[1907] = 0.0 + k[262] + k[626]*y[IDX_HCNII] + k[1151];
    data[1908] = 0.0 - k[482]*y[IDX_CNI];
    data[1909] = 0.0 - k[69]*y[IDX_CNI];
    data[1910] = 0.0 - k[920]*y[IDX_CNI];
    data[1911] = 0.0 - k[103]*y[IDX_CNI] - k[513]*y[IDX_CNI];
    data[1912] = 0.0 - k[1033]*y[IDX_CNI];
    data[1913] = 0.0 - k[902]*y[IDX_CNI];
    data[1914] = 0.0 + k[620]*y[IDX_HCNII];
    data[1915] = 0.0 - k[157]*y[IDX_CNI];
    data[1916] = 0.0 - k[811]*y[IDX_CNI];
    data[1917] = 0.0 + k[326]*y[IDX_eM] + k[385]*y[IDX_CI] + k[426]*y[IDX_CH2I] +
        k[463]*y[IDX_CHI] + k[565]*y[IDX_H2OI] + k[620]*y[IDX_CO2I] +
        k[621]*y[IDX_COI] + k[622]*y[IDX_H2COI] + k[623]*y[IDX_HCNI] +
        k[624]*y[IDX_HCOI] + k[626]*y[IDX_HNCI] + k[784]*y[IDX_NH2I] +
        k[798]*y[IDX_NHI] + k[853]*y[IDX_OHI];
    data[1918] = 0.0 - k[747]*y[IDX_CNI];
    data[1919] = 0.0 + k[183]*y[IDX_CNII] + k[784]*y[IDX_HCNII];
    data[1920] = 0.0 + k[865]*y[IDX_CI];
    data[1921] = 0.0 + k[378]*y[IDX_OCNI];
    data[1922] = 0.0 + k[65]*y[IDX_CNII] + k[259] + k[623]*y[IDX_HCNII] +
        k[976]*y[IDX_HI] + k[1063]*y[IDX_OI] + k[1094]*y[IDX_OHI] + k[1147];
    data[1923] = 0.0 + k[37]*y[IDX_CNII] + k[426]*y[IDX_HCNII] - k[880]*y[IDX_CNI];
    data[1924] = 0.0 + k[199]*y[IDX_CNII] + k[798]*y[IDX_HCNII] + k[869]*y[IDX_CI] -
        k[1035]*y[IDX_CNI];
    data[1925] = 0.0 - k[836]*y[IDX_CNI];
    data[1926] = 0.0 - k[69]*y[IDX_N2II] - k[103]*y[IDX_H2II] - k[157]*y[IDX_NII] -
        k[251] - k[482]*y[IDX_HNOII] - k[483]*y[IDX_O2HII] - k[513]*y[IDX_H2II]
        - k[581]*y[IDX_H3II] - k[663]*y[IDX_HeII] - k[664]*y[IDX_HeII] -
        k[747]*y[IDX_NHII] - k[811]*y[IDX_OII] - k[836]*y[IDX_OHII] -
        k[880]*y[IDX_CH2I] - k[902]*y[IDX_CH3I] - k[920]*y[IDX_CH4I] -
        k[942]*y[IDX_H2COI] - k[943]*y[IDX_HCOI] - k[944]*y[IDX_HNOI] -
        k[945]*y[IDX_NO2I] - k[946]*y[IDX_NOI] - k[947]*y[IDX_NOI] -
        k[948]*y[IDX_O2I] - k[949]*y[IDX_O2I] - k[950]*y[IDX_SiH4I] -
        k[959]*y[IDX_H2I] - k[1011]*y[IDX_NI] - k[1033]*y[IDX_NH3I] -
        k[1035]*y[IDX_NHI] - k[1057]*y[IDX_OI] - k[1058]*y[IDX_OI] -
        k[1090]*y[IDX_OHI] - k[1091]*y[IDX_OHI] - k[1130] - k[1263];
    data[1927] = 0.0 + k[64]*y[IDX_CNII] + k[622]*y[IDX_HCNII] - k[942]*y[IDX_CNI];
    data[1928] = 0.0 + k[66]*y[IDX_CNII] + k[624]*y[IDX_HCNII] - k[943]*y[IDX_CNI];
    data[1929] = 0.0 - k[663]*y[IDX_CNI] - k[664]*y[IDX_CNI] + k[698]*y[IDX_OCNI];
    data[1930] = 0.0 + k[53]*y[IDX_CNII] + k[463]*y[IDX_HCNII] + k[928]*y[IDX_NI];
    data[1931] = 0.0 - k[581]*y[IDX_CNI];
    data[1932] = 0.0 + k[67]*y[IDX_CNII] + k[871]*y[IDX_CI] - k[946]*y[IDX_CNI] -
        k[947]*y[IDX_CNI];
    data[1933] = 0.0 + k[744]*y[IDX_SiCII] + k[928]*y[IDX_CHI] - k[1011]*y[IDX_CNI] +
        k[1027]*y[IDX_SiCI] + k[1194]*y[IDX_CI];
    data[1934] = 0.0 + k[225]*y[IDX_CNII] + k[853]*y[IDX_HCNII] - k[1090]*y[IDX_CNI] -
        k[1091]*y[IDX_CNI] + k[1094]*y[IDX_HCNI];
    data[1935] = 0.0 + k[68]*y[IDX_CNII] - k[948]*y[IDX_CNI] - k[949]*y[IDX_CNI];
    data[1936] = 0.0 + k[27]*y[IDX_CNII] + k[385]*y[IDX_HCNII] + k[865]*y[IDX_N2I] +
        k[869]*y[IDX_NHI] + k[871]*y[IDX_NOI] + k[874]*y[IDX_OCNI] +
        k[1194]*y[IDX_NI];
    data[1937] = 0.0 + k[565]*y[IDX_HCNII];
    data[1938] = 0.0 + k[216]*y[IDX_CNII] - k[1057]*y[IDX_CNI] - k[1058]*y[IDX_CNI] +
        k[1063]*y[IDX_HCNI] + k[1079]*y[IDX_OCNI];
    data[1939] = 0.0 + k[326]*y[IDX_HCNII] + k[327]*y[IDX_HCNHII];
    data[1940] = 0.0 + k[63]*y[IDX_CNII] + k[621]*y[IDX_HCNII];
    data[1941] = 0.0 - k[959]*y[IDX_CNI];
    data[1942] = 0.0 + k[126]*y[IDX_CNII] + k[976]*y[IDX_HCNI] + k[995]*y[IDX_OCNI];
    data[1943] = 0.0 + k[1347] + k[1348] + k[1349] + k[1350];
    data[1944] = 0.0 + k[1003]*y[IDX_HCOI];
    data[1945] = 0.0 + k[885]*y[IDX_CH2I] + k[909]*y[IDX_CH3I];
    data[1946] = 0.0 + k[438]*y[IDX_CH2I];
    data[1947] = 0.0 + k[999]*y[IDX_HCOI];
    data[1948] = 0.0 + k[247] + k[396]*y[IDX_CHII] + k[1119];
    data[1949] = 0.0 + k[148]*y[IDX_H2COII];
    data[1950] = 0.0 - k[49]*y[IDX_H2COI] - k[449]*y[IDX_H2COI];
    data[1951] = 0.0 - k[64]*y[IDX_H2COI] - k[479]*y[IDX_H2COI];
    data[1952] = 0.0 - k[633]*y[IDX_H2COI] - k[634]*y[IDX_H2COI];
    data[1953] = 0.0 - k[735]*y[IDX_H2COI];
    data[1954] = 0.0 - k[552]*y[IDX_H2COI];
    data[1955] = 0.0 + k[228]*y[IDX_H2COII];
    data[1956] = 0.0 + k[647]*y[IDX_H3COII];
    data[1957] = 0.0 - k[550]*y[IDX_H2COI];
    data[1958] = 0.0 - k[170]*y[IDX_H2COI] - k[730]*y[IDX_H2COI];
    data[1959] = 0.0 + k[320]*y[IDX_eM] + k[461]*y[IDX_CHI] + k[564]*y[IDX_H2OI] +
        k[628]*y[IDX_HCNI] + k[647]*y[IDX_HNCI] + k[782]*y[IDX_NH2I];
    data[1960] = 0.0 - k[70]*y[IDX_H2COI] - k[484]*y[IDX_H2COI];
    data[1961] = 0.0 - k[105]*y[IDX_H2COI] - k[517]*y[IDX_H2COI];
    data[1962] = 0.0 + k[194]*y[IDX_H2COII];
    data[1963] = 0.0 - k[903]*y[IDX_H2COI] + k[909]*y[IDX_NO2I] + k[911]*y[IDX_O2I] +
        k[916]*y[IDX_OI] + k[918]*y[IDX_OHI];
    data[1964] = 0.0 - k[159]*y[IDX_H2COI] - k[721]*y[IDX_H2COI] - k[722]*y[IDX_H2COI];
    data[1965] = 0.0 - k[209]*y[IDX_H2COI] - k[813]*y[IDX_H2COI];
    data[1966] = 0.0 - k[622]*y[IDX_H2COI];
    data[1967] = 0.0 - k[769]*y[IDX_H2COI] - k[770]*y[IDX_H2COI];
    data[1968] = 0.0 - k[175]*y[IDX_H2COI] - k[752]*y[IDX_H2COI] - k[753]*y[IDX_H2COI];
    data[1969] = 0.0 - k[116]*y[IDX_H2COI] - k[551]*y[IDX_H2COI];
    data[1970] = 0.0 - k[440]*y[IDX_H2COI];
    data[1971] = 0.0 + k[782]*y[IDX_H3COII];
    data[1972] = 0.0 - k[417]*y[IDX_H2COI];
    data[1973] = 0.0 - k[117]*y[IDX_H2COI] - k[554]*y[IDX_H2COI];
    data[1974] = 0.0 - k[607]*y[IDX_H2COI];
    data[1975] = 0.0 - k[16]*y[IDX_H2COI] - k[368]*y[IDX_H2COI] - k[369]*y[IDX_H2COI];
    data[1976] = 0.0 + k[628]*y[IDX_H3COII];
    data[1977] = 0.0 + k[396]*y[IDX_CH3OHI] - k[399]*y[IDX_H2COI] - k[400]*y[IDX_H2COI] -
        k[401]*y[IDX_H2COI];
    data[1978] = 0.0 + k[39]*y[IDX_H2COII] + k[438]*y[IDX_SiOII] - k[881]*y[IDX_H2COI] +
        k[885]*y[IDX_NO2I] + k[886]*y[IDX_NOI] + k[892]*y[IDX_O2I] +
        k[898]*y[IDX_OHI];
    data[1979] = 0.0 + k[39]*y[IDX_CH2I] + k[55]*y[IDX_CHI] + k[136]*y[IDX_HCOI] +
        k[148]*y[IDX_MgI] + k[194]*y[IDX_NH3I] + k[203]*y[IDX_NOI] +
        k[228]*y[IDX_SiI] - k[548]*y[IDX_H2COI] + k[1217]*y[IDX_eM];
    data[1980] = 0.0 - k[219]*y[IDX_H2COI] - k[839]*y[IDX_H2COI];
    data[1981] = 0.0 - k[942]*y[IDX_H2COI];
    data[1982] = 0.0 - k[16]*y[IDX_CII] - k[49]*y[IDX_CH4II] - k[64]*y[IDX_CNII] -
        k[70]*y[IDX_COII] - k[79]*y[IDX_HII] - k[105]*y[IDX_H2II] -
        k[116]*y[IDX_O2II] - k[117]*y[IDX_H2OII] - k[142]*y[IDX_HeII] -
        k[159]*y[IDX_NII] - k[170]*y[IDX_N2II] - k[175]*y[IDX_NHII] -
        k[209]*y[IDX_OII] - k[219]*y[IDX_OHII] - k[255] - k[368]*y[IDX_CII] -
        k[369]*y[IDX_CII] - k[399]*y[IDX_CHII] - k[400]*y[IDX_CHII] -
        k[401]*y[IDX_CHII] - k[417]*y[IDX_CH2II] - k[440]*y[IDX_CH3II] -
        k[449]*y[IDX_CH4II] - k[479]*y[IDX_CNII] - k[484]*y[IDX_COII] -
        k[497]*y[IDX_HII] - k[498]*y[IDX_HII] - k[517]*y[IDX_H2II] -
        k[548]*y[IDX_H2COII] - k[550]*y[IDX_HNOII] - k[551]*y[IDX_O2II] -
        k[552]*y[IDX_O2HII] - k[554]*y[IDX_H2OII] - k[585]*y[IDX_H3II] -
        k[607]*y[IDX_H3OII] - k[622]*y[IDX_HCNII] - k[633]*y[IDX_HCNHII] -
        k[634]*y[IDX_HCNHII] - k[635]*y[IDX_HCOII] - k[670]*y[IDX_HeII] -
        k[671]*y[IDX_HeII] - k[672]*y[IDX_HeII] - k[721]*y[IDX_NII] -
        k[722]*y[IDX_NII] - k[730]*y[IDX_N2II] - k[735]*y[IDX_N2HII] -
        k[752]*y[IDX_NHII] - k[753]*y[IDX_NHII] - k[769]*y[IDX_NH2II] -
        k[770]*y[IDX_NH2II] - k[813]*y[IDX_OII] - k[839]*y[IDX_OHII] -
        k[881]*y[IDX_CH2I] - k[903]*y[IDX_CH3I] - k[924]*y[IDX_CHI] -
        k[942]*y[IDX_CNI] - k[974]*y[IDX_HI] - k[1061]*y[IDX_OI] -
        k[1093]*y[IDX_OHI] - k[1136] - k[1137] - k[1138] - k[1139] - k[1252];
    data[1983] = 0.0 + k[136]*y[IDX_H2COII] + k[998]*y[IDX_HCOI] + k[998]*y[IDX_HCOI] +
        k[999]*y[IDX_HNOI] + k[1003]*y[IDX_O2HI];
    data[1984] = 0.0 - k[142]*y[IDX_H2COI] - k[670]*y[IDX_H2COI] - k[671]*y[IDX_H2COI] -
        k[672]*y[IDX_H2COI];
    data[1985] = 0.0 + k[55]*y[IDX_H2COII] + k[461]*y[IDX_H3COII] - k[924]*y[IDX_H2COI];
    data[1986] = 0.0 - k[585]*y[IDX_H2COI];
    data[1987] = 0.0 + k[203]*y[IDX_H2COII] + k[886]*y[IDX_CH2I];
    data[1988] = 0.0 + k[898]*y[IDX_CH2I] + k[918]*y[IDX_CH3I] - k[1093]*y[IDX_H2COI];
    data[1989] = 0.0 + k[892]*y[IDX_CH2I] + k[911]*y[IDX_CH3I];
    data[1990] = 0.0 - k[79]*y[IDX_H2COI] - k[497]*y[IDX_H2COI] - k[498]*y[IDX_H2COI];
    data[1991] = 0.0 - k[635]*y[IDX_H2COI];
    data[1992] = 0.0 + k[564]*y[IDX_H3COII];
    data[1993] = 0.0 + k[916]*y[IDX_CH3I] - k[1061]*y[IDX_H2COI];
    data[1994] = 0.0 + k[320]*y[IDX_H3COII] + k[1217]*y[IDX_H2COII];
    data[1995] = 0.0 - k[974]*y[IDX_H2COI];
    data[1996] = 0.0 + k[937]*y[IDX_CHI] - k[1003]*y[IDX_HCOI];
    data[1997] = 0.0 - k[138]*y[IDX_HCOI];
    data[1998] = 0.0 - k[999]*y[IDX_HCOI];
    data[1999] = 0.0 + k[366]*y[IDX_CII];
    data[2000] = 0.0 + k[149]*y[IDX_HCOII];
    data[2001] = 0.0 - k[66]*y[IDX_HCOI] - k[480]*y[IDX_HCOI];
    data[2002] = 0.0 - k[643]*y[IDX_HCOI];
    data[2003] = 0.0 - k[645]*y[IDX_HCOI];
    data[2004] = 0.0 + k[646]*y[IDX_H2COII];
    data[2005] = 0.0 - k[642]*y[IDX_HCOI];
    data[2006] = 0.0 - k[171]*y[IDX_HCOI] - k[731]*y[IDX_HCOI];
    data[2007] = 0.0 + k[321]*y[IDX_eM];
    data[2008] = 0.0 - k[71]*y[IDX_HCOI] + k[484]*y[IDX_H2COI];
    data[2009] = 0.0 - k[108]*y[IDX_HCOI] - k[519]*y[IDX_HCOI];
    data[2010] = 0.0 + k[903]*y[IDX_H2COI] - k[905]*y[IDX_HCOI] + k[912]*y[IDX_O2I];
    data[2011] = 0.0 + k[750]*y[IDX_NHII] + k[923]*y[IDX_CHI];
    data[2012] = 0.0 - k[162]*y[IDX_HCOI] - k[723]*y[IDX_HCOI];
    data[2013] = 0.0 - k[211]*y[IDX_HCOI] - k[816]*y[IDX_HCOI];
    data[2014] = 0.0 - k[624]*y[IDX_HCOI] - k[625]*y[IDX_HCOI];
    data[2015] = 0.0 - k[180]*y[IDX_HCOI] + k[770]*y[IDX_H2COI] - k[774]*y[IDX_HCOI];
    data[2016] = 0.0 + k[750]*y[IDX_CO2I] - k[759]*y[IDX_HCOI];
    data[2017] = 0.0 - k[137]*y[IDX_HCOI] - k[644]*y[IDX_HCOI];
    data[2018] = 0.0 - k[46]*y[IDX_HCOI] - k[441]*y[IDX_HCOI];
    data[2019] = 0.0 + k[780]*y[IDX_H2COII];
    data[2020] = 0.0 - k[419]*y[IDX_HCOI];
    data[2021] = 0.0 - k[118]*y[IDX_HCOI] - k[557]*y[IDX_HCOI] - k[558]*y[IDX_HCOI];
    data[2022] = 0.0 - k[189]*y[IDX_HCOI];
    data[2023] = 0.0 - k[17]*y[IDX_HCOI] + k[366]*y[IDX_CH3OHI] - k[372]*y[IDX_HCOI];
    data[2024] = 0.0 + k[627]*y[IDX_H2COII];
    data[2025] = 0.0 - k[31]*y[IDX_HCOI] - k[406]*y[IDX_HCOI] + k[413]*y[IDX_O2I];
    data[2026] = 0.0 + k[423]*y[IDX_H2COII] + k[881]*y[IDX_H2COI] - k[882]*y[IDX_HCOI] +
        k[893]*y[IDX_O2I] + k[896]*y[IDX_OI];
    data[2027] = 0.0 - k[136]*y[IDX_HCOI] + k[309]*y[IDX_eM] + k[423]*y[IDX_CH2I] +
        k[459]*y[IDX_CHI] + k[548]*y[IDX_H2COI] + k[563]*y[IDX_H2OI] +
        k[627]*y[IDX_HCNI] - k[641]*y[IDX_HCOI] + k[646]*y[IDX_HNCI] +
        k[780]*y[IDX_NH2I];
    data[2028] = 0.0 - k[221]*y[IDX_HCOI] - k[842]*y[IDX_HCOI] - k[843]*y[IDX_HCOI];
    data[2029] = 0.0 + k[942]*y[IDX_H2COI] - k[943]*y[IDX_HCOI];
    data[2030] = 0.0 + k[484]*y[IDX_COII] + k[548]*y[IDX_H2COII] + k[770]*y[IDX_NH2II] +
        k[881]*y[IDX_CH2I] + k[903]*y[IDX_CH3I] + k[924]*y[IDX_CHI] +
        k[942]*y[IDX_CNI] + k[974]*y[IDX_HI] + k[1061]*y[IDX_OI] +
        k[1093]*y[IDX_OHI];
    data[2031] = 0.0 - k[17]*y[IDX_CII] - k[31]*y[IDX_CHII] - k[46]*y[IDX_CH3II] -
        k[66]*y[IDX_CNII] - k[71]*y[IDX_COII] - k[82]*y[IDX_HII] -
        k[108]*y[IDX_H2II] - k[118]*y[IDX_H2OII] - k[136]*y[IDX_H2COII] -
        k[137]*y[IDX_O2II] - k[138]*y[IDX_SiOII] - k[162]*y[IDX_NII] -
        k[171]*y[IDX_N2II] - k[180]*y[IDX_NH2II] - k[189]*y[IDX_NH3II] -
        k[211]*y[IDX_OII] - k[221]*y[IDX_OHII] - k[260] - k[261] -
        k[372]*y[IDX_CII] - k[406]*y[IDX_CHII] - k[419]*y[IDX_CH2II] -
        k[441]*y[IDX_CH3II] - k[480]*y[IDX_CNII] - k[500]*y[IDX_HII] -
        k[501]*y[IDX_HII] - k[519]*y[IDX_H2II] - k[557]*y[IDX_H2OII] -
        k[558]*y[IDX_H2OII] - k[588]*y[IDX_H3II] - k[624]*y[IDX_HCNII] -
        k[625]*y[IDX_HCNII] - k[636]*y[IDX_HCOII] - k[641]*y[IDX_H2COII] -
        k[642]*y[IDX_HNOII] - k[643]*y[IDX_N2HII] - k[644]*y[IDX_O2II] -
        k[645]*y[IDX_O2HII] - k[680]*y[IDX_HeII] - k[681]*y[IDX_HeII] -
        k[682]*y[IDX_HeII] - k[723]*y[IDX_NII] - k[731]*y[IDX_N2II] -
        k[759]*y[IDX_NHII] - k[774]*y[IDX_NH2II] - k[816]*y[IDX_OII] -
        k[842]*y[IDX_OHII] - k[843]*y[IDX_OHII] - k[864]*y[IDX_CI] -
        k[882]*y[IDX_CH2I] - k[905]*y[IDX_CH3I] - k[925]*y[IDX_CHI] -
        k[943]*y[IDX_CNI] - k[977]*y[IDX_HI] - k[978]*y[IDX_HI] -
        k[997]*y[IDX_HCOI] - k[997]*y[IDX_HCOI] - k[997]*y[IDX_HCOI] -
        k[997]*y[IDX_HCOI] - k[998]*y[IDX_HCOI] - k[998]*y[IDX_HCOI] -
        k[998]*y[IDX_HCOI] - k[998]*y[IDX_HCOI] - k[999]*y[IDX_HNOI] -
        k[1000]*y[IDX_NOI] - k[1001]*y[IDX_O2I] - k[1002]*y[IDX_O2I] -
        k[1003]*y[IDX_O2HI] - k[1014]*y[IDX_NI] - k[1015]*y[IDX_NI] -
        k[1016]*y[IDX_NI] - k[1066]*y[IDX_OI] - k[1067]*y[IDX_OI] -
        k[1096]*y[IDX_OHI] - k[1149] - k[1150] - k[1261];
    data[2032] = 0.0 - k[680]*y[IDX_HCOI] - k[681]*y[IDX_HCOI] - k[682]*y[IDX_HCOI];
    data[2033] = 0.0 + k[459]*y[IDX_H2COII] + k[923]*y[IDX_CO2I] + k[924]*y[IDX_H2COI] -
        k[925]*y[IDX_HCOI] + k[931]*y[IDX_NOI] + k[936]*y[IDX_O2I] +
        k[937]*y[IDX_O2HI] + k[941]*y[IDX_OHI];
    data[2034] = 0.0 - k[588]*y[IDX_HCOI];
    data[2035] = 0.0 + k[931]*y[IDX_CHI] - k[1000]*y[IDX_HCOI];
    data[2036] = 0.0 - k[1014]*y[IDX_HCOI] - k[1015]*y[IDX_HCOI] - k[1016]*y[IDX_HCOI];
    data[2037] = 0.0 + k[941]*y[IDX_CHI] + k[1093]*y[IDX_H2COI] - k[1096]*y[IDX_HCOI];
    data[2038] = 0.0 + k[413]*y[IDX_CHII] + k[893]*y[IDX_CH2I] + k[912]*y[IDX_CH3I] +
        k[936]*y[IDX_CHI] - k[1001]*y[IDX_HCOI] - k[1002]*y[IDX_HCOI];
    data[2039] = 0.0 - k[864]*y[IDX_HCOI];
    data[2040] = 0.0 - k[82]*y[IDX_HCOI] - k[500]*y[IDX_HCOI] - k[501]*y[IDX_HCOI];
    data[2041] = 0.0 + k[149]*y[IDX_MgI] - k[636]*y[IDX_HCOI];
    data[2042] = 0.0 + k[563]*y[IDX_H2COII];
    data[2043] = 0.0 + k[896]*y[IDX_CH2I] + k[1061]*y[IDX_H2COI] - k[1066]*y[IDX_HCOI] -
        k[1067]*y[IDX_HCOI];
    data[2044] = 0.0 + k[309]*y[IDX_H2COII] + k[321]*y[IDX_H3COII];
    data[2045] = 0.0 + k[974]*y[IDX_H2COI] - k[977]*y[IDX_HCOI] - k[978]*y[IDX_HCOI];
    data[2046] = 0.0 - k[675]*y[IDX_HeII];
    data[2047] = 0.0 - k[700]*y[IDX_HeII];
    data[2048] = 0.0 - k[701]*y[IDX_HeII] - k[702]*y[IDX_HeII];
    data[2049] = 0.0 - k[697]*y[IDX_HeII] - k[698]*y[IDX_HeII];
    data[2050] = 0.0 - k[703]*y[IDX_HeII] - k[704]*y[IDX_HeII];
    data[2051] = 0.0 - k[707]*y[IDX_HeII] - k[708]*y[IDX_HeII];
    data[2052] = 0.0 - k[709]*y[IDX_HeII];
    data[2053] = 0.0 - k[705]*y[IDX_HeII] - k[706]*y[IDX_HeII];
    data[2054] = 0.0 - k[686]*y[IDX_HeII] - k[687]*y[IDX_HeII];
    data[2055] = 0.0 - k[656]*y[IDX_HeII] - k[657]*y[IDX_HeII];
    data[2056] = 0.0 - k[710]*y[IDX_HeII] - k[711]*y[IDX_HeII];
    data[2057] = 0.0 - k[147]*y[IDX_HeII];
    data[2058] = 0.0 - k[683]*y[IDX_HeII] - k[684]*y[IDX_HeII] - k[685]*y[IDX_HeII];
    data[2059] = 0.0 - k[140]*y[IDX_HeII] - k[658]*y[IDX_HeII] - k[659]*y[IDX_HeII] -
        k[660]*y[IDX_HeII] - k[661]*y[IDX_HeII];
    data[2060] = 0.0 - k[145]*y[IDX_HeII] - k[691]*y[IDX_HeII] - k[692]*y[IDX_HeII];
    data[2061] = 0.0 - k[655]*y[IDX_HeII];
    data[2062] = 0.0 - k[665]*y[IDX_HeII] - k[666]*y[IDX_HeII] - k[667]*y[IDX_HeII] -
        k[668]*y[IDX_HeII];
    data[2063] = 0.0 - k[689]*y[IDX_HeII] - k[690]*y[IDX_HeII];
    data[2064] = 0.0 - k[144]*y[IDX_HeII] - k[688]*y[IDX_HeII];
    data[2065] = 0.0 - k[676]*y[IDX_HeII] - k[677]*y[IDX_HeII] - k[678]*y[IDX_HeII] -
        k[679]*y[IDX_HeII];
    data[2066] = 0.0 - k[653]*y[IDX_HeII] - k[654]*y[IDX_HeII];
    data[2067] = 0.0 - k[693]*y[IDX_HeII];
    data[2068] = 0.0 - k[663]*y[IDX_HeII] - k[664]*y[IDX_HeII];
    data[2069] = 0.0 - k[142]*y[IDX_HeII] - k[670]*y[IDX_HeII] - k[671]*y[IDX_HeII] -
        k[672]*y[IDX_HeII];
    data[2070] = 0.0 - k[680]*y[IDX_HeII] - k[681]*y[IDX_HeII] - k[682]*y[IDX_HeII];
    data[2071] = 0.0 - k[115]*y[IDX_H2I] - k[130]*y[IDX_HI] - k[139]*y[IDX_CI] -
        k[140]*y[IDX_CH4I] - k[141]*y[IDX_CHI] - k[142]*y[IDX_H2COI] -
        k[143]*y[IDX_H2OI] - k[144]*y[IDX_N2I] - k[145]*y[IDX_NH3I] -
        k[146]*y[IDX_O2I] - k[147]*y[IDX_SiI] - k[536]*y[IDX_H2I] -
        k[653]*y[IDX_CH2I] - k[654]*y[IDX_CH2I] - k[655]*y[IDX_CH3I] -
        k[656]*y[IDX_CH3OHI] - k[657]*y[IDX_CH3OHI] - k[658]*y[IDX_CH4I] -
        k[659]*y[IDX_CH4I] - k[660]*y[IDX_CH4I] - k[661]*y[IDX_CH4I] -
        k[662]*y[IDX_CHI] - k[663]*y[IDX_CNI] - k[664]*y[IDX_CNI] -
        k[665]*y[IDX_CO2I] - k[666]*y[IDX_CO2I] - k[667]*y[IDX_CO2I] -
        k[668]*y[IDX_CO2I] - k[669]*y[IDX_COI] - k[670]*y[IDX_H2COI] -
        k[671]*y[IDX_H2COI] - k[672]*y[IDX_H2COI] - k[673]*y[IDX_H2OI] -
        k[674]*y[IDX_H2OI] - k[675]*y[IDX_H2SiOI] - k[676]*y[IDX_HCNI] -
        k[677]*y[IDX_HCNI] - k[678]*y[IDX_HCNI] - k[679]*y[IDX_HCNI] -
        k[680]*y[IDX_HCOI] - k[681]*y[IDX_HCOI] - k[682]*y[IDX_HCOI] -
        k[683]*y[IDX_HNCI] - k[684]*y[IDX_HNCI] - k[685]*y[IDX_HNCI] -
        k[686]*y[IDX_HNOI] - k[687]*y[IDX_HNOI] - k[688]*y[IDX_N2I] -
        k[689]*y[IDX_NH2I] - k[690]*y[IDX_NH2I] - k[691]*y[IDX_NH3I] -
        k[692]*y[IDX_NH3I] - k[693]*y[IDX_NHI] - k[694]*y[IDX_NOI] -
        k[695]*y[IDX_NOI] - k[696]*y[IDX_O2I] - k[697]*y[IDX_OCNI] -
        k[698]*y[IDX_OCNI] - k[699]*y[IDX_OHI] - k[700]*y[IDX_SiC3I] -
        k[701]*y[IDX_SiCI] - k[702]*y[IDX_SiCI] - k[703]*y[IDX_SiH2I] -
        k[704]*y[IDX_SiH2I] - k[705]*y[IDX_SiH3I] - k[706]*y[IDX_SiH3I] -
        k[707]*y[IDX_SiH4I] - k[708]*y[IDX_SiH4I] - k[709]*y[IDX_SiHI] -
        k[710]*y[IDX_SiOI] - k[711]*y[IDX_SiOI] - k[1218]*y[IDX_eM];
    data[2072] = 0.0 - k[141]*y[IDX_HeII] - k[662]*y[IDX_HeII];
    data[2073] = 0.0 + k[237] + k[265];
    data[2074] = 0.0 - k[694]*y[IDX_HeII] - k[695]*y[IDX_HeII];
    data[2075] = 0.0 - k[699]*y[IDX_HeII];
    data[2076] = 0.0 - k[146]*y[IDX_HeII] - k[696]*y[IDX_HeII];
    data[2077] = 0.0 - k[139]*y[IDX_HeII];
    data[2078] = 0.0 - k[143]*y[IDX_HeII] - k[673]*y[IDX_HeII] - k[674]*y[IDX_HeII];
    data[2079] = 0.0 - k[1218]*y[IDX_HeII];
    data[2080] = 0.0 - k[669]*y[IDX_HeII];
    data[2081] = 0.0 - k[115]*y[IDX_HeII] - k[536]*y[IDX_HeII];
    data[2082] = 0.0 - k[130]*y[IDX_HeII];
    data[2083] = 0.0 - k[937]*y[IDX_CHI] - k[938]*y[IDX_CHI];
    data[2084] = 0.0 - k[477]*y[IDX_CHI];
    data[2085] = 0.0 - k[478]*y[IDX_CHI];
    data[2086] = 0.0 - k[926]*y[IDX_CHI];
    data[2087] = 0.0 + k[365]*y[IDX_CII];
    data[2088] = 0.0 + k[32]*y[IDX_CHII];
    data[2089] = 0.0 - k[53]*y[IDX_CHI];
    data[2090] = 0.0 - k[464]*y[IDX_CHI] - k[465]*y[IDX_CHI];
    data[2091] = 0.0 - k[469]*y[IDX_CHI];
    data[2092] = 0.0 - k[474]*y[IDX_CHI];
    data[2093] = 0.0 - k[476]*y[IDX_CHI];
    data[2094] = 0.0 + k[35]*y[IDX_CHII];
    data[2095] = 0.0 - k[467]*y[IDX_CHI];
    data[2096] = 0.0 - k[58]*y[IDX_CHI];
    data[2097] = 0.0 + k[318]*y[IDX_eM] - k[461]*y[IDX_CHI];
    data[2098] = 0.0 + k[1127];
    data[2099] = 0.0 - k[54]*y[IDX_CHI] + k[422]*y[IDX_CH2I] - k[458]*y[IDX_CHI];
    data[2100] = 0.0 - k[102]*y[IDX_CHI] - k[512]*y[IDX_CHI];
    data[2101] = 0.0 + k[33]*y[IDX_CHII];
    data[2102] = 0.0 + k[246] + k[1118];
    data[2103] = 0.0 - k[923]*y[IDX_CHI];
    data[2104] = 0.0 - k[57]*y[IDX_CHI] - k[468]*y[IDX_CHI];
    data[2105] = 0.0 - k[60]*y[IDX_CHI] - k[472]*y[IDX_CHI] + k[815]*y[IDX_HCNI];
    data[2106] = 0.0 - k[463]*y[IDX_CHI];
    data[2107] = 0.0 - k[59]*y[IDX_CHI] - k[471]*y[IDX_CHI];
    data[2108] = 0.0 - k[470]*y[IDX_CHI];
    data[2109] = 0.0 - k[61]*y[IDX_CHI] - k[473]*y[IDX_CHI];
    data[2110] = 0.0 + k[299]*y[IDX_eM] + k[300]*y[IDX_eM];
    data[2111] = 0.0 + k[868]*y[IDX_CI];
    data[2112] = 0.0 + k[297]*y[IDX_eM] + k[1111];
    data[2113] = 0.0 - k[56]*y[IDX_CHI] - k[460]*y[IDX_CHI];
    data[2114] = 0.0 - k[462]*y[IDX_CHI];
    data[2115] = 0.0 - k[927]*y[IDX_CHI];
    data[2116] = 0.0 - k[15]*y[IDX_CHI] + k[365]*y[IDX_CH3OHI] + k[369]*y[IDX_H2COI];
    data[2117] = 0.0 + k[677]*y[IDX_HeII] + k[815]*y[IDX_OII];
    data[2118] = 0.0 + k[31]*y[IDX_HCOI] + k[32]*y[IDX_MgI] + k[33]*y[IDX_NH3I] +
        k[34]*y[IDX_NOI] + k[35]*y[IDX_SiI];
    data[2119] = 0.0 + k[243] + k[422]*y[IDX_COII] + k[863]*y[IDX_CI] + k[863]*y[IDX_CI]
        + k[878]*y[IDX_CH2I] + k[878]*y[IDX_CH2I] + k[880]*y[IDX_CNI] +
        k[897]*y[IDX_OI] + k[899]*y[IDX_OHI] + k[967]*y[IDX_HI] +
        k[1007]*y[IDX_NI] + k[1113];
    data[2120] = 0.0 - k[55]*y[IDX_CHI] - k[459]*y[IDX_CHI];
    data[2121] = 0.0 + k[870]*y[IDX_CI];
    data[2122] = 0.0 - k[62]*y[IDX_CHI] - k[475]*y[IDX_CHI];
    data[2123] = 0.0 + k[880]*y[IDX_CH2I];
    data[2124] = 0.0 + k[369]*y[IDX_CII] - k[924]*y[IDX_CHI];
    data[2125] = 0.0 + k[31]*y[IDX_CHII] + k[864]*y[IDX_CI] - k[925]*y[IDX_CHI];
    data[2126] = 0.0 - k[141]*y[IDX_CHI] - k[662]*y[IDX_CHI] + k[677]*y[IDX_HCNI];
    data[2127] = 0.0 - k[0]*y[IDX_OI] - k[2]*y[IDX_H2I] - k[9]*y[IDX_HI] -
        k[15]*y[IDX_CII] - k[53]*y[IDX_CNII] - k[54]*y[IDX_COII] -
        k[55]*y[IDX_H2COII] - k[56]*y[IDX_H2OII] - k[57]*y[IDX_NII] -
        k[58]*y[IDX_N2II] - k[59]*y[IDX_NH2II] - k[60]*y[IDX_OII] -
        k[61]*y[IDX_O2II] - k[62]*y[IDX_OHII] - k[78]*y[IDX_HII] -
        k[102]*y[IDX_H2II] - k[141]*y[IDX_HeII] - k[250] - k[458]*y[IDX_COII] -
        k[459]*y[IDX_H2COII] - k[460]*y[IDX_H2OII] - k[461]*y[IDX_H3COII] -
        k[462]*y[IDX_H3OII] - k[463]*y[IDX_HCNII] - k[464]*y[IDX_HCNHII] -
        k[465]*y[IDX_HCNHII] - k[466]*y[IDX_HCOII] - k[467]*y[IDX_HNOII] -
        k[468]*y[IDX_NII] - k[469]*y[IDX_N2HII] - k[470]*y[IDX_NHII] -
        k[471]*y[IDX_NH2II] - k[472]*y[IDX_OII] - k[473]*y[IDX_O2II] -
        k[474]*y[IDX_O2HII] - k[475]*y[IDX_OHII] - k[476]*y[IDX_SiII] -
        k[477]*y[IDX_SiHII] - k[478]*y[IDX_SiOII] - k[512]*y[IDX_H2II] -
        k[580]*y[IDX_H3II] - k[662]*y[IDX_HeII] - k[923]*y[IDX_CO2I] -
        k[924]*y[IDX_H2COI] - k[925]*y[IDX_HCOI] - k[926]*y[IDX_HNOI] -
        k[927]*y[IDX_N2I] - k[928]*y[IDX_NI] - k[929]*y[IDX_NI] -
        k[930]*y[IDX_NOI] - k[931]*y[IDX_NOI] - k[932]*y[IDX_NOI] -
        k[933]*y[IDX_O2I] - k[934]*y[IDX_O2I] - k[935]*y[IDX_O2I] -
        k[936]*y[IDX_O2I] - k[937]*y[IDX_O2HI] - k[938]*y[IDX_O2HI] -
        k[939]*y[IDX_OI] - k[940]*y[IDX_OI] - k[941]*y[IDX_OHI] -
        k[958]*y[IDX_H2I] - k[970]*y[IDX_HI] - k[1128] - k[1129] -
        k[1201]*y[IDX_H2I] - k[1253];
    data[2128] = 0.0 - k[580]*y[IDX_CHI];
    data[2129] = 0.0 + k[34]*y[IDX_CHII] - k[930]*y[IDX_CHI] - k[931]*y[IDX_CHI] -
        k[932]*y[IDX_CHI];
    data[2130] = 0.0 - k[928]*y[IDX_CHI] - k[929]*y[IDX_CHI] + k[1007]*y[IDX_CH2I];
    data[2131] = 0.0 + k[876]*y[IDX_CI] + k[899]*y[IDX_CH2I] - k[941]*y[IDX_CHI];
    data[2132] = 0.0 - k[933]*y[IDX_CHI] - k[934]*y[IDX_CHI] - k[935]*y[IDX_CHI] -
        k[936]*y[IDX_CHI];
    data[2133] = 0.0 + k[863]*y[IDX_CH2I] + k[863]*y[IDX_CH2I] + k[864]*y[IDX_HCOI] +
        k[868]*y[IDX_NH2I] + k[870]*y[IDX_NHI] + k[876]*y[IDX_OHI] +
        k[955]*y[IDX_H2I] + k[1206]*y[IDX_HI];
    data[2134] = 0.0 - k[78]*y[IDX_CHI];
    data[2135] = 0.0 - k[466]*y[IDX_CHI];
    data[2136] = 0.0 - k[0]*y[IDX_CHI] + k[897]*y[IDX_CH2I] - k[939]*y[IDX_CHI] -
        k[940]*y[IDX_CHI];
    data[2137] = 0.0 + k[297]*y[IDX_CH2II] + k[299]*y[IDX_CH3II] + k[300]*y[IDX_CH3II] +
        k[318]*y[IDX_H3COII];
    data[2138] = 0.0 - k[2]*y[IDX_CHI] + k[955]*y[IDX_CI] - k[958]*y[IDX_CHI] -
        k[1201]*y[IDX_CHI];
    data[2139] = 0.0 - k[9]*y[IDX_CHI] + k[967]*y[IDX_CH2I] - k[970]*y[IDX_CHI] +
        k[1206]*y[IDX_CI];
    data[2140] = 0.0 + k[537]*y[IDX_H2I];
    data[2141] = 0.0 - k[595]*y[IDX_H3II];
    data[2142] = 0.0 - k[602]*y[IDX_H3II];
    data[2143] = 0.0 - k[604]*y[IDX_H3II];
    data[2144] = 0.0 - k[605]*y[IDX_H3II];
    data[2145] = 0.0 - k[603]*y[IDX_H3II];
    data[2146] = 0.0 - k[590]*y[IDX_H3II];
    data[2147] = 0.0 - k[579]*y[IDX_H3II];
    data[2148] = 0.0 - k[591]*y[IDX_H3II];
    data[2149] = 0.0 - k[606]*y[IDX_H3II];
    data[2150] = 0.0 + k[544]*y[IDX_H2I];
    data[2151] = 0.0 - k[601]*y[IDX_H3II];
    data[2152] = 0.0 - k[589]*y[IDX_H3II];
    data[2153] = 0.0 + k[516]*y[IDX_H2I] + k[519]*y[IDX_HCOI];
    data[2154] = 0.0 - k[578]*y[IDX_H3II];
    data[2155] = 0.0 - k[582]*y[IDX_H3II];
    data[2156] = 0.0 + k[540]*y[IDX_H2I];
    data[2157] = 0.0 - k[593]*y[IDX_H3II];
    data[2158] = 0.0 - k[592]*y[IDX_H3II];
    data[2159] = 0.0 - k[587]*y[IDX_H3II];
    data[2160] = 0.0 - k[577]*y[IDX_H3II];
    data[2161] = 0.0 - k[594]*y[IDX_H3II];
    data[2162] = 0.0 - k[581]*y[IDX_H3II];
    data[2163] = 0.0 - k[585]*y[IDX_H3II];
    data[2164] = 0.0 + k[519]*y[IDX_H2II] - k[588]*y[IDX_H3II];
    data[2165] = 0.0 - k[580]*y[IDX_H3II];
    data[2166] = 0.0 - k[315]*y[IDX_eM] - k[316]*y[IDX_eM] - k[576]*y[IDX_CI] -
        k[577]*y[IDX_CH2I] - k[578]*y[IDX_CH3I] - k[579]*y[IDX_CH3OHI] -
        k[580]*y[IDX_CHI] - k[581]*y[IDX_CNI] - k[582]*y[IDX_CO2I] -
        k[583]*y[IDX_COI] - k[584]*y[IDX_COI] - k[585]*y[IDX_H2COI] -
        k[586]*y[IDX_H2OI] - k[587]*y[IDX_HCNI] - k[588]*y[IDX_HCOI] -
        k[589]*y[IDX_HNCI] - k[590]*y[IDX_HNOI] - k[591]*y[IDX_MgI] -
        k[592]*y[IDX_N2I] - k[593]*y[IDX_NH2I] - k[594]*y[IDX_NHI] -
        k[595]*y[IDX_NO2I] - k[596]*y[IDX_NOI] - k[597]*y[IDX_O2I] -
        k[598]*y[IDX_OI] - k[599]*y[IDX_OI] - k[600]*y[IDX_OHI] -
        k[601]*y[IDX_SiI] - k[602]*y[IDX_SiH2I] - k[603]*y[IDX_SiH3I] -
        k[604]*y[IDX_SiH4I] - k[605]*y[IDX_SiHI] - k[606]*y[IDX_SiOI] - k[1145]
        - k[1146];
    data[2167] = 0.0 - k[596]*y[IDX_H3II];
    data[2168] = 0.0 - k[600]*y[IDX_H3II];
    data[2169] = 0.0 - k[597]*y[IDX_H3II];
    data[2170] = 0.0 - k[576]*y[IDX_H3II];
    data[2171] = 0.0 - k[586]*y[IDX_H3II];
    data[2172] = 0.0 - k[598]*y[IDX_H3II] - k[599]*y[IDX_H3II];
    data[2173] = 0.0 - k[315]*y[IDX_H3II] - k[316]*y[IDX_H3II];
    data[2174] = 0.0 - k[583]*y[IDX_H3II] - k[584]*y[IDX_H3II];
    data[2175] = 0.0 + k[516]*y[IDX_H2II] + k[537]*y[IDX_HeHII] + k[540]*y[IDX_NHII] +
        k[544]*y[IDX_O2HII];
    data[2176] = 0.0 + k[675]*y[IDX_HeII];
    data[2177] = 0.0 + k[336]*y[IDX_eM] + k[537]*y[IDX_H2I] + k[618]*y[IDX_HI];
    data[2178] = 0.0 + k[700]*y[IDX_HeII];
    data[2179] = 0.0 + k[701]*y[IDX_HeII] + k[702]*y[IDX_HeII];
    data[2180] = 0.0 + k[697]*y[IDX_HeII] + k[698]*y[IDX_HeII];
    data[2181] = 0.0 + k[703]*y[IDX_HeII] + k[704]*y[IDX_HeII];
    data[2182] = 0.0 + k[707]*y[IDX_HeII] + k[708]*y[IDX_HeII];
    data[2183] = 0.0 + k[709]*y[IDX_HeII];
    data[2184] = 0.0 + k[705]*y[IDX_HeII] + k[706]*y[IDX_HeII];
    data[2185] = 0.0 + k[686]*y[IDX_HeII] + k[687]*y[IDX_HeII];
    data[2186] = 0.0 + k[656]*y[IDX_HeII] + k[657]*y[IDX_HeII];
    data[2187] = 0.0 + k[710]*y[IDX_HeII] + k[711]*y[IDX_HeII];
    data[2188] = 0.0 + k[147]*y[IDX_HeII];
    data[2189] = 0.0 + k[683]*y[IDX_HeII] + k[684]*y[IDX_HeII] + k[685]*y[IDX_HeII];
    data[2190] = 0.0 + k[140]*y[IDX_HeII] + k[658]*y[IDX_HeII] + k[659]*y[IDX_HeII] +
        k[660]*y[IDX_HeII] + k[661]*y[IDX_HeII];
    data[2191] = 0.0 - k[520]*y[IDX_HeI];
    data[2192] = 0.0 + k[145]*y[IDX_HeII] + k[691]*y[IDX_HeII] + k[692]*y[IDX_HeII];
    data[2193] = 0.0 + k[655]*y[IDX_HeII];
    data[2194] = 0.0 + k[665]*y[IDX_HeII] + k[666]*y[IDX_HeII] + k[667]*y[IDX_HeII] +
        k[668]*y[IDX_HeII];
    data[2195] = 0.0 + k[689]*y[IDX_HeII] + k[690]*y[IDX_HeII];
    data[2196] = 0.0 + k[144]*y[IDX_HeII] + k[688]*y[IDX_HeII];
    data[2197] = 0.0 + k[676]*y[IDX_HeII] + k[677]*y[IDX_HeII] + k[678]*y[IDX_HeII] +
        k[679]*y[IDX_HeII];
    data[2198] = 0.0 + k[653]*y[IDX_HeII] + k[654]*y[IDX_HeII];
    data[2199] = 0.0 + k[693]*y[IDX_HeII];
    data[2200] = 0.0 + k[663]*y[IDX_HeII] + k[664]*y[IDX_HeII];
    data[2201] = 0.0 + k[142]*y[IDX_HeII] + k[670]*y[IDX_HeII] + k[671]*y[IDX_HeII] +
        k[672]*y[IDX_HeII];
    data[2202] = 0.0 + k[680]*y[IDX_HeII] + k[682]*y[IDX_HeII];
    data[2203] = 0.0 + k[115]*y[IDX_H2I] + k[130]*y[IDX_HI] + k[139]*y[IDX_CI] +
        k[140]*y[IDX_CH4I] + k[141]*y[IDX_CHI] + k[142]*y[IDX_H2COI] +
        k[143]*y[IDX_H2OI] + k[144]*y[IDX_N2I] + k[145]*y[IDX_NH3I] +
        k[146]*y[IDX_O2I] + k[147]*y[IDX_SiI] + k[536]*y[IDX_H2I] +
        k[653]*y[IDX_CH2I] + k[654]*y[IDX_CH2I] + k[655]*y[IDX_CH3I] +
        k[656]*y[IDX_CH3OHI] + k[657]*y[IDX_CH3OHI] + k[658]*y[IDX_CH4I] +
        k[659]*y[IDX_CH4I] + k[660]*y[IDX_CH4I] + k[661]*y[IDX_CH4I] +
        k[662]*y[IDX_CHI] + k[663]*y[IDX_CNI] + k[664]*y[IDX_CNI] +
        k[665]*y[IDX_CO2I] + k[666]*y[IDX_CO2I] + k[667]*y[IDX_CO2I] +
        k[668]*y[IDX_CO2I] + k[669]*y[IDX_COI] + k[670]*y[IDX_H2COI] +
        k[671]*y[IDX_H2COI] + k[672]*y[IDX_H2COI] + k[673]*y[IDX_H2OI] +
        k[674]*y[IDX_H2OI] + k[675]*y[IDX_H2SiOI] + k[676]*y[IDX_HCNI] +
        k[677]*y[IDX_HCNI] + k[678]*y[IDX_HCNI] + k[679]*y[IDX_HCNI] +
        k[680]*y[IDX_HCOI] + k[682]*y[IDX_HCOI] + k[683]*y[IDX_HNCI] +
        k[684]*y[IDX_HNCI] + k[685]*y[IDX_HNCI] + k[686]*y[IDX_HNOI] +
        k[687]*y[IDX_HNOI] + k[688]*y[IDX_N2I] + k[689]*y[IDX_NH2I] +
        k[690]*y[IDX_NH2I] + k[691]*y[IDX_NH3I] + k[692]*y[IDX_NH3I] +
        k[693]*y[IDX_NHI] + k[694]*y[IDX_NOI] + k[695]*y[IDX_NOI] +
        k[696]*y[IDX_O2I] + k[697]*y[IDX_OCNI] + k[698]*y[IDX_OCNI] +
        k[699]*y[IDX_OHI] + k[700]*y[IDX_SiC3I] + k[701]*y[IDX_SiCI] +
        k[702]*y[IDX_SiCI] + k[703]*y[IDX_SiH2I] + k[704]*y[IDX_SiH2I] +
        k[705]*y[IDX_SiH3I] + k[706]*y[IDX_SiH3I] + k[707]*y[IDX_SiH4I] +
        k[708]*y[IDX_SiH4I] + k[709]*y[IDX_SiHI] + k[710]*y[IDX_SiOI] +
        k[711]*y[IDX_SiOI] + k[1218]*y[IDX_eM];
    data[2204] = 0.0 + k[141]*y[IDX_HeII] + k[662]*y[IDX_HeII];
    data[2205] = 0.0 - k[237] - k[265] - k[520]*y[IDX_H2II] - k[1198]*y[IDX_HII];
    data[2206] = 0.0 + k[694]*y[IDX_HeII] + k[695]*y[IDX_HeII];
    data[2207] = 0.0 + k[699]*y[IDX_HeII];
    data[2208] = 0.0 + k[146]*y[IDX_HeII] + k[696]*y[IDX_HeII];
    data[2209] = 0.0 + k[139]*y[IDX_HeII];
    data[2210] = 0.0 - k[1198]*y[IDX_HeI];
    data[2211] = 0.0 + k[143]*y[IDX_HeII] + k[673]*y[IDX_HeII] + k[674]*y[IDX_HeII];
    data[2212] = 0.0 + k[336]*y[IDX_HeHII] + k[1218]*y[IDX_HeII];
    data[2213] = 0.0 + k[669]*y[IDX_HeII];
    data[2214] = 0.0 + k[115]*y[IDX_HeII] + k[536]*y[IDX_HeII] + k[537]*y[IDX_HeHII];
    data[2215] = 0.0 + k[130]*y[IDX_HeII] + k[618]*y[IDX_HeHII];
    data[2216] = 0.0 + k[1343] + k[1344] + k[1345] + k[1346];
    data[2217] = 0.0 + k[311]*y[IDX_eM];
    data[2218] = 0.0 + k[276] + k[885]*y[IDX_CH2I] + k[945]*y[IDX_CNI] +
        k[952]*y[IDX_COI] + k[986]*y[IDX_HI] + k[1020]*y[IDX_NI] +
        k[1020]*y[IDX_NI] + k[1041]*y[IDX_NHI] + k[1075]*y[IDX_OI] + k[1164];
    data[2219] = 0.0 - k[1053]*y[IDX_NOI] + k[1054]*y[IDX_O2I] + k[1078]*y[IDX_OI];
    data[2220] = 0.0 - k[206]*y[IDX_NOI] + k[746]*y[IDX_NI];
    data[2221] = 0.0 + k[264] + k[687]*y[IDX_HeII] + k[883]*y[IDX_CH2I] +
        k[906]*y[IDX_CH3I] + k[926]*y[IDX_CHI] + k[944]*y[IDX_CNI] +
        k[981]*y[IDX_HI] + k[999]*y[IDX_HCOI] + k[1017]*y[IDX_NI] +
        k[1069]*y[IDX_OI] + k[1097]*y[IDX_OHI] + k[1153];
    data[2222] = 0.0 + k[715]*y[IDX_NII];
    data[2223] = 0.0 + k[151]*y[IDX_NOII];
    data[2224] = 0.0 - k[67]*y[IDX_NOI];
    data[2225] = 0.0 - k[807]*y[IDX_NOI];
    data[2226] = 0.0 + k[229]*y[IDX_NOII] - k[1105]*y[IDX_NOI];
    data[2227] = 0.0 + k[649]*y[IDX_HNOII];
    data[2228] = 0.0 - k[204]*y[IDX_NOI] + k[334]*y[IDX_eM] + k[388]*y[IDX_CI] +
        k[430]*y[IDX_CH2I] + k[467]*y[IDX_CHI] + k[482]*y[IDX_CNI] +
        k[486]*y[IDX_COI] + k[550]*y[IDX_H2COI] + k[568]*y[IDX_H2OI] +
        k[630]*y[IDX_HCNI] + k[642]*y[IDX_HCOI] + k[649]*y[IDX_HNCI] +
        k[652]*y[IDX_CO2I] + k[732]*y[IDX_N2I] + k[788]*y[IDX_NH2I] +
        k[800]*y[IDX_NHI] + k[856]*y[IDX_OHI];
    data[2229] = 0.0 - k[172]*y[IDX_NOI];
    data[2230] = 0.0 - k[72]*y[IDX_NOI];
    data[2231] = 0.0 - k[112]*y[IDX_NOI] - k[524]*y[IDX_NOI];
    data[2232] = 0.0 + k[906]*y[IDX_HNOI] - k[910]*y[IDX_NOI];
    data[2233] = 0.0 + k[652]*y[IDX_HNOII] + k[719]*y[IDX_NII] + k[1012]*y[IDX_NI];
    data[2234] = 0.0 - k[167]*y[IDX_NOI] + k[715]*y[IDX_CH3OHI] + k[719]*y[IDX_CO2I] -
        k[727]*y[IDX_NOI] + k[729]*y[IDX_O2I];
    data[2235] = 0.0 - k[132]*y[IDX_NOI];
    data[2236] = 0.0 - k[182]*y[IDX_NOI];
    data[2237] = 0.0 - k[178]*y[IDX_NOI] - k[764]*y[IDX_NOI];
    data[2238] = 0.0 - k[205]*y[IDX_NOI];
    data[2239] = 0.0 - k[48]*y[IDX_NOI];
    data[2240] = 0.0 + k[788]*y[IDX_HNOII] - k[1029]*y[IDX_NOI] - k[1030]*y[IDX_NOI];
    data[2241] = 0.0 - k[36]*y[IDX_NOI];
    data[2242] = 0.0 - k[120]*y[IDX_NOI];
    data[2243] = 0.0 - k[191]*y[IDX_NOI];
    data[2244] = 0.0 + k[151]*y[IDX_MgI] + k[229]*y[IDX_SiI];
    data[2245] = 0.0 + k[732]*y[IDX_HNOII] + k[1071]*y[IDX_OI];
    data[2246] = 0.0 - k[20]*y[IDX_NOI];
    data[2247] = 0.0 + k[630]*y[IDX_HNOII];
    data[2248] = 0.0 - k[34]*y[IDX_NOI];
    data[2249] = 0.0 + k[430]*y[IDX_HNOII] + k[883]*y[IDX_HNOI] + k[885]*y[IDX_NO2I] -
        k[886]*y[IDX_NOI] - k[887]*y[IDX_NOI] - k[888]*y[IDX_NOI];
    data[2250] = 0.0 - k[203]*y[IDX_NOI];
    data[2251] = 0.0 + k[800]*y[IDX_HNOII] + k[1041]*y[IDX_NO2I] - k[1042]*y[IDX_NOI] -
        k[1043]*y[IDX_NOI] + k[1045]*y[IDX_O2I] + k[1046]*y[IDX_OI];
    data[2252] = 0.0 - k[223]*y[IDX_NOI] - k[846]*y[IDX_NOI];
    data[2253] = 0.0 + k[482]*y[IDX_HNOII] + k[944]*y[IDX_HNOI] + k[945]*y[IDX_NO2I] -
        k[946]*y[IDX_NOI] - k[947]*y[IDX_NOI] + k[948]*y[IDX_O2I] +
        k[1058]*y[IDX_OI];
    data[2254] = 0.0 + k[550]*y[IDX_HNOII];
    data[2255] = 0.0 + k[642]*y[IDX_HNOII] + k[999]*y[IDX_HNOI] - k[1000]*y[IDX_NOI];
    data[2256] = 0.0 + k[687]*y[IDX_HNOI] - k[694]*y[IDX_NOI] - k[695]*y[IDX_NOI];
    data[2257] = 0.0 + k[467]*y[IDX_HNOII] + k[926]*y[IDX_HNOI] - k[930]*y[IDX_NOI] -
        k[931]*y[IDX_NOI] - k[932]*y[IDX_NOI];
    data[2258] = 0.0 - k[596]*y[IDX_NOI];
    data[2259] = 0.0 - k[20]*y[IDX_CII] - k[34]*y[IDX_CHII] - k[36]*y[IDX_CH2II] -
        k[48]*y[IDX_CH3II] - k[67]*y[IDX_CNII] - k[72]*y[IDX_COII] -
        k[87]*y[IDX_HII] - k[112]*y[IDX_H2II] - k[120]*y[IDX_H2OII] -
        k[132]*y[IDX_HCNII] - k[167]*y[IDX_NII] - k[172]*y[IDX_N2II] -
        k[178]*y[IDX_NHII] - k[182]*y[IDX_NH2II] - k[191]*y[IDX_NH3II] -
        k[203]*y[IDX_H2COII] - k[204]*y[IDX_HNOII] - k[205]*y[IDX_O2II] -
        k[206]*y[IDX_SiOII] - k[223]*y[IDX_OHII] - k[277] - k[278] -
        k[524]*y[IDX_H2II] - k[596]*y[IDX_H3II] - k[694]*y[IDX_HeII] -
        k[695]*y[IDX_HeII] - k[727]*y[IDX_NII] - k[764]*y[IDX_NHII] -
        k[807]*y[IDX_O2HII] - k[846]*y[IDX_OHII] - k[871]*y[IDX_CI] -
        k[872]*y[IDX_CI] - k[886]*y[IDX_CH2I] - k[887]*y[IDX_CH2I] -
        k[888]*y[IDX_CH2I] - k[910]*y[IDX_CH3I] - k[930]*y[IDX_CHI] -
        k[931]*y[IDX_CHI] - k[932]*y[IDX_CHI] - k[946]*y[IDX_CNI] -
        k[947]*y[IDX_CNI] - k[987]*y[IDX_HI] - k[988]*y[IDX_HI] -
        k[1000]*y[IDX_HCOI] - k[1022]*y[IDX_NI] - k[1029]*y[IDX_NH2I] -
        k[1030]*y[IDX_NH2I] - k[1042]*y[IDX_NHI] - k[1043]*y[IDX_NHI] -
        k[1051]*y[IDX_NOI] - k[1051]*y[IDX_NOI] - k[1051]*y[IDX_NOI] -
        k[1051]*y[IDX_NOI] - k[1052]*y[IDX_O2I] - k[1053]*y[IDX_OCNI] -
        k[1076]*y[IDX_OI] - k[1099]*y[IDX_OHI] - k[1105]*y[IDX_SiI] - k[1165] -
        k[1166] - k[1255];
    data[2260] = 0.0 + k[746]*y[IDX_SiOII] + k[1012]*y[IDX_CO2I] + k[1017]*y[IDX_HNOI] +
        k[1020]*y[IDX_NO2I] + k[1020]*y[IDX_NO2I] - k[1022]*y[IDX_NOI] +
        k[1023]*y[IDX_O2I] + k[1025]*y[IDX_OHI];
    data[2261] = 0.0 + k[856]*y[IDX_HNOII] + k[1025]*y[IDX_NI] + k[1097]*y[IDX_HNOI] -
        k[1099]*y[IDX_NOI];
    data[2262] = 0.0 + k[729]*y[IDX_NII] + k[948]*y[IDX_CNI] + k[1023]*y[IDX_NI] +
        k[1045]*y[IDX_NHI] - k[1052]*y[IDX_NOI] + k[1054]*y[IDX_OCNI];
    data[2263] = 0.0 + k[388]*y[IDX_HNOII] - k[871]*y[IDX_NOI] - k[872]*y[IDX_NOI];
    data[2264] = 0.0 - k[87]*y[IDX_NOI];
    data[2265] = 0.0 + k[568]*y[IDX_HNOII];
    data[2266] = 0.0 + k[1046]*y[IDX_NHI] + k[1058]*y[IDX_CNI] + k[1069]*y[IDX_HNOI] +
        k[1071]*y[IDX_N2I] + k[1075]*y[IDX_NO2I] - k[1076]*y[IDX_NOI] +
        k[1078]*y[IDX_OCNI];
    data[2267] = 0.0 + k[311]*y[IDX_H2NOII] + k[334]*y[IDX_HNOII];
    data[2268] = 0.0 + k[486]*y[IDX_HNOII] + k[952]*y[IDX_NO2I];
    data[2269] = 0.0 + k[981]*y[IDX_HNOI] + k[986]*y[IDX_NO2I] - k[987]*y[IDX_NOI] -
        k[988]*y[IDX_NOI];
    data[2270] = 0.0 - k[1013]*y[IDX_NI];
    data[2271] = 0.0 - k[744]*y[IDX_NI];
    data[2272] = 0.0 - k[1024]*y[IDX_NI];
    data[2273] = 0.0 - k[1027]*y[IDX_NI];
    data[2274] = 0.0 - k[1019]*y[IDX_NI] - k[1020]*y[IDX_NI] - k[1021]*y[IDX_NI];
    data[2275] = 0.0 - k[745]*y[IDX_NI] - k[746]*y[IDX_NI];
    data[2276] = 0.0 - k[1017]*y[IDX_NI];
    data[2277] = 0.0 + k[163]*y[IDX_NII];
    data[2278] = 0.0 + k[303]*y[IDX_eM] - k[737]*y[IDX_NI];
    data[2279] = 0.0 + k[339]*y[IDX_eM];
    data[2280] = 0.0 + k[1105]*y[IDX_NOI];
    data[2281] = 0.0 + k[684]*y[IDX_HeII] + k[760]*y[IDX_NHII];
    data[2282] = 0.0 - k[174]*y[IDX_NI] + k[337]*y[IDX_eM] + k[337]*y[IDX_eM] +
        k[825]*y[IDX_OI];
    data[2283] = 0.0 + k[156]*y[IDX_NII] + k[716]*y[IDX_NII];
    data[2284] = 0.0 + k[795]*y[IDX_NHI];
    data[2285] = 0.0 - k[522]*y[IDX_NI];
    data[2286] = 0.0 + k[165]*y[IDX_NII];
    data[2287] = 0.0 - k[1008]*y[IDX_NI] - k[1009]*y[IDX_NI] - k[1010]*y[IDX_NI];
    data[2288] = 0.0 + k[748]*y[IDX_NHII] - k[1012]*y[IDX_NI];
    data[2289] = 0.0 + k[57]*y[IDX_CHI] + k[155]*y[IDX_CH2I] + k[156]*y[IDX_CH4I] +
        k[157]*y[IDX_CNI] + k[158]*y[IDX_COI] + k[159]*y[IDX_H2COI] +
        k[160]*y[IDX_H2OI] + k[161]*y[IDX_HCNI] + k[162]*y[IDX_HCOI] +
        k[163]*y[IDX_MgI] + k[164]*y[IDX_NH2I] + k[165]*y[IDX_NH3I] +
        k[166]*y[IDX_NHI] + k[167]*y[IDX_NOI] + k[168]*y[IDX_O2I] +
        k[169]*y[IDX_OHI] + k[716]*y[IDX_CH4I] - k[1210]*y[IDX_NI] +
        k[1220]*y[IDX_eM];
    data[2290] = 0.0 + k[814]*y[IDX_HCNI] + k[817]*y[IDX_N2I];
    data[2291] = 0.0 + k[341]*y[IDX_eM] - k[741]*y[IDX_NI] + k[802]*y[IDX_NHI];
    data[2292] = 0.0 + k[340]*y[IDX_eM] + k[390]*y[IDX_CI] + k[432]*y[IDX_CH2I] +
        k[470]*y[IDX_CHI] + k[540]*y[IDX_H2I] - k[740]*y[IDX_NI] +
        k[747]*y[IDX_CNI] + k[748]*y[IDX_CO2I] + k[751]*y[IDX_COI] +
        k[752]*y[IDX_H2COI] + k[754]*y[IDX_H2OI] + k[758]*y[IDX_HCNI] +
        k[759]*y[IDX_HCOI] + k[760]*y[IDX_HNCI] + k[761]*y[IDX_N2I] +
        k[762]*y[IDX_NH2I] + k[763]*y[IDX_NHI] + k[766]*y[IDX_O2I] +
        k[767]*y[IDX_OI] + k[768]*y[IDX_OHI] + k[1156];
    data[2293] = 0.0 - k[742]*y[IDX_NI];
    data[2294] = 0.0 + k[164]*y[IDX_NII] + k[762]*y[IDX_NHII];
    data[2295] = 0.0 - k[736]*y[IDX_NI];
    data[2296] = 0.0 - k[738]*y[IDX_NI] - k[739]*y[IDX_NI] + k[797]*y[IDX_NHI];
    data[2297] = 0.0 + k[345]*y[IDX_eM];
    data[2298] = 0.0 + k[267] + k[267] + k[688]*y[IDX_HeII] + k[761]*y[IDX_NHII] +
        k[817]*y[IDX_OII] + k[865]*y[IDX_CI] + k[927]*y[IDX_CHI] +
        k[1071]*y[IDX_OI] + k[1155] + k[1155];
    data[2299] = 0.0 - k[1192]*y[IDX_NI];
    data[2300] = 0.0 + k[161]*y[IDX_NII] + k[678]*y[IDX_HeII] + k[679]*y[IDX_HeII] +
        k[758]*y[IDX_NHII] + k[814]*y[IDX_OII];
    data[2301] = 0.0 - k[408]*y[IDX_NI];
    data[2302] = 0.0 + k[155]*y[IDX_NII] + k[432]*y[IDX_NHII] + k[886]*y[IDX_NOI] -
        k[1005]*y[IDX_NI] - k[1006]*y[IDX_NI] - k[1007]*y[IDX_NI];
    data[2303] = 0.0 + k[796]*y[IDX_NHI];
    data[2304] = 0.0 + k[166]*y[IDX_NII] + k[274] + k[763]*y[IDX_NHII] +
        k[795]*y[IDX_COII] + k[796]*y[IDX_H2COII] + k[797]*y[IDX_H2OII] +
        k[802]*y[IDX_NH2II] + k[870]*y[IDX_CI] + k[985]*y[IDX_HI] -
        k[1018]*y[IDX_NI] + k[1035]*y[IDX_CNI] + k[1040]*y[IDX_NHI] +
        k[1040]*y[IDX_NHI] + k[1047]*y[IDX_OI] + k[1048]*y[IDX_OHI] + k[1162];
    data[2305] = 0.0 - k[743]*y[IDX_NI];
    data[2306] = 0.0 + k[157]*y[IDX_NII] + k[251] + k[664]*y[IDX_HeII] +
        k[747]*y[IDX_NHII] + k[947]*y[IDX_NOI] - k[1011]*y[IDX_NI] +
        k[1035]*y[IDX_NHI] + k[1057]*y[IDX_OI] + k[1130];
    data[2307] = 0.0 + k[159]*y[IDX_NII] + k[752]*y[IDX_NHII];
    data[2308] = 0.0 + k[162]*y[IDX_NII] + k[759]*y[IDX_NHII] - k[1014]*y[IDX_NI] -
        k[1015]*y[IDX_NI] - k[1016]*y[IDX_NI];
    data[2309] = 0.0 + k[664]*y[IDX_CNI] + k[678]*y[IDX_HCNI] + k[679]*y[IDX_HCNI] +
        k[684]*y[IDX_HNCI] + k[688]*y[IDX_N2I] + k[694]*y[IDX_NOI];
    data[2310] = 0.0 + k[57]*y[IDX_NII] + k[470]*y[IDX_NHII] + k[927]*y[IDX_N2I] -
        k[928]*y[IDX_NI] - k[929]*y[IDX_NI] + k[931]*y[IDX_NOI];
    data[2311] = 0.0 + k[167]*y[IDX_NII] + k[278] + k[694]*y[IDX_HeII] + k[872]*y[IDX_CI]
        + k[886]*y[IDX_CH2I] + k[931]*y[IDX_CHI] + k[947]*y[IDX_CNI] +
        k[988]*y[IDX_HI] - k[1022]*y[IDX_NI] + k[1076]*y[IDX_OI] +
        k[1105]*y[IDX_SiI] + k[1166];
    data[2312] = 0.0 - k[174]*y[IDX_N2II] - k[238] - k[268] - k[408]*y[IDX_CHII] -
        k[522]*y[IDX_H2II] - k[736]*y[IDX_CH2II] - k[737]*y[IDX_CNII] -
        k[738]*y[IDX_H2OII] - k[739]*y[IDX_H2OII] - k[740]*y[IDX_NHII] -
        k[741]*y[IDX_NH2II] - k[742]*y[IDX_O2II] - k[743]*y[IDX_OHII] -
        k[744]*y[IDX_SiCII] - k[745]*y[IDX_SiOII] - k[746]*y[IDX_SiOII] -
        k[928]*y[IDX_CHI] - k[929]*y[IDX_CHI] - k[960]*y[IDX_H2I] -
        k[1005]*y[IDX_CH2I] - k[1006]*y[IDX_CH2I] - k[1007]*y[IDX_CH2I] -
        k[1008]*y[IDX_CH3I] - k[1009]*y[IDX_CH3I] - k[1010]*y[IDX_CH3I] -
        k[1011]*y[IDX_CNI] - k[1012]*y[IDX_CO2I] - k[1013]*y[IDX_H2CNI] -
        k[1014]*y[IDX_HCOI] - k[1015]*y[IDX_HCOI] - k[1016]*y[IDX_HCOI] -
        k[1017]*y[IDX_HNOI] - k[1018]*y[IDX_NHI] - k[1019]*y[IDX_NO2I] -
        k[1020]*y[IDX_NO2I] - k[1021]*y[IDX_NO2I] - k[1022]*y[IDX_NOI] -
        k[1023]*y[IDX_O2I] - k[1024]*y[IDX_O2HI] - k[1025]*y[IDX_OHI] -
        k[1026]*y[IDX_OHI] - k[1027]*y[IDX_SiCI] - k[1192]*y[IDX_CII] -
        k[1194]*y[IDX_CI] - k[1210]*y[IDX_NII] - k[1290];
    data[2313] = 0.0 + k[169]*y[IDX_NII] + k[768]*y[IDX_NHII] - k[1025]*y[IDX_NI] -
        k[1026]*y[IDX_NI] + k[1048]*y[IDX_NHI];
    data[2314] = 0.0 + k[168]*y[IDX_NII] + k[766]*y[IDX_NHII] - k[1023]*y[IDX_NI];
    data[2315] = 0.0 + k[390]*y[IDX_NHII] + k[865]*y[IDX_N2I] + k[870]*y[IDX_NHI] +
        k[872]*y[IDX_NOI] - k[1194]*y[IDX_NI];
    data[2316] = 0.0 + k[160]*y[IDX_NII] + k[754]*y[IDX_NHII];
    data[2317] = 0.0 + k[767]*y[IDX_NHII] + k[825]*y[IDX_N2II] + k[1047]*y[IDX_NHI] +
        k[1057]*y[IDX_CNI] + k[1071]*y[IDX_N2I] + k[1076]*y[IDX_NOI];
    data[2318] = 0.0 + k[303]*y[IDX_CNII] + k[337]*y[IDX_N2II] + k[337]*y[IDX_N2II] +
        k[339]*y[IDX_N2HII] + k[340]*y[IDX_NHII] + k[341]*y[IDX_NH2II] +
        k[345]*y[IDX_NOII] + k[1220]*y[IDX_NII];
    data[2319] = 0.0 + k[158]*y[IDX_NII] + k[751]*y[IDX_NHII];
    data[2320] = 0.0 + k[540]*y[IDX_NHII] - k[960]*y[IDX_NI];
    data[2321] = 0.0 + k[985]*y[IDX_NHI] + k[988]*y[IDX_NOI];
    data[2322] = 0.0 + k[937]*y[IDX_CHI] + k[954]*y[IDX_COI] + k[992]*y[IDX_HI] +
        k[992]*y[IDX_HI] + k[1077]*y[IDX_OI] - k[1100]*y[IDX_OHI] + k[1171];
    data[2323] = 0.0 + k[504]*y[IDX_HII] + k[595]*y[IDX_H3II] + k[986]*y[IDX_HI];
    data[2324] = 0.0 + k[862]*y[IDX_O2I];
    data[2325] = 0.0 + k[995]*y[IDX_HI];
    data[2326] = 0.0 + k[363]*y[IDX_eM];
    data[2327] = 0.0 + k[1088]*y[IDX_OI];
    data[2328] = 0.0 + k[333]*y[IDX_eM];
    data[2329] = 0.0 + k[982]*y[IDX_HI] + k[1069]*y[IDX_OI] - k[1097]*y[IDX_OHI];
    data[2330] = 0.0 + k[248] + k[657]*y[IDX_HeII] + k[809]*y[IDX_OII] + k[1121];
    data[2331] = 0.0 + k[822]*y[IDX_OI];
    data[2332] = 0.0 - k[225]*y[IDX_OHI] + k[560]*y[IDX_H2OI];
    data[2333] = 0.0 - k[857]*y[IDX_OHI];
    data[2334] = 0.0 - k[858]*y[IDX_OHI];
    data[2335] = 0.0 - k[859]*y[IDX_OHI];
    data[2336] = 0.0 - k[1102]*y[IDX_OHI];
    data[2337] = 0.0 + k[559]*y[IDX_H2OII];
    data[2338] = 0.0 - k[856]*y[IDX_OHI];
    data[2339] = 0.0 - k[227]*y[IDX_OHI] + k[569]*y[IDX_H2OI];
    data[2340] = 0.0 + k[317]*y[IDX_eM];
    data[2341] = 0.0 + k[810]*y[IDX_OII] - k[922]*y[IDX_OHI] + k[1056]*y[IDX_OI];
    data[2342] = 0.0 - k[226]*y[IDX_OHI] + k[562]*y[IDX_H2OI] - k[851]*y[IDX_OHI];
    data[2343] = 0.0 - k[114]*y[IDX_OHI] - k[527]*y[IDX_OHI];
    data[2344] = 0.0 + k[222]*y[IDX_OHII] + k[1074]*y[IDX_OI] - k[1098]*y[IDX_OHI];
    data[2345] = 0.0 + k[904]*y[IDX_H2OI] + k[911]*y[IDX_O2I] - k[917]*y[IDX_OHI] -
        k[918]*y[IDX_OHI] - k[919]*y[IDX_OHI];
    data[2346] = 0.0 + k[971]*y[IDX_HI];
    data[2347] = 0.0 - k[169]*y[IDX_OHI];
    data[2348] = 0.0 - k[215]*y[IDX_OHI] + k[809]*y[IDX_CH3OHI] + k[810]*y[IDX_CH4I] +
        k[813]*y[IDX_H2COI] - k[819]*y[IDX_OHI];
    data[2349] = 0.0 - k[853]*y[IDX_OHI];
    data[2350] = 0.0 + k[772]*y[IDX_H2OI] + k[778]*y[IDX_O2I];
    data[2351] = 0.0 + k[757]*y[IDX_H2OI] + k[765]*y[IDX_O2I] - k[768]*y[IDX_OHI];
    data[2352] = 0.0 - k[445]*y[IDX_OHI];
    data[2353] = 0.0 + k[188]*y[IDX_OHII] + k[781]*y[IDX_H2OII] + k[1030]*y[IDX_NOI] -
        k[1031]*y[IDX_OHI] - k[1032]*y[IDX_OHI] + k[1073]*y[IDX_OI];
    data[2354] = 0.0 + k[420]*y[IDX_O2I];
    data[2355] = 0.0 + k[314]*y[IDX_eM] + k[383]*y[IDX_CI] + k[424]*y[IDX_CH2I] +
        k[460]*y[IDX_CHI] + k[553]*y[IDX_COI] + k[554]*y[IDX_H2COI] +
        k[555]*y[IDX_H2OI] + k[556]*y[IDX_HCNI] + k[558]*y[IDX_HCOI] +
        k[559]*y[IDX_HNCI] + k[781]*y[IDX_NH2I] - k[852]*y[IDX_OHI];
    data[2356] = 0.0 + k[324]*y[IDX_eM] + k[325]*y[IDX_eM];
    data[2357] = 0.0 - k[379]*y[IDX_OHI];
    data[2358] = 0.0 + k[556]*y[IDX_H2OII] + k[1063]*y[IDX_OI] - k[1094]*y[IDX_OHI] -
        k[1095]*y[IDX_OHI];
    data[2359] = 0.0 + k[411]*y[IDX_O2I] - k[415]*y[IDX_OHI];
    data[2360] = 0.0 + k[45]*y[IDX_OHII] + k[424]*y[IDX_H2OII] + k[887]*y[IDX_NOI] +
        k[893]*y[IDX_O2I] + k[897]*y[IDX_OI] - k[898]*y[IDX_OHI] -
        k[899]*y[IDX_OHI] - k[900]*y[IDX_OHI];
    data[2361] = 0.0 + k[1036]*y[IDX_H2OI] + k[1043]*y[IDX_NOI] + k[1045]*y[IDX_O2I] +
        k[1047]*y[IDX_OI] - k[1048]*y[IDX_OHI] - k[1049]*y[IDX_OHI] -
        k[1050]*y[IDX_OHI];
    data[2362] = 0.0 + k[45]*y[IDX_CH2I] + k[62]*y[IDX_CHI] + k[188]*y[IDX_NH2I] +
        k[219]*y[IDX_H2COI] + k[220]*y[IDX_H2OI] + k[221]*y[IDX_HCOI] +
        k[222]*y[IDX_NH3I] + k[223]*y[IDX_NOI] + k[224]*y[IDX_O2I] -
        k[847]*y[IDX_OHI];
    data[2363] = 0.0 - k[1090]*y[IDX_OHI] - k[1091]*y[IDX_OHI];
    data[2364] = 0.0 + k[219]*y[IDX_OHII] + k[554]*y[IDX_H2OII] + k[813]*y[IDX_OII] +
        k[1061]*y[IDX_OI] - k[1093]*y[IDX_OHI];
    data[2365] = 0.0 + k[221]*y[IDX_OHII] + k[558]*y[IDX_H2OII] + k[1001]*y[IDX_O2I] +
        k[1067]*y[IDX_OI] - k[1096]*y[IDX_OHI];
    data[2366] = 0.0 + k[657]*y[IDX_CH3OHI] + k[674]*y[IDX_H2OI] - k[699]*y[IDX_OHI];
    data[2367] = 0.0 + k[62]*y[IDX_OHII] + k[460]*y[IDX_H2OII] + k[935]*y[IDX_O2I] +
        k[937]*y[IDX_O2HI] + k[940]*y[IDX_OI] - k[941]*y[IDX_OHI];
    data[2368] = 0.0 + k[595]*y[IDX_NO2I] - k[600]*y[IDX_OHI];
    data[2369] = 0.0 + k[223]*y[IDX_OHII] + k[887]*y[IDX_CH2I] + k[988]*y[IDX_HI] +
        k[1030]*y[IDX_NH2I] + k[1043]*y[IDX_NHI] - k[1099]*y[IDX_OHI];
    data[2370] = 0.0 - k[1025]*y[IDX_OHI] - k[1026]*y[IDX_OHI];
    data[2371] = 0.0 - k[7]*y[IDX_H2I] - k[13]*y[IDX_HI] - k[90]*y[IDX_HII] -
        k[114]*y[IDX_H2II] - k[169]*y[IDX_NII] - k[215]*y[IDX_OII] -
        k[225]*y[IDX_CNII] - k[226]*y[IDX_COII] - k[227]*y[IDX_N2II] - k[284] -
        k[379]*y[IDX_CII] - k[415]*y[IDX_CHII] - k[445]*y[IDX_CH3II] -
        k[527]*y[IDX_H2II] - k[600]*y[IDX_H3II] - k[699]*y[IDX_HeII] -
        k[768]*y[IDX_NHII] - k[819]*y[IDX_OII] - k[847]*y[IDX_OHII] -
        k[851]*y[IDX_COII] - k[852]*y[IDX_H2OII] - k[853]*y[IDX_HCNII] -
        k[854]*y[IDX_HCOII] - k[855]*y[IDX_HCOII] - k[856]*y[IDX_HNOII] -
        k[857]*y[IDX_N2HII] - k[858]*y[IDX_O2HII] - k[859]*y[IDX_SiII] -
        k[875]*y[IDX_CI] - k[876]*y[IDX_CI] - k[898]*y[IDX_CH2I] -
        k[899]*y[IDX_CH2I] - k[900]*y[IDX_CH2I] - k[917]*y[IDX_CH3I] -
        k[918]*y[IDX_CH3I] - k[919]*y[IDX_CH3I] - k[922]*y[IDX_CH4I] -
        k[941]*y[IDX_CHI] - k[966]*y[IDX_H2I] - k[996]*y[IDX_HI] -
        k[1025]*y[IDX_NI] - k[1026]*y[IDX_NI] - k[1031]*y[IDX_NH2I] -
        k[1032]*y[IDX_NH2I] - k[1048]*y[IDX_NHI] - k[1049]*y[IDX_NHI] -
        k[1050]*y[IDX_NHI] - k[1080]*y[IDX_OI] - k[1090]*y[IDX_CNI] -
        k[1091]*y[IDX_CNI] - k[1092]*y[IDX_COI] - k[1093]*y[IDX_H2COI] -
        k[1094]*y[IDX_HCNI] - k[1095]*y[IDX_HCNI] - k[1096]*y[IDX_HCOI] -
        k[1097]*y[IDX_HNOI] - k[1098]*y[IDX_NH3I] - k[1099]*y[IDX_NOI] -
        k[1100]*y[IDX_O2HI] - k[1101]*y[IDX_OHI] - k[1101]*y[IDX_OHI] -
        k[1101]*y[IDX_OHI] - k[1101]*y[IDX_OHI] - k[1102]*y[IDX_SiI] - k[1174] -
        k[1175] - k[1208]*y[IDX_HI] - k[1254];
    data[2372] = 0.0 + k[224]*y[IDX_OHII] + k[411]*y[IDX_CHII] + k[420]*y[IDX_CH2II] +
        k[765]*y[IDX_NHII] + k[778]*y[IDX_NH2II] + k[862]*y[IDX_SiH2II] +
        k[893]*y[IDX_CH2I] + k[911]*y[IDX_CH3I] + k[935]*y[IDX_CHI] +
        k[964]*y[IDX_H2I] + k[964]*y[IDX_H2I] + k[989]*y[IDX_HI] +
        k[1001]*y[IDX_HCOI] + k[1045]*y[IDX_NHI];
    data[2373] = 0.0 + k[383]*y[IDX_H2OII] - k[875]*y[IDX_OHI] - k[876]*y[IDX_OHI];
    data[2374] = 0.0 - k[90]*y[IDX_OHI] + k[504]*y[IDX_NO2I];
    data[2375] = 0.0 - k[854]*y[IDX_OHI] - k[855]*y[IDX_OHI];
    data[2376] = 0.0 + k[4]*y[IDX_H2I] + k[11]*y[IDX_HI] + k[220]*y[IDX_OHII] + k[256] +
        k[555]*y[IDX_H2OII] + k[560]*y[IDX_CNII] + k[562]*y[IDX_COII] +
        k[569]*y[IDX_N2II] + k[674]*y[IDX_HeII] + k[757]*y[IDX_NHII] +
        k[772]*y[IDX_NH2II] + k[904]*y[IDX_CH3I] + k[975]*y[IDX_HI] +
        k[1036]*y[IDX_NHI] + k[1062]*y[IDX_OI] + k[1062]*y[IDX_OI] + k[1142];
    data[2377] = 0.0 + k[822]*y[IDX_CH4II] + k[897]*y[IDX_CH2I] + k[940]*y[IDX_CHI] +
        k[965]*y[IDX_H2I] + k[1047]*y[IDX_NHI] + k[1056]*y[IDX_CH4I] +
        k[1061]*y[IDX_H2COI] + k[1062]*y[IDX_H2OI] + k[1062]*y[IDX_H2OI] +
        k[1063]*y[IDX_HCNI] + k[1067]*y[IDX_HCOI] + k[1069]*y[IDX_HNOI] +
        k[1073]*y[IDX_NH2I] + k[1074]*y[IDX_NH3I] + k[1077]*y[IDX_O2HI] -
        k[1080]*y[IDX_OHI] + k[1088]*y[IDX_SiH4I] + k[1207]*y[IDX_HI];
    data[2378] = 0.0 + k[314]*y[IDX_H2OII] + k[317]*y[IDX_H3COII] + k[324]*y[IDX_H3OII] +
        k[325]*y[IDX_H3OII] + k[333]*y[IDX_HCO2II] + k[363]*y[IDX_SiOHII];
    data[2379] = 0.0 + k[553]*y[IDX_H2OII] + k[954]*y[IDX_O2HI] + k[972]*y[IDX_HI] -
        k[1092]*y[IDX_OHI];
    data[2380] = 0.0 + k[4]*y[IDX_H2OI] - k[7]*y[IDX_OHI] + k[964]*y[IDX_O2I] +
        k[964]*y[IDX_O2I] + k[965]*y[IDX_OI] - k[966]*y[IDX_OHI];
    data[2381] = 0.0 + k[11]*y[IDX_H2OI] - k[13]*y[IDX_OHI] + k[971]*y[IDX_CO2I] +
        k[972]*y[IDX_COI] + k[975]*y[IDX_H2OI] + k[982]*y[IDX_HNOI] +
        k[986]*y[IDX_NO2I] + k[988]*y[IDX_NOI] + k[989]*y[IDX_O2I] +
        k[992]*y[IDX_O2HI] + k[992]*y[IDX_O2HI] + k[995]*y[IDX_OCNI] -
        k[996]*y[IDX_OHI] + k[1207]*y[IDX_OI] - k[1208]*y[IDX_OHI];
    data[2382] = 0.0 + k[1355] + k[1356] + k[1357] + k[1358];
    data[2383] = 0.0 + k[281] + k[914]*y[IDX_CH3I] + k[938]*y[IDX_CHI] + k[991]*y[IDX_HI]
        + k[1003]*y[IDX_HCOI] + k[1024]*y[IDX_NI] + k[1077]*y[IDX_OI] +
        k[1100]*y[IDX_OHI] + k[1170];
    data[2384] = 0.0 + k[818]*y[IDX_OII] + k[1021]*y[IDX_NI] + k[1075]*y[IDX_OI];
    data[2385] = 0.0 - k[862]*y[IDX_O2I];
    data[2386] = 0.0 - k[1054]*y[IDX_O2I] - k[1055]*y[IDX_O2I] + k[1079]*y[IDX_OI];
    data[2387] = 0.0 + k[835]*y[IDX_OI];
    data[2388] = 0.0 + k[824]*y[IDX_OI];
    data[2389] = 0.0 + k[1070]*y[IDX_OI];
    data[2390] = 0.0 + k[820]*y[IDX_O2II];
    data[2391] = 0.0 + k[152]*y[IDX_O2II];
    data[2392] = 0.0 - k[51]*y[IDX_O2I];
    data[2393] = 0.0 - k[68]*y[IDX_O2I] - k[481]*y[IDX_O2I];
    data[2394] = 0.0 + k[347]*y[IDX_eM] + k[392]*y[IDX_CI] + k[436]*y[IDX_CH2I] +
        k[474]*y[IDX_CHI] + k[483]*y[IDX_CNI] + k[488]*y[IDX_COI] +
        k[544]*y[IDX_H2I] + k[552]*y[IDX_H2COI] + k[571]*y[IDX_H2OI] +
        k[632]*y[IDX_HCNI] + k[645]*y[IDX_HCOI] + k[651]*y[IDX_HNCI] +
        k[733]*y[IDX_N2I] + k[790]*y[IDX_NH2I] + k[805]*y[IDX_NHI] +
        k[807]*y[IDX_NOI] + k[821]*y[IDX_CO2I] + k[829]*y[IDX_OI] +
        k[858]*y[IDX_OHI];
    data[2395] = 0.0 + k[230]*y[IDX_O2II] - k[1106]*y[IDX_O2I];
    data[2396] = 0.0 + k[651]*y[IDX_O2HII];
    data[2397] = 0.0 - k[173]*y[IDX_O2I];
    data[2398] = 0.0 - k[921]*y[IDX_O2I];
    data[2399] = 0.0 - k[73]*y[IDX_O2I];
    data[2400] = 0.0 - k[113]*y[IDX_O2I] - k[525]*y[IDX_O2I];
    data[2401] = 0.0 + k[198]*y[IDX_O2II];
    data[2402] = 0.0 - k[911]*y[IDX_O2I] - k[912]*y[IDX_O2I] - k[913]*y[IDX_O2I] +
        k[914]*y[IDX_O2HI];
    data[2403] = 0.0 + k[668]*y[IDX_HeII] + k[821]*y[IDX_O2HII] + k[1059]*y[IDX_OI];
    data[2404] = 0.0 - k[168]*y[IDX_O2I] - k[728]*y[IDX_O2I] - k[729]*y[IDX_O2I];
    data[2405] = 0.0 - k[214]*y[IDX_O2I] + k[818]*y[IDX_NO2I];
    data[2406] = 0.0 - k[133]*y[IDX_O2I];
    data[2407] = 0.0 - k[777]*y[IDX_O2I] - k[778]*y[IDX_O2I];
    data[2408] = 0.0 - k[179]*y[IDX_O2I] - k[765]*y[IDX_O2I] - k[766]*y[IDX_O2I];
    data[2409] = 0.0 + k[30]*y[IDX_CI] + k[44]*y[IDX_CH2I] + k[61]*y[IDX_CHI] +
        k[116]*y[IDX_H2COI] + k[137]*y[IDX_HCOI] + k[152]*y[IDX_MgI] +
        k[187]*y[IDX_NH2I] + k[198]*y[IDX_NH3I] + k[205]*y[IDX_NOI] +
        k[230]*y[IDX_SiI] + k[551]*y[IDX_H2COI] + k[820]*y[IDX_CH3OHI];
    data[2410] = 0.0 - k[442]*y[IDX_O2I];
    data[2411] = 0.0 + k[187]*y[IDX_O2II] + k[790]*y[IDX_O2HII];
    data[2412] = 0.0 - k[420]*y[IDX_O2I];
    data[2413] = 0.0 - k[121]*y[IDX_O2I];
    data[2414] = 0.0 + k[733]*y[IDX_O2HII];
    data[2415] = 0.0 - k[376]*y[IDX_O2I] - k[377]*y[IDX_O2I];
    data[2416] = 0.0 + k[632]*y[IDX_O2HII];
    data[2417] = 0.0 - k[411]*y[IDX_O2I] - k[412]*y[IDX_O2I] - k[413]*y[IDX_O2I];
    data[2418] = 0.0 + k[44]*y[IDX_O2II] + k[436]*y[IDX_O2HII] - k[889]*y[IDX_O2I] -
        k[890]*y[IDX_O2I] - k[891]*y[IDX_O2I] - k[892]*y[IDX_O2I] -
        k[893]*y[IDX_O2I];
    data[2419] = 0.0 - k[549]*y[IDX_O2I];
    data[2420] = 0.0 + k[805]*y[IDX_O2HII] - k[1044]*y[IDX_O2I] - k[1045]*y[IDX_O2I];
    data[2421] = 0.0 - k[224]*y[IDX_O2I];
    data[2422] = 0.0 + k[483]*y[IDX_O2HII] - k[948]*y[IDX_O2I] - k[949]*y[IDX_O2I];
    data[2423] = 0.0 + k[116]*y[IDX_O2II] + k[551]*y[IDX_O2II] + k[552]*y[IDX_O2HII];
    data[2424] = 0.0 + k[137]*y[IDX_O2II] + k[645]*y[IDX_O2HII] - k[1001]*y[IDX_O2I] -
        k[1002]*y[IDX_O2I] + k[1003]*y[IDX_O2HI];
    data[2425] = 0.0 - k[146]*y[IDX_O2I] + k[668]*y[IDX_CO2I] - k[696]*y[IDX_O2I];
    data[2426] = 0.0 + k[61]*y[IDX_O2II] + k[474]*y[IDX_O2HII] - k[933]*y[IDX_O2I] -
        k[934]*y[IDX_O2I] - k[935]*y[IDX_O2I] - k[936]*y[IDX_O2I] +
        k[938]*y[IDX_O2HI];
    data[2427] = 0.0 - k[597]*y[IDX_O2I];
    data[2428] = 0.0 + k[205]*y[IDX_O2II] + k[807]*y[IDX_O2HII] + k[1051]*y[IDX_NOI] +
        k[1051]*y[IDX_NOI] - k[1052]*y[IDX_O2I] + k[1076]*y[IDX_OI];
    data[2429] = 0.0 + k[1021]*y[IDX_NO2I] - k[1023]*y[IDX_O2I] + k[1024]*y[IDX_O2HI];
    data[2430] = 0.0 + k[858]*y[IDX_O2HII] + k[1080]*y[IDX_OI] + k[1100]*y[IDX_O2HI];
    data[2431] = 0.0 - k[6]*y[IDX_H2I] - k[12]*y[IDX_HI] - k[51]*y[IDX_CH4II] -
        k[68]*y[IDX_CNII] - k[73]*y[IDX_COII] - k[88]*y[IDX_HII] -
        k[113]*y[IDX_H2II] - k[121]*y[IDX_H2OII] - k[133]*y[IDX_HCNII] -
        k[146]*y[IDX_HeII] - k[168]*y[IDX_NII] - k[173]*y[IDX_N2II] -
        k[179]*y[IDX_NHII] - k[214]*y[IDX_OII] - k[224]*y[IDX_OHII] - k[279] -
        k[280] - k[376]*y[IDX_CII] - k[377]*y[IDX_CII] - k[411]*y[IDX_CHII] -
        k[412]*y[IDX_CHII] - k[413]*y[IDX_CHII] - k[420]*y[IDX_CH2II] -
        k[442]*y[IDX_CH3II] - k[481]*y[IDX_CNII] - k[525]*y[IDX_H2II] -
        k[549]*y[IDX_H2COII] - k[597]*y[IDX_H3II] - k[696]*y[IDX_HeII] -
        k[728]*y[IDX_NII] - k[729]*y[IDX_NII] - k[765]*y[IDX_NHII] -
        k[766]*y[IDX_NHII] - k[777]*y[IDX_NH2II] - k[778]*y[IDX_NH2II] -
        k[862]*y[IDX_SiH2II] - k[873]*y[IDX_CI] - k[889]*y[IDX_CH2I] -
        k[890]*y[IDX_CH2I] - k[891]*y[IDX_CH2I] - k[892]*y[IDX_CH2I] -
        k[893]*y[IDX_CH2I] - k[911]*y[IDX_CH3I] - k[912]*y[IDX_CH3I] -
        k[913]*y[IDX_CH3I] - k[921]*y[IDX_CH4I] - k[933]*y[IDX_CHI] -
        k[934]*y[IDX_CHI] - k[935]*y[IDX_CHI] - k[936]*y[IDX_CHI] -
        k[948]*y[IDX_CNI] - k[949]*y[IDX_CNI] - k[953]*y[IDX_COI] -
        k[963]*y[IDX_H2I] - k[964]*y[IDX_H2I] - k[989]*y[IDX_HI] -
        k[1001]*y[IDX_HCOI] - k[1002]*y[IDX_HCOI] - k[1023]*y[IDX_NI] -
        k[1044]*y[IDX_NHI] - k[1045]*y[IDX_NHI] - k[1052]*y[IDX_NOI] -
        k[1054]*y[IDX_OCNI] - k[1055]*y[IDX_OCNI] - k[1106]*y[IDX_SiI] - k[1168]
        - k[1169] - k[1292];
    data[2432] = 0.0 + k[30]*y[IDX_O2II] + k[392]*y[IDX_O2HII] - k[873]*y[IDX_O2I];
    data[2433] = 0.0 - k[88]*y[IDX_O2I];
    data[2434] = 0.0 + k[571]*y[IDX_O2HII];
    data[2435] = 0.0 + k[824]*y[IDX_HCO2II] + k[829]*y[IDX_O2HII] + k[835]*y[IDX_SiOII] +
        k[1059]*y[IDX_CO2I] + k[1070]*y[IDX_HNOI] + k[1075]*y[IDX_NO2I] +
        k[1076]*y[IDX_NOI] + k[1077]*y[IDX_O2HI] + k[1079]*y[IDX_OCNI] +
        k[1080]*y[IDX_OHI] + k[1211]*y[IDX_OI] + k[1211]*y[IDX_OI];
    data[2436] = 0.0 + k[347]*y[IDX_O2HII];
    data[2437] = 0.0 + k[488]*y[IDX_O2HII] - k[953]*y[IDX_O2I];
    data[2438] = 0.0 - k[6]*y[IDX_O2I] + k[544]*y[IDX_O2HII] - k[963]*y[IDX_O2I] -
        k[964]*y[IDX_O2I];
    data[2439] = 0.0 - k[12]*y[IDX_O2I] - k[989]*y[IDX_O2I] + k[991]*y[IDX_O2HI];
    data[2440] = 0.0 + k[351]*y[IDX_eM];
    data[2441] = 0.0 - k[1004]*y[IDX_CI];
    data[2442] = 0.0 + k[350]*y[IDX_eM];
    data[2443] = 0.0 + k[22]*y[IDX_CII] + k[286];
    data[2444] = 0.0 + k[23]*y[IDX_CII] + k[287] + k[700]*y[IDX_HeII] + k[1177];
    data[2445] = 0.0 + k[349]*y[IDX_eM] + k[831]*y[IDX_OI];
    data[2446] = 0.0 + k[24]*y[IDX_CII] + k[288] + k[701]*y[IDX_HeII] + k[1084]*y[IDX_OI]
        + k[1178];
    data[2447] = 0.0 - k[874]*y[IDX_CI];
    data[2448] = 0.0 + k[25]*y[IDX_CII];
    data[2449] = 0.0 - k[394]*y[IDX_CI];
    data[2450] = 0.0 - k[877]*y[IDX_CI];
    data[2451] = 0.0 + k[26]*y[IDX_CII];
    data[2452] = 0.0 - k[395]*y[IDX_CI];
    data[2453] = 0.0 - k[387]*y[IDX_CI];
    data[2454] = 0.0 + k[18]*y[IDX_CII];
    data[2455] = 0.0 - k[27]*y[IDX_CI] + k[303]*y[IDX_eM] + k[737]*y[IDX_NI];
    data[2456] = 0.0 - k[389]*y[IDX_CI];
    data[2457] = 0.0 - k[392]*y[IDX_CI];
    data[2458] = 0.0 + k[21]*y[IDX_CII] + k[1104]*y[IDX_COI];
    data[2459] = 0.0 + k[407]*y[IDX_CHII] + k[685]*y[IDX_HeII];
    data[2460] = 0.0 - k[388]*y[IDX_CI];
    data[2461] = 0.0 - k[29]*y[IDX_CI];
    data[2462] = 0.0 - k[28]*y[IDX_CI] + k[304]*y[IDX_eM] + k[458]*y[IDX_CHI];
    data[2463] = 0.0 - k[509]*y[IDX_CI];
    data[2464] = 0.0 + k[19]*y[IDX_CII];
    data[2465] = 0.0 + k[667]*y[IDX_HeII];
    data[2466] = 0.0 + k[720]*y[IDX_COI];
    data[2467] = 0.0 + k[811]*y[IDX_CNI] - k[1195]*y[IDX_CI];
    data[2468] = 0.0 - k[385]*y[IDX_CI];
    data[2469] = 0.0 - k[390]*y[IDX_CI];
    data[2470] = 0.0 - k[30]*y[IDX_CI] - k[391]*y[IDX_CI];
    data[2471] = 0.0 - k[866]*y[IDX_CI] - k[867]*y[IDX_CI] - k[868]*y[IDX_CI];
    data[2472] = 0.0 + k[295]*y[IDX_eM] + k[296]*y[IDX_eM];
    data[2473] = 0.0 - k[383]*y[IDX_CI];
    data[2474] = 0.0 - k[384]*y[IDX_CI];
    data[2475] = 0.0 - k[865]*y[IDX_CI];
    data[2476] = 0.0 + k[14]*y[IDX_CH2I] + k[15]*y[IDX_CHI] + k[16]*y[IDX_H2COI] +
        k[17]*y[IDX_HCOI] + k[18]*y[IDX_MgI] + k[19]*y[IDX_NH3I] +
        k[20]*y[IDX_NOI] + k[21]*y[IDX_SiI] + k[22]*y[IDX_SiC2I] +
        k[23]*y[IDX_SiC3I] + k[24]*y[IDX_SiCI] + k[25]*y[IDX_SiH2I] +
        k[26]*y[IDX_SiH3I] + k[1214]*y[IDX_eM];
    data[2477] = 0.0 + k[405]*y[IDX_CHII];
    data[2478] = 0.0 + k[294]*y[IDX_eM] + k[400]*y[IDX_H2COI] + k[403]*y[IDX_H2OI] +
        k[405]*y[IDX_HCNI] + k[407]*y[IDX_HNCI] + k[1108];
    data[2479] = 0.0 + k[14]*y[IDX_CII] - k[863]*y[IDX_CI];
    data[2480] = 0.0 - k[869]*y[IDX_CI] - k[870]*y[IDX_CI];
    data[2481] = 0.0 - k[393]*y[IDX_CI];
    data[2482] = 0.0 + k[251] + k[663]*y[IDX_HeII] + k[811]*y[IDX_OII] +
        k[1011]*y[IDX_NI] + k[1058]*y[IDX_OI] + k[1130];
    data[2483] = 0.0 + k[16]*y[IDX_CII] + k[400]*y[IDX_CHII];
    data[2484] = 0.0 + k[17]*y[IDX_CII] - k[864]*y[IDX_CI];
    data[2485] = 0.0 - k[139]*y[IDX_CI] + k[663]*y[IDX_CNI] + k[667]*y[IDX_CO2I] +
        k[685]*y[IDX_HNCI] + k[700]*y[IDX_SiC3I] + k[701]*y[IDX_SiCI];
    data[2486] = 0.0 + k[2]*y[IDX_H2I] + k[9]*y[IDX_HI] + k[15]*y[IDX_CII] + k[250] +
        k[458]*y[IDX_COII] + k[929]*y[IDX_NI] + k[940]*y[IDX_OI] +
        k[970]*y[IDX_HI] + k[1128];
    data[2487] = 0.0 - k[576]*y[IDX_CI];
    data[2488] = 0.0 + k[20]*y[IDX_CII] - k[871]*y[IDX_CI] - k[872]*y[IDX_CI];
    data[2489] = 0.0 + k[737]*y[IDX_CNII] + k[929]*y[IDX_CHI] + k[1011]*y[IDX_CNI] -
        k[1194]*y[IDX_CI];
    data[2490] = 0.0 - k[875]*y[IDX_CI] - k[876]*y[IDX_CI];
    data[2491] = 0.0 - k[873]*y[IDX_CI];
    data[2492] = 0.0 - k[27]*y[IDX_CNII] - k[28]*y[IDX_COII] - k[29]*y[IDX_N2II] -
        k[30]*y[IDX_O2II] - k[139]*y[IDX_HeII] - k[231] - k[240] -
        k[383]*y[IDX_H2OII] - k[384]*y[IDX_H3OII] - k[385]*y[IDX_HCNII] -
        k[386]*y[IDX_HCOII] - k[387]*y[IDX_HCO2II] - k[388]*y[IDX_HNOII] -
        k[389]*y[IDX_N2HII] - k[390]*y[IDX_NHII] - k[391]*y[IDX_O2II] -
        k[392]*y[IDX_O2HII] - k[393]*y[IDX_OHII] - k[394]*y[IDX_SiHII] -
        k[395]*y[IDX_SiOII] - k[509]*y[IDX_H2II] - k[576]*y[IDX_H3II] -
        k[863]*y[IDX_CH2I] - k[864]*y[IDX_HCOI] - k[865]*y[IDX_N2I] -
        k[866]*y[IDX_NH2I] - k[867]*y[IDX_NH2I] - k[868]*y[IDX_NH2I] -
        k[869]*y[IDX_NHI] - k[870]*y[IDX_NHI] - k[871]*y[IDX_NOI] -
        k[872]*y[IDX_NOI] - k[873]*y[IDX_O2I] - k[874]*y[IDX_OCNI] -
        k[875]*y[IDX_OHI] - k[876]*y[IDX_OHI] - k[877]*y[IDX_SiHI] -
        k[955]*y[IDX_H2I] - k[1004]*y[IDX_HNCOI] - k[1107] - k[1194]*y[IDX_NI] -
        k[1195]*y[IDX_OII] - k[1196]*y[IDX_OI] - k[1200]*y[IDX_H2I] -
        k[1206]*y[IDX_HI] - k[1250];
    data[2493] = 0.0 - k[386]*y[IDX_CI];
    data[2494] = 0.0 + k[403]*y[IDX_CHII];
    data[2495] = 0.0 + k[831]*y[IDX_SiCII] + k[940]*y[IDX_CHI] + k[1058]*y[IDX_CNI] +
        k[1084]*y[IDX_SiCI] - k[1196]*y[IDX_CI];
    data[2496] = 0.0 + k[294]*y[IDX_CHII] + k[295]*y[IDX_CH2II] + k[296]*y[IDX_CH2II] +
        k[303]*y[IDX_CNII] + k[304]*y[IDX_COII] + k[349]*y[IDX_SiCII] +
        k[350]*y[IDX_SiC2II] + k[351]*y[IDX_SiC3II] + k[1214]*y[IDX_CII];
    data[2497] = 0.0 + k[253] + k[720]*y[IDX_NII] + k[972]*y[IDX_HI] + k[1104]*y[IDX_SiI]
        + k[1133];
    data[2498] = 0.0 + k[2]*y[IDX_CHI] - k[955]*y[IDX_CI] - k[1200]*y[IDX_CI];
    data[2499] = 0.0 + k[9]*y[IDX_CHI] + k[970]*y[IDX_CHI] + k[972]*y[IDX_COI] -
        k[1206]*y[IDX_CI];
    data[2500] = 0.0 - k[499]*y[IDX_HII];
    data[2501] = 0.0 - k[502]*y[IDX_HII];
    data[2502] = 0.0 - k[92]*y[IDX_HII];
    data[2503] = 0.0 - k[93]*y[IDX_HII];
    data[2504] = 0.0 - k[94]*y[IDX_HII];
    data[2505] = 0.0 - k[504]*y[IDX_HII];
    data[2506] = 0.0 - k[95]*y[IDX_HII] - k[505]*y[IDX_HII];
    data[2507] = 0.0 - k[97]*y[IDX_HII] - k[507]*y[IDX_HII];
    data[2508] = 0.0 - k[98]*y[IDX_HII] - k[508]*y[IDX_HII];
    data[2509] = 0.0 - k[96]*y[IDX_HII] - k[506]*y[IDX_HII];
    data[2510] = 0.0 - k[503]*y[IDX_HII] + k[687]*y[IDX_HeII];
    data[2511] = 0.0 - k[492]*y[IDX_HII] - k[493]*y[IDX_HII] - k[494]*y[IDX_HII];
    data[2512] = 0.0 - k[83]*y[IDX_HII];
    data[2513] = 0.0 - k[99]*y[IDX_HII];
    data[2514] = 0.0 + k[126]*y[IDX_HI];
    data[2515] = 0.0 - k[91]*y[IDX_HII];
    data[2516] = 0.0 - k[1]*y[IDX_HII] + k[1]*y[IDX_HII];
    data[2517] = 0.0 - k[77]*y[IDX_HII] - k[495]*y[IDX_HII] + k[661]*y[IDX_HeII];
    data[2518] = 0.0 + k[127]*y[IDX_HI];
    data[2519] = 0.0 + k[128]*y[IDX_HI] + k[1134];
    data[2520] = 0.0 - k[85]*y[IDX_HII];
    data[2521] = 0.0 - k[76]*y[IDX_HII];
    data[2522] = 0.0 - k[496]*y[IDX_HII];
    data[2523] = 0.0 + k[131]*y[IDX_HI];
    data[2524] = 0.0 + k[129]*y[IDX_HI];
    data[2525] = 0.0 + k[1156];
    data[2526] = 0.0 - k[84]*y[IDX_HII];
    data[2527] = 0.0 + k[1111];
    data[2528] = 0.0 - k[81]*y[IDX_HII];
    data[2529] = 0.0 + k[1108];
    data[2530] = 0.0 - k[75]*y[IDX_HII] - k[491]*y[IDX_HII];
    data[2531] = 0.0 - k[86]*y[IDX_HII];
    data[2532] = 0.0 - k[79]*y[IDX_HII] - k[497]*y[IDX_HII] - k[498]*y[IDX_HII];
    data[2533] = 0.0 - k[82]*y[IDX_HII] - k[500]*y[IDX_HII] - k[501]*y[IDX_HII];
    data[2534] = 0.0 + k[130]*y[IDX_HI] + k[536]*y[IDX_H2I] + k[661]*y[IDX_CH4I] +
        k[674]*y[IDX_H2OI] + k[687]*y[IDX_HNOI];
    data[2535] = 0.0 - k[78]*y[IDX_HII];
    data[2536] = 0.0 + k[1146];
    data[2537] = 0.0 - k[1198]*y[IDX_HII];
    data[2538] = 0.0 - k[87]*y[IDX_HII];
    data[2539] = 0.0 - k[90]*y[IDX_HII];
    data[2540] = 0.0 - k[88]*y[IDX_HII];
    data[2541] = 0.0 - k[1]*y[IDX_HNCI] + k[1]*y[IDX_HNCI] - k[75]*y[IDX_CH2I] -
        k[76]*y[IDX_CH3I] - k[77]*y[IDX_CH4I] - k[78]*y[IDX_CHI] -
        k[79]*y[IDX_H2COI] - k[80]*y[IDX_H2OI] - k[81]*y[IDX_HCNI] -
        k[82]*y[IDX_HCOI] - k[83]*y[IDX_MgI] - k[84]*y[IDX_NH2I] -
        k[85]*y[IDX_NH3I] - k[86]*y[IDX_NHI] - k[87]*y[IDX_NOI] -
        k[88]*y[IDX_O2I] - k[89]*y[IDX_OI] - k[90]*y[IDX_OHI] - k[91]*y[IDX_SiI]
        - k[92]*y[IDX_SiC2I] - k[93]*y[IDX_SiC3I] - k[94]*y[IDX_SiCI] -
        k[95]*y[IDX_SiH2I] - k[96]*y[IDX_SiH3I] - k[97]*y[IDX_SiH4I] -
        k[98]*y[IDX_SiHI] - k[99]*y[IDX_SiOI] - k[491]*y[IDX_CH2I] -
        k[492]*y[IDX_CH3OHI] - k[493]*y[IDX_CH3OHI] - k[494]*y[IDX_CH3OHI] -
        k[495]*y[IDX_CH4I] - k[496]*y[IDX_CO2I] - k[497]*y[IDX_H2COI] -
        k[498]*y[IDX_H2COI] - k[499]*y[IDX_H2SiOI] - k[500]*y[IDX_HCOI] -
        k[501]*y[IDX_HCOI] - k[502]*y[IDX_HNCOI] - k[503]*y[IDX_HNOI] -
        k[504]*y[IDX_NO2I] - k[505]*y[IDX_SiH2I] - k[506]*y[IDX_SiH3I] -
        k[507]*y[IDX_SiH4I] - k[508]*y[IDX_SiHI] - k[1197]*y[IDX_HI] -
        k[1198]*y[IDX_HeI] - k[1216]*y[IDX_eM];
    data[2542] = 0.0 - k[80]*y[IDX_HII] + k[674]*y[IDX_HeII];
    data[2543] = 0.0 - k[89]*y[IDX_HII];
    data[2544] = 0.0 - k[1216]*y[IDX_HII];
    data[2545] = 0.0 + k[233] + k[536]*y[IDX_HeII];
    data[2546] = 0.0 + k[126]*y[IDX_CNII] + k[127]*y[IDX_COII] + k[128]*y[IDX_H2II] +
        k[129]*y[IDX_HCNII] + k[130]*y[IDX_HeII] + k[131]*y[IDX_OII] + k[236] +
        k[258] - k[1197]*y[IDX_HII];
    data[2547] = 0.0 + k[5]*y[IDX_H2I];
    data[2548] = 0.0 + k[489]*y[IDX_COI];
    data[2549] = 0.0 - k[637]*y[IDX_HCOII];
    data[2550] = 0.0 - k[638]*y[IDX_HCOII];
    data[2551] = 0.0 - k[639]*y[IDX_HCOII];
    data[2552] = 0.0 + k[138]*y[IDX_HCOI] + k[478]*y[IDX_CHI];
    data[2553] = 0.0 + k[485]*y[IDX_COI] + k[824]*y[IDX_OI];
    data[2554] = 0.0 + k[494]*y[IDX_HII];
    data[2555] = 0.0 - k[149]*y[IDX_HCOII];
    data[2556] = 0.0 + k[448]*y[IDX_COI];
    data[2557] = 0.0 - k[640]*y[IDX_HCOII];
    data[2558] = 0.0 + k[66]*y[IDX_HCOI] + k[479]*y[IDX_H2COI] + k[561]*y[IDX_H2OI];
    data[2559] = 0.0 + k[487]*y[IDX_COI];
    data[2560] = 0.0 + k[488]*y[IDX_COI];
    data[2561] = 0.0 - k[861]*y[IDX_HCOII];
    data[2562] = 0.0 - k[648]*y[IDX_HCOII];
    data[2563] = 0.0 + k[486]*y[IDX_COI];
    data[2564] = 0.0 + k[171]*y[IDX_HCOI] + k[730]*y[IDX_H2COI];
    data[2565] = 0.0 + k[451]*y[IDX_COII];
    data[2566] = 0.0 + k[71]*y[IDX_HCOI] + k[422]*y[IDX_CH2I] + k[451]*y[IDX_CH4I] +
        k[458]*y[IDX_CHI] + k[484]*y[IDX_H2COI] + k[532]*y[IDX_H2I] +
        k[562]*y[IDX_H2OI] + k[779]*y[IDX_NH2I] + k[792]*y[IDX_NH3I] +
        k[795]*y[IDX_NHI] + k[851]*y[IDX_OHI];
    data[2567] = 0.0 + k[108]*y[IDX_HCOI] + k[515]*y[IDX_COI] + k[517]*y[IDX_H2COI];
    data[2568] = 0.0 + k[792]*y[IDX_COII];
    data[2569] = 0.0 + k[398]*y[IDX_CHII] + k[496]*y[IDX_HII];
    data[2570] = 0.0 + k[162]*y[IDX_HCOI] + k[721]*y[IDX_H2COI];
    data[2571] = 0.0 + k[211]*y[IDX_HCOI] + k[813]*y[IDX_H2COI] + k[814]*y[IDX_HCNI];
    data[2572] = 0.0 + k[621]*y[IDX_COI];
    data[2573] = 0.0 + k[180]*y[IDX_HCOI];
    data[2574] = 0.0 + k[751]*y[IDX_COI] + k[753]*y[IDX_H2COI];
    data[2575] = 0.0 + k[137]*y[IDX_HCOI] + k[473]*y[IDX_CHI] + k[551]*y[IDX_H2COI];
    data[2576] = 0.0 + k[46]*y[IDX_HCOI] + k[440]*y[IDX_H2COI] + k[444]*y[IDX_OI];
    data[2577] = 0.0 + k[779]*y[IDX_COII] - k[787]*y[IDX_HCOII];
    data[2578] = 0.0 + k[417]*y[IDX_H2COI] + k[420]*y[IDX_O2I] + k[421]*y[IDX_OI];
    data[2579] = 0.0 + k[118]*y[IDX_HCOI] + k[553]*y[IDX_COI];
    data[2580] = 0.0 + k[189]*y[IDX_HCOI];
    data[2581] = 0.0 + k[384]*y[IDX_CI];
    data[2582] = 0.0 + k[17]*y[IDX_HCOI] + k[369]*y[IDX_H2COI] + k[370]*y[IDX_H2OI];
    data[2583] = 0.0 - k[629]*y[IDX_HCOII] + k[814]*y[IDX_OII];
    data[2584] = 0.0 + k[31]*y[IDX_HCOI] + k[398]*y[IDX_CO2I] + k[401]*y[IDX_H2COI] +
        k[404]*y[IDX_H2OI] + k[412]*y[IDX_O2I];
    data[2585] = 0.0 + k[422]*y[IDX_COII] - k[429]*y[IDX_HCOII];
    data[2586] = 0.0 + k[136]*y[IDX_HCOI] + k[549]*y[IDX_O2I];
    data[2587] = 0.0 + k[795]*y[IDX_COII] - k[799]*y[IDX_HCOII];
    data[2588] = 0.0 + k[221]*y[IDX_HCOI] + k[838]*y[IDX_COI];
    data[2589] = 0.0 + k[369]*y[IDX_CII] + k[401]*y[IDX_CHII] + k[417]*y[IDX_CH2II] +
        k[440]*y[IDX_CH3II] + k[479]*y[IDX_CNII] + k[484]*y[IDX_COII] +
        k[498]*y[IDX_HII] + k[517]*y[IDX_H2II] + k[551]*y[IDX_O2II] -
        k[635]*y[IDX_HCOII] + k[671]*y[IDX_HeII] + k[721]*y[IDX_NII] +
        k[730]*y[IDX_N2II] + k[753]*y[IDX_NHII] + k[813]*y[IDX_OII] + k[1139];
    data[2590] = 0.0 + k[17]*y[IDX_CII] + k[31]*y[IDX_CHII] + k[46]*y[IDX_CH3II] +
        k[66]*y[IDX_CNII] + k[71]*y[IDX_COII] + k[82]*y[IDX_HII] +
        k[108]*y[IDX_H2II] + k[118]*y[IDX_H2OII] + k[136]*y[IDX_H2COII] +
        k[137]*y[IDX_O2II] + k[138]*y[IDX_SiOII] + k[162]*y[IDX_NII] +
        k[171]*y[IDX_N2II] + k[180]*y[IDX_NH2II] + k[189]*y[IDX_NH3II] +
        k[211]*y[IDX_OII] + k[221]*y[IDX_OHII] + k[261] - k[636]*y[IDX_HCOII] +
        k[1150];
    data[2591] = 0.0 + k[671]*y[IDX_H2COI];
    data[2592] = 0.0 + k[0]*y[IDX_OI] + k[458]*y[IDX_COII] - k[466]*y[IDX_HCOII] +
        k[473]*y[IDX_O2II] + k[478]*y[IDX_SiOII];
    data[2593] = 0.0 + k[583]*y[IDX_COI];
    data[2594] = 0.0 + k[851]*y[IDX_COII] - k[854]*y[IDX_HCOII] - k[855]*y[IDX_HCOII];
    data[2595] = 0.0 + k[412]*y[IDX_CHII] + k[420]*y[IDX_CH2II] + k[549]*y[IDX_H2COII];
    data[2596] = 0.0 + k[384]*y[IDX_H3OII] - k[386]*y[IDX_HCOII];
    data[2597] = 0.0 + k[82]*y[IDX_HCOI] + k[494]*y[IDX_CH3OHI] + k[496]*y[IDX_CO2I] +
        k[498]*y[IDX_H2COI];
    data[2598] = 0.0 - k[149]*y[IDX_MgI] - k[330]*y[IDX_eM] - k[386]*y[IDX_CI] -
        k[429]*y[IDX_CH2I] - k[466]*y[IDX_CHI] - k[566]*y[IDX_H2OI] -
        k[629]*y[IDX_HCNI] - k[635]*y[IDX_H2COI] - k[636]*y[IDX_HCOI] -
        k[637]*y[IDX_SiH2I] - k[638]*y[IDX_SiH4I] - k[639]*y[IDX_SiHI] -
        k[640]*y[IDX_SiOI] - k[648]*y[IDX_HNCI] - k[787]*y[IDX_NH2I] -
        k[799]*y[IDX_NHI] - k[854]*y[IDX_OHI] - k[855]*y[IDX_OHI] -
        k[861]*y[IDX_SiI] - k[1148] - k[1281];
    data[2599] = 0.0 + k[370]*y[IDX_CII] + k[404]*y[IDX_CHII] + k[561]*y[IDX_CNII] +
        k[562]*y[IDX_COII] - k[566]*y[IDX_HCOII];
    data[2600] = 0.0 + k[0]*y[IDX_CHI] + k[421]*y[IDX_CH2II] + k[444]*y[IDX_CH3II] +
        k[824]*y[IDX_HCO2II];
    data[2601] = 0.0 - k[330]*y[IDX_HCOII];
    data[2602] = 0.0 + k[448]*y[IDX_CH4II] + k[485]*y[IDX_HCO2II] + k[486]*y[IDX_HNOII] +
        k[487]*y[IDX_N2HII] + k[488]*y[IDX_O2HII] + k[489]*y[IDX_SiH4II] +
        k[515]*y[IDX_H2II] + k[553]*y[IDX_H2OII] + k[583]*y[IDX_H3II] +
        k[621]*y[IDX_HCNII] + k[751]*y[IDX_NHII] + k[838]*y[IDX_OHII];
    data[2603] = 0.0 + k[5]*y[IDX_HOCII] + k[532]*y[IDX_COII];
    data[2604] = 0.0 + k[1315] + k[1316] + k[1317] + k[1318];
    data[2605] = 0.0 - k[575]*y[IDX_H2OI];
    data[2606] = 0.0 - k[574]*y[IDX_H2OI];
    data[2607] = 0.0 + k[990]*y[IDX_HI] + k[1100]*y[IDX_OHI];
    data[2608] = 0.0 + k[611]*y[IDX_H3OII];
    data[2609] = 0.0 - k[573]*y[IDX_H2OI];
    data[2610] = 0.0 + k[612]*y[IDX_H3OII];
    data[2611] = 0.0 - k[567]*y[IDX_H2OI];
    data[2612] = 0.0 + k[1097]*y[IDX_OHI];
    data[2613] = 0.0 + k[492]*y[IDX_HII] + k[579]*y[IDX_H3II] + k[808]*y[IDX_OII];
    data[2614] = 0.0 + k[119]*y[IDX_H2OII];
    data[2615] = 0.0 - k[450]*y[IDX_H2OI];
    data[2616] = 0.0 + k[613]*y[IDX_H3OII];
    data[2617] = 0.0 - k[560]*y[IDX_H2OI] - k[561]*y[IDX_H2OI];
    data[2618] = 0.0 - k[570]*y[IDX_H2OI];
    data[2619] = 0.0 - k[571]*y[IDX_H2OI];
    data[2620] = 0.0 - k[572]*y[IDX_H2OI];
    data[2621] = 0.0 + k[122]*y[IDX_H2OII] + k[610]*y[IDX_H3OII];
    data[2622] = 0.0 + k[609]*y[IDX_H3OII];
    data[2623] = 0.0 - k[568]*y[IDX_H2OI];
    data[2624] = 0.0 - k[125]*y[IDX_H2OI] - k[569]*y[IDX_H2OI];
    data[2625] = 0.0 + k[318]*y[IDX_eM] - k[564]*y[IDX_H2OI];
    data[2626] = 0.0 + k[922]*y[IDX_OHI];
    data[2627] = 0.0 - k[123]*y[IDX_H2OI] - k[562]*y[IDX_H2OI];
    data[2628] = 0.0 - k[106]*y[IDX_H2OI] - k[518]*y[IDX_H2OI];
    data[2629] = 0.0 + k[195]*y[IDX_H2OII] + k[1098]*y[IDX_OHI];
    data[2630] = 0.0 - k[904]*y[IDX_H2OI] + k[910]*y[IDX_NOI] + k[912]*y[IDX_O2I] +
        k[919]*y[IDX_OHI];
    data[2631] = 0.0 - k[160]*y[IDX_H2OI];
    data[2632] = 0.0 - k[210]*y[IDX_H2OI] + k[808]*y[IDX_CH3OHI];
    data[2633] = 0.0 - k[124]*y[IDX_H2OI] - k[565]*y[IDX_H2OI];
    data[2634] = 0.0 - k[771]*y[IDX_H2OI] - k[772]*y[IDX_H2OI];
    data[2635] = 0.0 - k[176]*y[IDX_H2OI] - k[754]*y[IDX_H2OI] - k[755]*y[IDX_H2OI] -
        k[756]*y[IDX_H2OI] - k[757]*y[IDX_H2OI];
    data[2636] = 0.0 + k[185]*y[IDX_H2OII] + k[783]*y[IDX_H3OII] + k[1029]*y[IDX_NOI] +
        k[1031]*y[IDX_OHI];
    data[2637] = 0.0 - k[418]*y[IDX_H2OI];
    data[2638] = 0.0 + k[40]*y[IDX_CH2I] + k[56]*y[IDX_CHI] + k[117]*y[IDX_H2COI] +
        k[118]*y[IDX_HCOI] + k[119]*y[IDX_MgI] + k[120]*y[IDX_NOI] +
        k[121]*y[IDX_O2I] + k[122]*y[IDX_SiI] + k[185]*y[IDX_NH2I] +
        k[195]*y[IDX_NH3I] - k[555]*y[IDX_H2OI];
    data[2639] = 0.0 + k[322]*y[IDX_eM] + k[425]*y[IDX_CH2I] + k[462]*y[IDX_CHI] +
        k[607]*y[IDX_H2COI] + k[608]*y[IDX_HCNI] + k[609]*y[IDX_HNCI] +
        k[610]*y[IDX_SiI] + k[611]*y[IDX_SiH2I] + k[612]*y[IDX_SiHI] +
        k[613]*y[IDX_SiOI] + k[783]*y[IDX_NH2I];
    data[2640] = 0.0 - k[370]*y[IDX_H2OI] - k[371]*y[IDX_H2OI];
    data[2641] = 0.0 + k[608]*y[IDX_H3OII] + k[1094]*y[IDX_OHI];
    data[2642] = 0.0 - k[402]*y[IDX_H2OI] - k[403]*y[IDX_H2OI] - k[404]*y[IDX_H2OI];
    data[2643] = 0.0 + k[40]*y[IDX_H2OII] + k[425]*y[IDX_H3OII] + k[891]*y[IDX_O2I] +
        k[899]*y[IDX_OHI];
    data[2644] = 0.0 - k[563]*y[IDX_H2OI];
    data[2645] = 0.0 - k[1036]*y[IDX_H2OI] + k[1048]*y[IDX_OHI];
    data[2646] = 0.0 - k[220]*y[IDX_H2OI] - k[840]*y[IDX_H2OI];
    data[2647] = 0.0 + k[117]*y[IDX_H2OII] + k[607]*y[IDX_H3OII] + k[1093]*y[IDX_OHI];
    data[2648] = 0.0 + k[118]*y[IDX_H2OII] + k[1096]*y[IDX_OHI];
    data[2649] = 0.0 - k[143]*y[IDX_H2OI] - k[673]*y[IDX_H2OI] - k[674]*y[IDX_H2OI];
    data[2650] = 0.0 + k[56]*y[IDX_H2OII] + k[462]*y[IDX_H3OII];
    data[2651] = 0.0 + k[579]*y[IDX_CH3OHI] - k[586]*y[IDX_H2OI];
    data[2652] = 0.0 + k[120]*y[IDX_H2OII] + k[910]*y[IDX_CH3I] + k[1029]*y[IDX_NH2I];
    data[2653] = 0.0 + k[899]*y[IDX_CH2I] + k[919]*y[IDX_CH3I] + k[922]*y[IDX_CH4I] +
        k[966]*y[IDX_H2I] + k[1031]*y[IDX_NH2I] + k[1048]*y[IDX_NHI] +
        k[1093]*y[IDX_H2COI] + k[1094]*y[IDX_HCNI] + k[1096]*y[IDX_HCOI] +
        k[1097]*y[IDX_HNOI] + k[1098]*y[IDX_NH3I] + k[1100]*y[IDX_O2HI] +
        k[1101]*y[IDX_OHI] + k[1101]*y[IDX_OHI] + k[1208]*y[IDX_HI];
    data[2654] = 0.0 + k[121]*y[IDX_H2OII] + k[891]*y[IDX_CH2I] + k[912]*y[IDX_CH3I];
    data[2655] = 0.0 - k[80]*y[IDX_H2OI] + k[492]*y[IDX_CH3OHI];
    data[2656] = 0.0 - k[566]*y[IDX_H2OI];
    data[2657] = 0.0 - k[4]*y[IDX_H2I] - k[11]*y[IDX_HI] - k[80]*y[IDX_HII] -
        k[106]*y[IDX_H2II] - k[123]*y[IDX_COII] - k[124]*y[IDX_HCNII] -
        k[125]*y[IDX_N2II] - k[143]*y[IDX_HeII] - k[160]*y[IDX_NII] -
        k[176]*y[IDX_NHII] - k[210]*y[IDX_OII] - k[220]*y[IDX_OHII] - k[256] -
        k[370]*y[IDX_CII] - k[371]*y[IDX_CII] - k[402]*y[IDX_CHII] -
        k[403]*y[IDX_CHII] - k[404]*y[IDX_CHII] - k[418]*y[IDX_CH2II] -
        k[450]*y[IDX_CH4II] - k[518]*y[IDX_H2II] - k[555]*y[IDX_H2OII] -
        k[560]*y[IDX_CNII] - k[561]*y[IDX_CNII] - k[562]*y[IDX_COII] -
        k[563]*y[IDX_H2COII] - k[564]*y[IDX_H3COII] - k[565]*y[IDX_HCNII] -
        k[566]*y[IDX_HCOII] - k[567]*y[IDX_HCO2II] - k[568]*y[IDX_HNOII] -
        k[569]*y[IDX_N2II] - k[570]*y[IDX_N2HII] - k[571]*y[IDX_O2HII] -
        k[572]*y[IDX_SiII] - k[573]*y[IDX_SiHII] - k[574]*y[IDX_SiH4II] -
        k[575]*y[IDX_SiH5II] - k[586]*y[IDX_H3II] - k[673]*y[IDX_HeII] -
        k[674]*y[IDX_HeII] - k[754]*y[IDX_NHII] - k[755]*y[IDX_NHII] -
        k[756]*y[IDX_NHII] - k[757]*y[IDX_NHII] - k[771]*y[IDX_NH2II] -
        k[772]*y[IDX_NH2II] - k[840]*y[IDX_OHII] - k[904]*y[IDX_CH3I] -
        k[975]*y[IDX_HI] - k[1036]*y[IDX_NHI] - k[1062]*y[IDX_OI] - k[1141] -
        k[1142] - k[1257];
    data[2658] = 0.0 - k[1062]*y[IDX_H2OI];
    data[2659] = 0.0 + k[318]*y[IDX_H3COII] + k[322]*y[IDX_H3OII];
    data[2660] = 0.0 - k[4]*y[IDX_H2OI] + k[966]*y[IDX_OHI];
    data[2661] = 0.0 - k[11]*y[IDX_H2OI] - k[975]*y[IDX_H2OI] + k[990]*y[IDX_O2HI] +
        k[1208]*y[IDX_OHI];
    data[2662] = 0.0 - k[1060]*y[IDX_OI];
    data[2663] = 0.0 - k[1081]*y[IDX_OI];
    data[2664] = 0.0 - k[1082]*y[IDX_OI];
    data[2665] = 0.0 - k[831]*y[IDX_OI];
    data[2666] = 0.0 + k[990]*y[IDX_HI] - k[1077]*y[IDX_OI] + k[1171];
    data[2667] = 0.0 - k[1083]*y[IDX_OI] - k[1084]*y[IDX_OI];
    data[2668] = 0.0 + k[276] + k[1019]*y[IDX_NI] + k[1019]*y[IDX_NI] - k[1075]*y[IDX_OI]
        + k[1164];
    data[2669] = 0.0 - k[834]*y[IDX_OI];
    data[2670] = 0.0 - k[833]*y[IDX_OI];
    data[2671] = 0.0 + k[283] + k[697]*y[IDX_HeII] + k[993]*y[IDX_HI] - k[1078]*y[IDX_OI]
        - k[1079]*y[IDX_OI] + k[1172];
    data[2672] = 0.0 - k[1085]*y[IDX_OI] - k[1086]*y[IDX_OI];
    data[2673] = 0.0 - k[832]*y[IDX_OI];
    data[2674] = 0.0 - k[1088]*y[IDX_OI];
    data[2675] = 0.0 + k[849]*y[IDX_OHII] - k[1089]*y[IDX_OI];
    data[2676] = 0.0 - k[1087]*y[IDX_OI];
    data[2677] = 0.0 + k[362]*y[IDX_eM] - k[835]*y[IDX_OI] + k[1189];
    data[2678] = 0.0 + k[332]*y[IDX_eM] - k[824]*y[IDX_OI];
    data[2679] = 0.0 + k[980]*y[IDX_HI] - k[1068]*y[IDX_OI] - k[1069]*y[IDX_OI] -
        k[1070]*y[IDX_OI];
    data[2680] = 0.0 - k[822]*y[IDX_OI];
    data[2681] = 0.0 + k[293] + k[710]*y[IDX_HeII] + k[850]*y[IDX_OHII] + k[1190];
    data[2682] = 0.0 - k[216]*y[IDX_OI];
    data[2683] = 0.0 - k[826]*y[IDX_OI];
    data[2684] = 0.0 - k[829]*y[IDX_OI];
    data[2685] = 0.0 - k[1212]*y[IDX_OI];
    data[2686] = 0.0 + k[848]*y[IDX_OHII] + k[1106]*y[IDX_O2I] - k[1213]*y[IDX_OI];
    data[2687] = 0.0 + k[844]*y[IDX_OHII];
    data[2688] = 0.0 - k[218]*y[IDX_OI] - k[825]*y[IDX_OI];
    data[2689] = 0.0 + k[207]*y[IDX_OII] - k[1056]*y[IDX_OI];
    data[2690] = 0.0 - k[217]*y[IDX_OI] + k[304]*y[IDX_eM] + k[851]*y[IDX_OHI] + k[1131];
    data[2691] = 0.0 - k[526]*y[IDX_OI];
    data[2692] = 0.0 + k[213]*y[IDX_OII] - k[1074]*y[IDX_OI];
    data[2693] = 0.0 - k[915]*y[IDX_OI] - k[916]*y[IDX_OI] + k[917]*y[IDX_OHI];
    data[2694] = 0.0 + k[252] + k[496]*y[IDX_HII] + k[665]*y[IDX_HeII] +
        k[837]*y[IDX_OHII] - k[1059]*y[IDX_OI] + k[1132];
    data[2695] = 0.0 + k[727]*y[IDX_NOI] + k[728]*y[IDX_O2I];
    data[2696] = 0.0 + k[43]*y[IDX_CH2I] + k[60]*y[IDX_CHI] + k[131]*y[IDX_HI] +
        k[202]*y[IDX_NHI] + k[207]*y[IDX_CH4I] + k[208]*y[IDX_COI] +
        k[209]*y[IDX_H2COI] + k[210]*y[IDX_H2OI] + k[211]*y[IDX_HCOI] +
        k[212]*y[IDX_NH2I] + k[213]*y[IDX_NH3I] + k[214]*y[IDX_O2I] +
        k[215]*y[IDX_OHI] + k[1221]*y[IDX_eM];
    data[2697] = 0.0 + k[777]*y[IDX_O2I] - k[827]*y[IDX_OI];
    data[2698] = 0.0 + k[756]*y[IDX_H2OI] + k[764]*y[IDX_NOI] - k[767]*y[IDX_OI];
    data[2699] = 0.0 + k[346]*y[IDX_eM] + k[346]*y[IDX_eM] + k[391]*y[IDX_CI] +
        k[435]*y[IDX_CH2I] + k[473]*y[IDX_CHI] + k[742]*y[IDX_NI] +
        k[804]*y[IDX_NHI] + k[1167];
    data[2700] = 0.0 + k[442]*y[IDX_O2I] - k[443]*y[IDX_OI] - k[444]*y[IDX_OI];
    data[2701] = 0.0 + k[212]*y[IDX_OII] + k[791]*y[IDX_OHII] + k[1032]*y[IDX_OHI] -
        k[1072]*y[IDX_OI] - k[1073]*y[IDX_OI];
    data[2702] = 0.0 - k[421]*y[IDX_OI];
    data[2703] = 0.0 + k[312]*y[IDX_eM] + k[313]*y[IDX_eM] - k[823]*y[IDX_OI] +
        k[852]*y[IDX_OHI];
    data[2704] = 0.0 - k[828]*y[IDX_OI];
    data[2705] = 0.0 + k[345]*y[IDX_eM];
    data[2706] = 0.0 + k[323]*y[IDX_eM];
    data[2707] = 0.0 + k[845]*y[IDX_OHII] - k[1071]*y[IDX_OI];
    data[2708] = 0.0 + k[376]*y[IDX_O2I] - k[1193]*y[IDX_OI];
    data[2709] = 0.0 + k[841]*y[IDX_OHII] - k[1063]*y[IDX_OI] - k[1064]*y[IDX_OI] -
        k[1065]*y[IDX_OI];
    data[2710] = 0.0 + k[412]*y[IDX_O2I] - k[414]*y[IDX_OI];
    data[2711] = 0.0 + k[43]*y[IDX_OII] + k[435]*y[IDX_O2II] + k[437]*y[IDX_OHII] +
        k[892]*y[IDX_O2I] - k[894]*y[IDX_OI] - k[895]*y[IDX_OI] -
        k[896]*y[IDX_OI] - k[897]*y[IDX_OI] + k[900]*y[IDX_OHI];
    data[2712] = 0.0 + k[306]*y[IDX_eM];
    data[2713] = 0.0 + k[202]*y[IDX_OII] + k[804]*y[IDX_O2II] + k[806]*y[IDX_OHII] +
        k[1042]*y[IDX_NOI] + k[1044]*y[IDX_O2I] - k[1046]*y[IDX_OI] -
        k[1047]*y[IDX_OI] + k[1050]*y[IDX_OHI];
    data[2714] = 0.0 + k[348]*y[IDX_eM] + k[393]*y[IDX_CI] + k[437]*y[IDX_CH2I] +
        k[475]*y[IDX_CHI] + k[791]*y[IDX_NH2I] + k[806]*y[IDX_NHI] -
        k[830]*y[IDX_OI] + k[836]*y[IDX_CNI] + k[837]*y[IDX_CO2I] +
        k[838]*y[IDX_COI] + k[839]*y[IDX_H2COI] + k[840]*y[IDX_H2OI] +
        k[841]*y[IDX_HCNI] + k[843]*y[IDX_HCOI] + k[844]*y[IDX_HNCI] +
        k[845]*y[IDX_N2I] + k[846]*y[IDX_NOI] + k[847]*y[IDX_OHI] +
        k[848]*y[IDX_SiI] + k[849]*y[IDX_SiHI] + k[850]*y[IDX_SiOI];
    data[2715] = 0.0 + k[836]*y[IDX_OHII] + k[949]*y[IDX_O2I] - k[1057]*y[IDX_OI] -
        k[1058]*y[IDX_OI] + k[1090]*y[IDX_OHI];
    data[2716] = 0.0 + k[209]*y[IDX_OII] + k[672]*y[IDX_HeII] + k[839]*y[IDX_OHII] -
        k[1061]*y[IDX_OI];
    data[2717] = 0.0 + k[211]*y[IDX_OII] + k[682]*y[IDX_HeII] + k[843]*y[IDX_OHII] +
        k[978]*y[IDX_HI] + k[1015]*y[IDX_NI] - k[1066]*y[IDX_OI] -
        k[1067]*y[IDX_OI];
    data[2718] = 0.0 + k[665]*y[IDX_CO2I] + k[669]*y[IDX_COI] + k[672]*y[IDX_H2COI] +
        k[682]*y[IDX_HCOI] + k[695]*y[IDX_NOI] + k[696]*y[IDX_O2I] +
        k[697]*y[IDX_OCNI] + k[710]*y[IDX_SiOI];
    data[2719] = 0.0 - k[0]*y[IDX_OI] + k[60]*y[IDX_OII] + k[473]*y[IDX_O2II] +
        k[475]*y[IDX_OHII] + k[930]*y[IDX_NOI] + k[934]*y[IDX_O2I] +
        k[936]*y[IDX_O2I] - k[939]*y[IDX_OI] - k[940]*y[IDX_OI];
    data[2720] = 0.0 - k[598]*y[IDX_OI] - k[599]*y[IDX_OI];
    data[2721] = 0.0 + k[278] + k[695]*y[IDX_HeII] + k[727]*y[IDX_NII] +
        k[764]*y[IDX_NHII] + k[846]*y[IDX_OHII] + k[871]*y[IDX_CI] +
        k[930]*y[IDX_CHI] + k[987]*y[IDX_HI] + k[1022]*y[IDX_NI] +
        k[1042]*y[IDX_NHI] + k[1052]*y[IDX_O2I] - k[1076]*y[IDX_OI] + k[1166];
    data[2722] = 0.0 + k[742]*y[IDX_O2II] + k[1015]*y[IDX_HCOI] + k[1019]*y[IDX_NO2I] +
        k[1019]*y[IDX_NO2I] + k[1022]*y[IDX_NOI] + k[1023]*y[IDX_O2I] +
        k[1026]*y[IDX_OHI];
    data[2723] = 0.0 + k[7]*y[IDX_H2I] + k[13]*y[IDX_HI] + k[215]*y[IDX_OII] + k[284] +
        k[847]*y[IDX_OHII] + k[851]*y[IDX_COII] + k[852]*y[IDX_H2OII] +
        k[876]*y[IDX_CI] + k[900]*y[IDX_CH2I] + k[917]*y[IDX_CH3I] +
        k[996]*y[IDX_HI] + k[1026]*y[IDX_NI] + k[1032]*y[IDX_NH2I] +
        k[1050]*y[IDX_NHI] - k[1080]*y[IDX_OI] + k[1090]*y[IDX_CNI] +
        k[1101]*y[IDX_OHI] + k[1101]*y[IDX_OHI] + k[1174];
    data[2724] = 0.0 + k[6]*y[IDX_H2I] + k[6]*y[IDX_H2I] + k[12]*y[IDX_HI] +
        k[12]*y[IDX_HI] + k[214]*y[IDX_OII] + k[280] + k[280] +
        k[376]*y[IDX_CII] + k[412]*y[IDX_CHII] + k[442]*y[IDX_CH3II] +
        k[696]*y[IDX_HeII] + k[728]*y[IDX_NII] + k[777]*y[IDX_NH2II] +
        k[873]*y[IDX_CI] + k[892]*y[IDX_CH2I] + k[934]*y[IDX_CHI] +
        k[936]*y[IDX_CHI] + k[949]*y[IDX_CNI] + k[953]*y[IDX_COI] +
        k[989]*y[IDX_HI] + k[1023]*y[IDX_NI] + k[1044]*y[IDX_NHI] +
        k[1052]*y[IDX_NOI] + k[1106]*y[IDX_SiI] + k[1169] + k[1169];
    data[2725] = 0.0 + k[391]*y[IDX_O2II] + k[393]*y[IDX_OHII] + k[871]*y[IDX_NOI] +
        k[873]*y[IDX_O2I] + k[876]*y[IDX_OHI] - k[1196]*y[IDX_OI];
    data[2726] = 0.0 - k[89]*y[IDX_OI] + k[496]*y[IDX_CO2I];
    data[2727] = 0.0 + k[210]*y[IDX_OII] + k[756]*y[IDX_NHII] + k[840]*y[IDX_OHII] -
        k[1062]*y[IDX_OI];
    data[2728] = 0.0 - k[0]*y[IDX_CHI] - k[89]*y[IDX_HII] - k[216]*y[IDX_CNII] -
        k[217]*y[IDX_COII] - k[218]*y[IDX_N2II] - k[239] - k[282] -
        k[414]*y[IDX_CHII] - k[421]*y[IDX_CH2II] - k[443]*y[IDX_CH3II] -
        k[444]*y[IDX_CH3II] - k[526]*y[IDX_H2II] - k[598]*y[IDX_H3II] -
        k[599]*y[IDX_H3II] - k[767]*y[IDX_NHII] - k[822]*y[IDX_CH4II] -
        k[823]*y[IDX_H2OII] - k[824]*y[IDX_HCO2II] - k[825]*y[IDX_N2II] -
        k[826]*y[IDX_N2HII] - k[827]*y[IDX_NH2II] - k[828]*y[IDX_NH3II] -
        k[829]*y[IDX_O2HII] - k[830]*y[IDX_OHII] - k[831]*y[IDX_SiCII] -
        k[832]*y[IDX_SiHII] - k[833]*y[IDX_SiH2II] - k[834]*y[IDX_SiH3II] -
        k[835]*y[IDX_SiOII] - k[894]*y[IDX_CH2I] - k[895]*y[IDX_CH2I] -
        k[896]*y[IDX_CH2I] - k[897]*y[IDX_CH2I] - k[915]*y[IDX_CH3I] -
        k[916]*y[IDX_CH3I] - k[939]*y[IDX_CHI] - k[940]*y[IDX_CHI] -
        k[965]*y[IDX_H2I] - k[1046]*y[IDX_NHI] - k[1047]*y[IDX_NHI] -
        k[1056]*y[IDX_CH4I] - k[1057]*y[IDX_CNI] - k[1058]*y[IDX_CNI] -
        k[1059]*y[IDX_CO2I] - k[1060]*y[IDX_H2CNI] - k[1061]*y[IDX_H2COI] -
        k[1062]*y[IDX_H2OI] - k[1063]*y[IDX_HCNI] - k[1064]*y[IDX_HCNI] -
        k[1065]*y[IDX_HCNI] - k[1066]*y[IDX_HCOI] - k[1067]*y[IDX_HCOI] -
        k[1068]*y[IDX_HNOI] - k[1069]*y[IDX_HNOI] - k[1070]*y[IDX_HNOI] -
        k[1071]*y[IDX_N2I] - k[1072]*y[IDX_NH2I] - k[1073]*y[IDX_NH2I] -
        k[1074]*y[IDX_NH3I] - k[1075]*y[IDX_NO2I] - k[1076]*y[IDX_NOI] -
        k[1077]*y[IDX_O2HI] - k[1078]*y[IDX_OCNI] - k[1079]*y[IDX_OCNI] -
        k[1080]*y[IDX_OHI] - k[1081]*y[IDX_SiC2I] - k[1082]*y[IDX_SiC3I] -
        k[1083]*y[IDX_SiCI] - k[1084]*y[IDX_SiCI] - k[1085]*y[IDX_SiH2I] -
        k[1086]*y[IDX_SiH2I] - k[1087]*y[IDX_SiH3I] - k[1088]*y[IDX_SiH4I] -
        k[1089]*y[IDX_SiHI] - k[1193]*y[IDX_CII] - k[1196]*y[IDX_CI] -
        k[1207]*y[IDX_HI] - k[1211]*y[IDX_OI] - k[1211]*y[IDX_OI] -
        k[1211]*y[IDX_OI] - k[1211]*y[IDX_OI] - k[1212]*y[IDX_SiII] -
        k[1213]*y[IDX_SiI] - k[1291];
    data[2729] = 0.0 + k[304]*y[IDX_COII] + k[306]*y[IDX_H2COII] + k[312]*y[IDX_H2OII] +
        k[313]*y[IDX_H2OII] + k[323]*y[IDX_H3OII] + k[332]*y[IDX_HCO2II] +
        k[345]*y[IDX_NOII] + k[346]*y[IDX_O2II] + k[346]*y[IDX_O2II] +
        k[348]*y[IDX_OHII] + k[362]*y[IDX_SiOII] + k[1221]*y[IDX_OII];
    data[2730] = 0.0 + k[208]*y[IDX_OII] + k[253] + k[669]*y[IDX_HeII] +
        k[838]*y[IDX_OHII] + k[953]*y[IDX_O2I] + k[1133];
    data[2731] = 0.0 + k[6]*y[IDX_O2I] + k[6]*y[IDX_O2I] + k[7]*y[IDX_OHI] -
        k[965]*y[IDX_OI];
    data[2732] = 0.0 + k[12]*y[IDX_O2I] + k[12]*y[IDX_O2I] + k[13]*y[IDX_OHI] +
        k[131]*y[IDX_OII] + k[978]*y[IDX_HCOI] + k[980]*y[IDX_HNOI] +
        k[987]*y[IDX_NOI] + k[989]*y[IDX_O2I] + k[990]*y[IDX_O2HI] +
        k[993]*y[IDX_OCNI] + k[996]*y[IDX_OHI] - k[1207]*y[IDX_OI];
    data[2733] = 0.0 - k[351]*y[IDX_eM];
    data[2734] = 0.0 - k[310]*y[IDX_eM] - k[311]*y[IDX_eM];
    data[2735] = 0.0 - k[335]*y[IDX_eM];
    data[2736] = 0.0 - k[336]*y[IDX_eM];
    data[2737] = 0.0 - k[350]*y[IDX_eM];
    data[2738] = 0.0 - k[360]*y[IDX_eM] - k[361]*y[IDX_eM];
    data[2739] = 0.0 - k[358]*y[IDX_eM] - k[359]*y[IDX_eM];
    data[2740] = 0.0 - k[349]*y[IDX_eM];
    data[2741] = 0.0 - k[356]*y[IDX_eM] - k[357]*y[IDX_eM];
    data[2742] = 0.0 - k[353]*y[IDX_eM] - k[354]*y[IDX_eM] - k[355]*y[IDX_eM];
    data[2743] = 0.0 + k[1180];
    data[2744] = 0.0 - k[363]*y[IDX_eM] - k[364]*y[IDX_eM];
    data[2745] = 0.0 - k[352]*y[IDX_eM];
    data[2746] = 0.0 + k[1183];
    data[2747] = 0.0 - k[362]*y[IDX_eM];
    data[2748] = 0.0 - k[331]*y[IDX_eM] - k[332]*y[IDX_eM] - k[333]*y[IDX_eM];
    data[2749] = 0.0 + k[1120];
    data[2750] = 0.0 + k[266] + k[1154];
    data[2751] = 0.0 - k[1219]*y[IDX_eM];
    data[2752] = 0.0 - k[301]*y[IDX_eM] - k[302]*y[IDX_eM];
    data[2753] = 0.0 + k[1191];
    data[2754] = 0.0 - k[303]*y[IDX_eM];
    data[2755] = 0.0 - k[327]*y[IDX_eM] - k[328]*y[IDX_eM] - k[329]*y[IDX_eM];
    data[2756] = 0.0 - k[338]*y[IDX_eM] - k[339]*y[IDX_eM];
    data[2757] = 0.0 - k[347]*y[IDX_eM];
    data[2758] = 0.0 - k[1222]*y[IDX_eM];
    data[2759] = 0.0 + k[285] + k[1176];
    data[2760] = 0.0 - k[334]*y[IDX_eM];
    data[2761] = 0.0 - k[337]*y[IDX_eM];
    data[2762] = 0.0 - k[317]*y[IDX_eM] - k[318]*y[IDX_eM] - k[319]*y[IDX_eM] -
        k[320]*y[IDX_eM] - k[321]*y[IDX_eM];
    data[2763] = 0.0 + k[1126];
    data[2764] = 0.0 - k[304]*y[IDX_eM];
    data[2765] = 0.0 - k[305]*y[IDX_eM];
    data[2766] = 0.0 + k[272] + k[1160];
    data[2767] = 0.0 + k[245] + k[1117];
    data[2768] = 0.0 - k[1220]*y[IDX_eM];
    data[2769] = 0.0 - k[1221]*y[IDX_eM];
    data[2770] = 0.0 - k[326]*y[IDX_eM];
    data[2771] = 0.0 - k[341]*y[IDX_eM] - k[342]*y[IDX_eM];
    data[2772] = 0.0 - k[340]*y[IDX_eM];
    data[2773] = 0.0 - k[346]*y[IDX_eM];
    data[2774] = 0.0 - k[298]*y[IDX_eM] - k[299]*y[IDX_eM] - k[300]*y[IDX_eM] -
        k[1215]*y[IDX_eM];
    data[2775] = 0.0 + k[269] + k[1157];
    data[2776] = 0.0 - k[295]*y[IDX_eM] - k[296]*y[IDX_eM] - k[297]*y[IDX_eM];
    data[2777] = 0.0 - k[312]*y[IDX_eM] - k[313]*y[IDX_eM] - k[314]*y[IDX_eM];
    data[2778] = 0.0 - k[343]*y[IDX_eM] - k[344]*y[IDX_eM];
    data[2779] = 0.0 - k[345]*y[IDX_eM];
    data[2780] = 0.0 - k[322]*y[IDX_eM] - k[323]*y[IDX_eM] - k[324]*y[IDX_eM] -
        k[325]*y[IDX_eM];
    data[2781] = 0.0 - k[1214]*y[IDX_eM];
    data[2782] = 0.0 - k[294]*y[IDX_eM];
    data[2783] = 0.0 + k[242] + k[1112];
    data[2784] = 0.0 - k[306]*y[IDX_eM] - k[307]*y[IDX_eM] - k[308]*y[IDX_eM] -
        k[309]*y[IDX_eM] - k[1217]*y[IDX_eM];
    data[2785] = 0.0 + k[275] + k[1163];
    data[2786] = 0.0 - k[348]*y[IDX_eM];
    data[2787] = 0.0 + k[1138] + k[1139];
    data[2788] = 0.0 + k[261] + k[1150];
    data[2789] = 0.0 - k[1218]*y[IDX_eM];
    data[2790] = 0.0 + k[0]*y[IDX_OI] + k[1129];
    data[2791] = 0.0 - k[315]*y[IDX_eM] - k[316]*y[IDX_eM];
    data[2792] = 0.0 + k[237] + k[265];
    data[2793] = 0.0 + k[277] + k[1165];
    data[2794] = 0.0 + k[238] + k[268];
    data[2795] = 0.0 + k[1175];
    data[2796] = 0.0 + k[279] + k[1168];
    data[2797] = 0.0 + k[231] + k[240] + k[1107];
    data[2798] = 0.0 - k[1216]*y[IDX_eM];
    data[2799] = 0.0 - k[330]*y[IDX_eM];
    data[2800] = 0.0 + k[1141];
    data[2801] = 0.0 + k[0]*y[IDX_CHI] + k[239] + k[282];
    data[2802] = 0.0 - k[8]*y[IDX_H2I] + k[8]*y[IDX_H2I] - k[294]*y[IDX_CHII] -
        k[295]*y[IDX_CH2II] - k[296]*y[IDX_CH2II] - k[297]*y[IDX_CH2II] -
        k[298]*y[IDX_CH3II] - k[299]*y[IDX_CH3II] - k[300]*y[IDX_CH3II] -
        k[301]*y[IDX_CH4II] - k[302]*y[IDX_CH4II] - k[303]*y[IDX_CNII] -
        k[304]*y[IDX_COII] - k[305]*y[IDX_H2II] - k[306]*y[IDX_H2COII] -
        k[307]*y[IDX_H2COII] - k[308]*y[IDX_H2COII] - k[309]*y[IDX_H2COII] -
        k[310]*y[IDX_H2NOII] - k[311]*y[IDX_H2NOII] - k[312]*y[IDX_H2OII] -
        k[313]*y[IDX_H2OII] - k[314]*y[IDX_H2OII] - k[315]*y[IDX_H3II] -
        k[316]*y[IDX_H3II] - k[317]*y[IDX_H3COII] - k[318]*y[IDX_H3COII] -
        k[319]*y[IDX_H3COII] - k[320]*y[IDX_H3COII] - k[321]*y[IDX_H3COII] -
        k[322]*y[IDX_H3OII] - k[323]*y[IDX_H3OII] - k[324]*y[IDX_H3OII] -
        k[325]*y[IDX_H3OII] - k[326]*y[IDX_HCNII] - k[327]*y[IDX_HCNHII] -
        k[328]*y[IDX_HCNHII] - k[329]*y[IDX_HCNHII] - k[330]*y[IDX_HCOII] -
        k[331]*y[IDX_HCO2II] - k[332]*y[IDX_HCO2II] - k[333]*y[IDX_HCO2II] -
        k[334]*y[IDX_HNOII] - k[335]*y[IDX_HOCII] - k[336]*y[IDX_HeHII] -
        k[337]*y[IDX_N2II] - k[338]*y[IDX_N2HII] - k[339]*y[IDX_N2HII] -
        k[340]*y[IDX_NHII] - k[341]*y[IDX_NH2II] - k[342]*y[IDX_NH2II] -
        k[343]*y[IDX_NH3II] - k[344]*y[IDX_NH3II] - k[345]*y[IDX_NOII] -
        k[346]*y[IDX_O2II] - k[347]*y[IDX_O2HII] - k[348]*y[IDX_OHII] -
        k[349]*y[IDX_SiCII] - k[350]*y[IDX_SiC2II] - k[351]*y[IDX_SiC3II] -
        k[352]*y[IDX_SiHII] - k[353]*y[IDX_SiH2II] - k[354]*y[IDX_SiH2II] -
        k[355]*y[IDX_SiH2II] - k[356]*y[IDX_SiH3II] - k[357]*y[IDX_SiH3II] -
        k[358]*y[IDX_SiH4II] - k[359]*y[IDX_SiH4II] - k[360]*y[IDX_SiH5II] -
        k[361]*y[IDX_SiH5II] - k[362]*y[IDX_SiOII] - k[363]*y[IDX_SiOHII] -
        k[364]*y[IDX_SiOHII] - k[1214]*y[IDX_CII] - k[1215]*y[IDX_CH3II] -
        k[1216]*y[IDX_HII] - k[1217]*y[IDX_H2COII] - k[1218]*y[IDX_HeII] -
        k[1219]*y[IDX_MgII] - k[1220]*y[IDX_NII] - k[1221]*y[IDX_OII] -
        k[1222]*y[IDX_SiII] - k[1306];
    data[2803] = 0.0 + k[232];
    data[2804] = 0.0 - k[8]*y[IDX_eM] + k[8]*y[IDX_eM] + k[233] + k[234];
    data[2805] = 0.0 + k[236] + k[258];
    data[2806] = 0.0 + k[1331] + k[1332] + k[1333] + k[1334];
    data[2807] = 0.0 + k[263] + k[502]*y[IDX_HII] + k[1004]*y[IDX_CI] + k[1152];
    data[2808] = 0.0 + k[335]*y[IDX_eM];
    data[2809] = 0.0 + k[1081]*y[IDX_OI];
    data[2810] = 0.0 + k[1082]*y[IDX_OI];
    data[2811] = 0.0 - k[489]*y[IDX_COI];
    data[2812] = 0.0 - k[954]*y[IDX_COI];
    data[2813] = 0.0 + k[1083]*y[IDX_OI];
    data[2814] = 0.0 - k[952]*y[IDX_COI];
    data[2815] = 0.0 + k[874]*y[IDX_CI] + k[994]*y[IDX_HI] + k[1055]*y[IDX_O2I] +
        k[1078]*y[IDX_OI];
    data[2816] = 0.0 + k[637]*y[IDX_HCOII];
    data[2817] = 0.0 + k[638]*y[IDX_HCOII];
    data[2818] = 0.0 + k[639]*y[IDX_HCOII];
    data[2819] = 0.0 + k[395]*y[IDX_CI] - k[490]*y[IDX_COI];
    data[2820] = 0.0 + k[332]*y[IDX_eM] + k[333]*y[IDX_eM] - k[485]*y[IDX_COI];
    data[2821] = 0.0 - k[951]*y[IDX_COI];
    data[2822] = 0.0 - k[448]*y[IDX_COI];
    data[2823] = 0.0 + k[382]*y[IDX_CII] + k[640]*y[IDX_HCOII];
    data[2824] = 0.0 - k[63]*y[IDX_COI] + k[480]*y[IDX_HCOI] + k[481]*y[IDX_O2I];
    data[2825] = 0.0 - k[487]*y[IDX_COI];
    data[2826] = 0.0 - k[488]*y[IDX_COI];
    data[2827] = 0.0 + k[861]*y[IDX_HCOII] + k[1103]*y[IDX_CO2I] - k[1104]*y[IDX_COI];
    data[2828] = 0.0 + k[648]*y[IDX_HCOII];
    data[2829] = 0.0 - k[486]*y[IDX_COI];
    data[2830] = 0.0 - k[74]*y[IDX_COI] + k[731]*y[IDX_HCOI];
    data[2831] = 0.0 + k[319]*y[IDX_eM];
    data[2832] = 0.0 + k[52]*y[IDX_COII];
    data[2833] = 0.0 + k[28]*y[IDX_CI] + k[38]*y[IDX_CH2I] + k[52]*y[IDX_CH4I] +
        k[54]*y[IDX_CHI] + k[70]*y[IDX_H2COI] + k[71]*y[IDX_HCOI] +
        k[72]*y[IDX_NOI] + k[73]*y[IDX_O2I] + k[123]*y[IDX_H2OI] +
        k[127]*y[IDX_HI] + k[134]*y[IDX_HCNI] + k[184]*y[IDX_NH2I] +
        k[193]*y[IDX_NH3I] + k[200]*y[IDX_NHI] + k[217]*y[IDX_OI] +
        k[226]*y[IDX_OHI];
    data[2834] = 0.0 - k[104]*y[IDX_COI] - k[515]*y[IDX_COI] + k[519]*y[IDX_HCOI];
    data[2835] = 0.0 + k[193]*y[IDX_COII];
    data[2836] = 0.0 + k[905]*y[IDX_HCOI] + k[915]*y[IDX_OI];
    data[2837] = 0.0 + k[252] + k[367]*y[IDX_CII] + k[398]*y[IDX_CHII] +
        k[416]*y[IDX_CH2II] + k[666]*y[IDX_HeII] + k[749]*y[IDX_NHII] +
        k[812]*y[IDX_OII] + k[923]*y[IDX_CHI] + k[971]*y[IDX_HI] +
        k[1012]*y[IDX_NI] + k[1059]*y[IDX_OI] + k[1103]*y[IDX_SiI] + k[1132];
    data[2838] = 0.0 - k[158]*y[IDX_COI] - k[720]*y[IDX_COI] + k[723]*y[IDX_HCOI];
    data[2839] = 0.0 - k[208]*y[IDX_COI] + k[812]*y[IDX_CO2I] + k[816]*y[IDX_HCOI];
    data[2840] = 0.0 - k[621]*y[IDX_COI] + k[625]*y[IDX_HCOI];
    data[2841] = 0.0 + k[749]*y[IDX_CO2I] - k[751]*y[IDX_COI];
    data[2842] = 0.0 + k[644]*y[IDX_HCOI];
    data[2843] = 0.0 + k[441]*y[IDX_HCOI];
    data[2844] = 0.0 + k[184]*y[IDX_COII] + k[787]*y[IDX_HCOII];
    data[2845] = 0.0 + k[416]*y[IDX_CO2I] + k[419]*y[IDX_HCOI];
    data[2846] = 0.0 - k[553]*y[IDX_COI] + k[557]*y[IDX_HCOI];
    data[2847] = 0.0 + k[367]*y[IDX_CO2I] + k[368]*y[IDX_H2COI] + k[372]*y[IDX_HCOI] +
        k[377]*y[IDX_O2I] + k[382]*y[IDX_SiOI];
    data[2848] = 0.0 + k[134]*y[IDX_COII] + k[629]*y[IDX_HCOII] + k[1064]*y[IDX_OI] +
        k[1095]*y[IDX_OHI];
    data[2849] = 0.0 + k[398]*y[IDX_CO2I] + k[399]*y[IDX_H2COI] + k[406]*y[IDX_HCOI];
    data[2850] = 0.0 + k[38]*y[IDX_COII] + k[429]*y[IDX_HCOII] + k[882]*y[IDX_HCOI] +
        k[891]*y[IDX_O2I] + k[894]*y[IDX_OI] + k[895]*y[IDX_OI];
    data[2851] = 0.0 + k[307]*y[IDX_eM] + k[308]*y[IDX_eM] + k[641]*y[IDX_HCOI];
    data[2852] = 0.0 + k[200]*y[IDX_COII] + k[799]*y[IDX_HCOII];
    data[2853] = 0.0 - k[838]*y[IDX_COI] + k[842]*y[IDX_HCOI];
    data[2854] = 0.0 + k[943]*y[IDX_HCOI] + k[946]*y[IDX_NOI] + k[948]*y[IDX_O2I] +
        k[1057]*y[IDX_OI];
    data[2855] = 0.0 + k[70]*y[IDX_COII] + k[255] + k[368]*y[IDX_CII] +
        k[399]*y[IDX_CHII] + k[635]*y[IDX_HCOII] + k[1136] + k[1137];
    data[2856] = 0.0 + k[71]*y[IDX_COII] + k[260] + k[372]*y[IDX_CII] +
        k[406]*y[IDX_CHII] + k[419]*y[IDX_CH2II] + k[441]*y[IDX_CH3II] +
        k[480]*y[IDX_CNII] + k[501]*y[IDX_HII] + k[519]*y[IDX_H2II] +
        k[557]*y[IDX_H2OII] + k[625]*y[IDX_HCNII] + k[636]*y[IDX_HCOII] +
        k[641]*y[IDX_H2COII] + k[644]*y[IDX_O2II] + k[681]*y[IDX_HeII] +
        k[723]*y[IDX_NII] + k[731]*y[IDX_N2II] + k[816]*y[IDX_OII] +
        k[842]*y[IDX_OHII] + k[864]*y[IDX_CI] + k[882]*y[IDX_CH2I] +
        k[905]*y[IDX_CH3I] + k[925]*y[IDX_CHI] + k[943]*y[IDX_CNI] +
        k[977]*y[IDX_HI] + k[997]*y[IDX_HCOI] + k[997]*y[IDX_HCOI] +
        k[997]*y[IDX_HCOI] + k[997]*y[IDX_HCOI] + k[998]*y[IDX_HCOI] +
        k[998]*y[IDX_HCOI] + k[1000]*y[IDX_NOI] + k[1002]*y[IDX_O2I] +
        k[1014]*y[IDX_NI] + k[1067]*y[IDX_OI] + k[1096]*y[IDX_OHI] + k[1149];
    data[2857] = 0.0 + k[666]*y[IDX_CO2I] - k[669]*y[IDX_COI] + k[681]*y[IDX_HCOI];
    data[2858] = 0.0 + k[54]*y[IDX_COII] + k[466]*y[IDX_HCOII] + k[923]*y[IDX_CO2I] +
        k[925]*y[IDX_HCOI] + k[934]*y[IDX_O2I] + k[935]*y[IDX_O2I] +
        k[939]*y[IDX_OI];
    data[2859] = 0.0 - k[583]*y[IDX_COI] - k[584]*y[IDX_COI];
    data[2860] = 0.0 + k[72]*y[IDX_COII] + k[872]*y[IDX_CI] + k[946]*y[IDX_CNI] +
        k[1000]*y[IDX_HCOI];
    data[2861] = 0.0 + k[1012]*y[IDX_CO2I] + k[1014]*y[IDX_HCOI];
    data[2862] = 0.0 + k[226]*y[IDX_COII] + k[854]*y[IDX_HCOII] + k[875]*y[IDX_CI] -
        k[1092]*y[IDX_COI] + k[1095]*y[IDX_HCNI] + k[1096]*y[IDX_HCOI];
    data[2863] = 0.0 + k[73]*y[IDX_COII] + k[377]*y[IDX_CII] + k[481]*y[IDX_CNII] +
        k[873]*y[IDX_CI] + k[891]*y[IDX_CH2I] + k[934]*y[IDX_CHI] +
        k[935]*y[IDX_CHI] + k[948]*y[IDX_CNI] - k[953]*y[IDX_COI] +
        k[1002]*y[IDX_HCOI] + k[1055]*y[IDX_OCNI];
    data[2864] = 0.0 + k[28]*y[IDX_COII] + k[386]*y[IDX_HCOII] + k[395]*y[IDX_SiOII] +
        k[864]*y[IDX_HCOI] + k[872]*y[IDX_NOI] + k[873]*y[IDX_O2I] +
        k[874]*y[IDX_OCNI] + k[875]*y[IDX_OHI] + k[1004]*y[IDX_HNCOI] +
        k[1196]*y[IDX_OI];
    data[2865] = 0.0 + k[501]*y[IDX_HCOI] + k[502]*y[IDX_HNCOI];
    data[2866] = 0.0 + k[330]*y[IDX_eM] + k[386]*y[IDX_CI] + k[429]*y[IDX_CH2I] +
        k[466]*y[IDX_CHI] + k[566]*y[IDX_H2OI] + k[629]*y[IDX_HCNI] +
        k[635]*y[IDX_H2COI] + k[636]*y[IDX_HCOI] + k[637]*y[IDX_SiH2I] +
        k[638]*y[IDX_SiH4I] + k[639]*y[IDX_SiHI] + k[640]*y[IDX_SiOI] +
        k[648]*y[IDX_HNCI] + k[787]*y[IDX_NH2I] + k[799]*y[IDX_NHI] +
        k[854]*y[IDX_OHI] + k[861]*y[IDX_SiI];
    data[2867] = 0.0 + k[123]*y[IDX_COII] + k[566]*y[IDX_HCOII];
    data[2868] = 0.0 + k[217]*y[IDX_COII] + k[894]*y[IDX_CH2I] + k[895]*y[IDX_CH2I] +
        k[915]*y[IDX_CH3I] + k[939]*y[IDX_CHI] + k[1057]*y[IDX_CNI] +
        k[1059]*y[IDX_CO2I] + k[1064]*y[IDX_HCNI] + k[1067]*y[IDX_HCOI] +
        k[1078]*y[IDX_OCNI] + k[1081]*y[IDX_SiC2I] + k[1082]*y[IDX_SiC3I] +
        k[1083]*y[IDX_SiCI] + k[1196]*y[IDX_CI];
    data[2869] = 0.0 + k[307]*y[IDX_H2COII] + k[308]*y[IDX_H2COII] + k[319]*y[IDX_H3COII]
        + k[330]*y[IDX_HCOII] + k[332]*y[IDX_HCO2II] + k[333]*y[IDX_HCO2II] +
        k[335]*y[IDX_HOCII];
    data[2870] = 0.0 - k[63]*y[IDX_CNII] - k[74]*y[IDX_N2II] - k[104]*y[IDX_H2II] -
        k[158]*y[IDX_NII] - k[208]*y[IDX_OII] - k[232] - k[253] -
        k[448]*y[IDX_CH4II] - k[485]*y[IDX_HCO2II] - k[486]*y[IDX_HNOII] -
        k[487]*y[IDX_N2HII] - k[488]*y[IDX_O2HII] - k[489]*y[IDX_SiH4II] -
        k[490]*y[IDX_SiOII] - k[515]*y[IDX_H2II] - k[553]*y[IDX_H2OII] -
        k[583]*y[IDX_H3II] - k[584]*y[IDX_H3II] - k[621]*y[IDX_HCNII] -
        k[669]*y[IDX_HeII] - k[720]*y[IDX_NII] - k[751]*y[IDX_NHII] -
        k[838]*y[IDX_OHII] - k[951]*y[IDX_HNOI] - k[952]*y[IDX_NO2I] -
        k[953]*y[IDX_O2I] - k[954]*y[IDX_O2HI] - k[972]*y[IDX_HI] -
        k[1092]*y[IDX_OHI] - k[1104]*y[IDX_SiI] - k[1133] - k[1227] - k[1251] -
        k[1304];
    data[2871] = 0.0 + k[127]*y[IDX_COII] + k[971]*y[IDX_CO2I] - k[972]*y[IDX_COI] +
        k[977]*y[IDX_HCOI] + k[994]*y[IDX_OCNI];
    data[2872] = 0.0 + k[973]*y[IDX_HI] + k[1060]*y[IDX_OI];
    data[2873] = 0.0 + k[311]*y[IDX_eM];
    data[2874] = 0.0 + k[257] + k[499]*y[IDX_HII] + k[1143];
    data[2875] = 0.0 - k[5]*y[IDX_H2I] + k[5]*y[IDX_H2I];
    data[2876] = 0.0 - k[537]*y[IDX_H2I];
    data[2877] = 0.0 + k[360]*y[IDX_eM];
    data[2878] = 0.0 + k[358]*y[IDX_eM] - k[546]*y[IDX_H2I];
    data[2879] = 0.0 + k[991]*y[IDX_HI];
    data[2880] = 0.0 + k[595]*y[IDX_H3II];
    data[2881] = 0.0 + k[357]*y[IDX_eM] + k[834]*y[IDX_OI] - k[1204]*y[IDX_H2I];
    data[2882] = 0.0 + k[353]*y[IDX_eM];
    data[2883] = 0.0 + k[380]*y[IDX_CII] + k[505]*y[IDX_HII] + k[602]*y[IDX_H3II] +
        k[703]*y[IDX_HeII] + k[1085]*y[IDX_OI];
    data[2884] = 0.0 + k[619]*y[IDX_HI] - k[1203]*y[IDX_H2I];
    data[2885] = 0.0 + k[291] + k[507]*y[IDX_HII] + k[604]*y[IDX_H3II] +
        k[707]*y[IDX_HeII] + k[707]*y[IDX_HeII] + k[708]*y[IDX_HeII] + k[1185] +
        k[1187];
    data[2886] = 0.0 + k[508]*y[IDX_HII] + k[605]*y[IDX_H3II];
    data[2887] = 0.0 + k[506]*y[IDX_HII] + k[603]*y[IDX_H3II] + k[705]*y[IDX_HeII] +
        k[1184];
    data[2888] = 0.0 - k[547]*y[IDX_H2I];
    data[2889] = 0.0 + k[503]*y[IDX_HII] + k[590]*y[IDX_H3II] + k[981]*y[IDX_HI];
    data[2890] = 0.0 + k[247] + k[493]*y[IDX_HII] + k[494]*y[IDX_HII] + k[494]*y[IDX_HII]
        + k[579]*y[IDX_H3II] + k[1119];
    data[2891] = 0.0 + k[591]*y[IDX_H3II];
    data[2892] = 0.0 + k[617]*y[IDX_HI] + k[1122];
    data[2893] = 0.0 + k[606]*y[IDX_H3II];
    data[2894] = 0.0 - k[531]*y[IDX_H2I];
    data[2895] = 0.0 - k[544]*y[IDX_H2I];
    data[2896] = 0.0 - k[1202]*y[IDX_H2I];
    data[2897] = 0.0 + k[601]*y[IDX_H3II];
    data[2898] = 0.0 + k[589]*y[IDX_H3II];
    data[2899] = 0.0 + k[455]*y[IDX_CH4I] - k[539]*y[IDX_H2I];
    data[2900] = 0.0 + k[319]*y[IDX_eM];
    data[2901] = 0.0 + k[101]*y[IDX_H2II] + k[249] + k[455]*y[IDX_N2II] +
        k[495]*y[IDX_HII] + k[511]*y[IDX_H2II] + k[658]*y[IDX_HeII] +
        k[659]*y[IDX_HeII] + k[717]*y[IDX_NII] + k[969]*y[IDX_HI] + k[1124] +
        k[1127];
    data[2902] = 0.0 - k[532]*y[IDX_H2I] - k[533]*y[IDX_H2I];
    data[2903] = 0.0 + k[100]*y[IDX_CH2I] + k[101]*y[IDX_CH4I] + k[102]*y[IDX_CHI] +
        k[103]*y[IDX_CNI] + k[104]*y[IDX_COI] + k[105]*y[IDX_H2COI] +
        k[106]*y[IDX_H2OI] + k[107]*y[IDX_HCNI] + k[108]*y[IDX_HCOI] +
        k[109]*y[IDX_NH2I] + k[110]*y[IDX_NH3I] + k[111]*y[IDX_NHI] +
        k[112]*y[IDX_NOI] + k[113]*y[IDX_O2I] + k[114]*y[IDX_OHI] +
        k[128]*y[IDX_HI] + k[511]*y[IDX_CH4I] - k[516]*y[IDX_H2I] +
        k[517]*y[IDX_H2COI];
    data[2904] = 0.0 + k[110]*y[IDX_H2II] + k[273] + k[374]*y[IDX_CII] +
        k[691]*y[IDX_HeII] + k[724]*y[IDX_NII] + k[984]*y[IDX_HI] + k[1161];
    data[2905] = 0.0 + k[246] + k[578]*y[IDX_H3II] + k[655]*y[IDX_HeII] +
        k[915]*y[IDX_OI] + k[918]*y[IDX_OHI] - k[957]*y[IDX_H2I] +
        k[968]*y[IDX_HI] + k[1009]*y[IDX_NI] + k[1118];
    data[2906] = 0.0 + k[582]*y[IDX_H3II];
    data[2907] = 0.0 - k[538]*y[IDX_H2I] + k[717]*y[IDX_CH4I] + k[724]*y[IDX_NH3I];
    data[2908] = 0.0 - k[543]*y[IDX_H2I];
    data[2909] = 0.0 - k[535]*y[IDX_H2I];
    data[2910] = 0.0 - k[542]*y[IDX_H2I];
    data[2911] = 0.0 - k[540]*y[IDX_H2I] - k[541]*y[IDX_H2I] + k[755]*y[IDX_H2OI];
    data[2912] = 0.0 + k[299]*y[IDX_eM] + k[444]*y[IDX_OI] + k[445]*y[IDX_OHI] +
        k[616]*y[IDX_HI] + k[794]*y[IDX_NHI] + k[1114];
    data[2913] = 0.0 + k[109]*y[IDX_H2II] + k[409]*y[IDX_CHII] + k[593]*y[IDX_H3II] +
        k[689]*y[IDX_HeII] - k[961]*y[IDX_H2I] + k[983]*y[IDX_HI];
    data[2914] = 0.0 + k[295]*y[IDX_eM] - k[530]*y[IDX_H2I] + k[615]*y[IDX_HI] + k[1109];
    data[2915] = 0.0 + k[312]*y[IDX_eM] - k[534]*y[IDX_H2I] + k[739]*y[IDX_NI] +
        k[823]*y[IDX_OI];
    data[2916] = 0.0 + k[828]*y[IDX_OI];
    data[2917] = 0.0 + k[323]*y[IDX_eM] + k[324]*y[IDX_eM] + k[384]*y[IDX_CI];
    data[2918] = 0.0 + k[592]*y[IDX_H3II];
    data[2919] = 0.0 + k[374]*y[IDX_NH3I] + k[380]*y[IDX_SiH2I] - k[528]*y[IDX_H2I] -
        k[1199]*y[IDX_H2I];
    data[2920] = 0.0 + k[107]*y[IDX_H2II] + k[587]*y[IDX_H3II] + k[976]*y[IDX_HI];
    data[2921] = 0.0 + k[404]*y[IDX_H2OI] + k[409]*y[IDX_NH2I] + k[410]*y[IDX_NHI] +
        k[415]*y[IDX_OHI] - k[529]*y[IDX_H2I] + k[614]*y[IDX_HI];
    data[2922] = 0.0 + k[100]*y[IDX_H2II] + k[491]*y[IDX_HII] + k[577]*y[IDX_H3II] +
        k[653]*y[IDX_HeII] + k[889]*y[IDX_O2I] + k[894]*y[IDX_OI] -
        k[956]*y[IDX_H2I] + k[967]*y[IDX_HI];
    data[2923] = 0.0 + k[307]*y[IDX_eM];
    data[2924] = 0.0 + k[111]*y[IDX_H2II] + k[410]*y[IDX_CHII] + k[594]*y[IDX_H3II] +
        k[794]*y[IDX_CH3II] - k[962]*y[IDX_H2I] + k[985]*y[IDX_HI] +
        k[1038]*y[IDX_NHI] + k[1038]*y[IDX_NHI];
    data[2925] = 0.0 - k[545]*y[IDX_H2I];
    data[2926] = 0.0 + k[103]*y[IDX_H2II] + k[581]*y[IDX_H3II] - k[959]*y[IDX_H2I];
    data[2927] = 0.0 + k[105]*y[IDX_H2II] + k[255] + k[497]*y[IDX_HII] +
        k[498]*y[IDX_HII] + k[517]*y[IDX_H2II] + k[585]*y[IDX_H3II] +
        k[670]*y[IDX_HeII] + k[974]*y[IDX_HI] + k[1136];
    data[2928] = 0.0 + k[108]*y[IDX_H2II] + k[500]*y[IDX_HII] + k[588]*y[IDX_H3II] +
        k[977]*y[IDX_HI] + k[997]*y[IDX_HCOI] + k[997]*y[IDX_HCOI];
    data[2929] = 0.0 - k[115]*y[IDX_H2I] - k[536]*y[IDX_H2I] + k[653]*y[IDX_CH2I] +
        k[655]*y[IDX_CH3I] + k[658]*y[IDX_CH4I] + k[659]*y[IDX_CH4I] +
        k[670]*y[IDX_H2COI] + k[689]*y[IDX_NH2I] + k[691]*y[IDX_NH3I] +
        k[703]*y[IDX_SiH2I] + k[705]*y[IDX_SiH3I] + k[707]*y[IDX_SiH4I] +
        k[707]*y[IDX_SiH4I] + k[708]*y[IDX_SiH4I];
    data[2930] = 0.0 - k[2]*y[IDX_H2I] + k[2]*y[IDX_H2I] + k[102]*y[IDX_H2II] +
        k[580]*y[IDX_H3II] - k[958]*y[IDX_H2I] + k[970]*y[IDX_HI] -
        k[1201]*y[IDX_H2I];
    data[2931] = 0.0 + k[315]*y[IDX_eM] + k[576]*y[IDX_CI] + k[577]*y[IDX_CH2I] +
        k[578]*y[IDX_CH3I] + k[579]*y[IDX_CH3OHI] + k[580]*y[IDX_CHI] +
        k[581]*y[IDX_CNI] + k[582]*y[IDX_CO2I] + k[583]*y[IDX_COI] +
        k[584]*y[IDX_COI] + k[585]*y[IDX_H2COI] + k[586]*y[IDX_H2OI] +
        k[587]*y[IDX_HCNI] + k[588]*y[IDX_HCOI] + k[589]*y[IDX_HNCI] +
        k[590]*y[IDX_HNOI] + k[591]*y[IDX_MgI] + k[592]*y[IDX_N2I] +
        k[593]*y[IDX_NH2I] + k[594]*y[IDX_NHI] + k[595]*y[IDX_NO2I] +
        k[596]*y[IDX_NOI] + k[597]*y[IDX_O2I] + k[599]*y[IDX_OI] +
        k[600]*y[IDX_OHI] + k[601]*y[IDX_SiI] + k[602]*y[IDX_SiH2I] +
        k[603]*y[IDX_SiH3I] + k[604]*y[IDX_SiH4I] + k[605]*y[IDX_SiHI] +
        k[606]*y[IDX_SiOI] + k[1146];
    data[2932] = 0.0 + k[112]*y[IDX_H2II] + k[596]*y[IDX_H3II];
    data[2933] = 0.0 + k[739]*y[IDX_H2OII] - k[960]*y[IDX_H2I] + k[1009]*y[IDX_CH3I];
    data[2934] = 0.0 - k[7]*y[IDX_H2I] + k[7]*y[IDX_H2I] + k[114]*y[IDX_H2II] +
        k[415]*y[IDX_CHII] + k[445]*y[IDX_CH3II] + k[600]*y[IDX_H3II] +
        k[918]*y[IDX_CH3I] - k[966]*y[IDX_H2I] + k[996]*y[IDX_HI];
    data[2935] = 0.0 - k[6]*y[IDX_H2I] + k[6]*y[IDX_H2I] + k[113]*y[IDX_H2II] +
        k[597]*y[IDX_H3II] + k[889]*y[IDX_CH2I] - k[963]*y[IDX_H2I] -
        k[964]*y[IDX_H2I];
    data[2936] = 0.0 + k[384]*y[IDX_H3OII] + k[576]*y[IDX_H3II] - k[955]*y[IDX_H2I] -
        k[1200]*y[IDX_H2I];
    data[2937] = 0.0 + k[491]*y[IDX_CH2I] + k[493]*y[IDX_CH3OHI] + k[494]*y[IDX_CH3OHI] +
        k[494]*y[IDX_CH3OHI] + k[495]*y[IDX_CH4I] + k[497]*y[IDX_H2COI] +
        k[498]*y[IDX_H2COI] + k[499]*y[IDX_H2SiOI] + k[500]*y[IDX_HCOI] +
        k[503]*y[IDX_HNOI] + k[505]*y[IDX_SiH2I] + k[506]*y[IDX_SiH3I] +
        k[507]*y[IDX_SiH4I] + k[508]*y[IDX_SiHI];
    data[2938] = 0.0 - k[4]*y[IDX_H2I] + k[4]*y[IDX_H2I] + k[106]*y[IDX_H2II] +
        k[404]*y[IDX_CHII] + k[586]*y[IDX_H3II] + k[755]*y[IDX_NHII] +
        k[975]*y[IDX_HI];
    data[2939] = 0.0 + k[444]*y[IDX_CH3II] + k[599]*y[IDX_H3II] + k[823]*y[IDX_H2OII] +
        k[828]*y[IDX_NH3II] + k[834]*y[IDX_SiH3II] + k[894]*y[IDX_CH2I] +
        k[915]*y[IDX_CH3I] - k[965]*y[IDX_H2I] + k[1060]*y[IDX_H2CNI] +
        k[1085]*y[IDX_SiH2I];
    data[2940] = 0.0 - k[8]*y[IDX_H2I] + k[295]*y[IDX_CH2II] + k[299]*y[IDX_CH3II] +
        k[307]*y[IDX_H2COII] + k[311]*y[IDX_H2NOII] + k[312]*y[IDX_H2OII] +
        k[315]*y[IDX_H3II] + k[319]*y[IDX_H3COII] + k[323]*y[IDX_H3OII] +
        k[324]*y[IDX_H3OII] + k[353]*y[IDX_SiH2II] + k[357]*y[IDX_SiH3II] +
        k[358]*y[IDX_SiH4II] + k[360]*y[IDX_SiH5II];
    data[2941] = 0.0 + k[104]*y[IDX_H2II] + k[583]*y[IDX_H3II] + k[584]*y[IDX_H3II];
    data[2942] = 0.0 - k[2]*y[IDX_CHI] + k[2]*y[IDX_CHI] - k[3]*y[IDX_H2I] -
        k[3]*y[IDX_H2I] - k[3]*y[IDX_H2I] - k[3]*y[IDX_H2I] + k[3]*y[IDX_H2I] +
        k[3]*y[IDX_H2I] - k[4]*y[IDX_H2OI] + k[4]*y[IDX_H2OI] -
        k[5]*y[IDX_HOCII] + k[5]*y[IDX_HOCII] - k[6]*y[IDX_O2I] +
        k[6]*y[IDX_O2I] - k[7]*y[IDX_OHI] + k[7]*y[IDX_OHI] - k[8]*y[IDX_eM] -
        k[10]*y[IDX_HI] - k[115]*y[IDX_HeII] - k[233] - k[234] - k[235] -
        k[516]*y[IDX_H2II] - k[528]*y[IDX_CII] - k[529]*y[IDX_CHII] -
        k[530]*y[IDX_CH2II] - k[531]*y[IDX_CNII] - k[532]*y[IDX_COII] -
        k[533]*y[IDX_COII] - k[534]*y[IDX_H2OII] - k[535]*y[IDX_HCNII] -
        k[536]*y[IDX_HeII] - k[537]*y[IDX_HeHII] - k[538]*y[IDX_NII] -
        k[539]*y[IDX_N2II] - k[540]*y[IDX_NHII] - k[541]*y[IDX_NHII] -
        k[542]*y[IDX_NH2II] - k[543]*y[IDX_OII] - k[544]*y[IDX_O2HII] -
        k[545]*y[IDX_OHII] - k[546]*y[IDX_SiH4II] - k[547]*y[IDX_SiOII] -
        k[955]*y[IDX_CI] - k[956]*y[IDX_CH2I] - k[957]*y[IDX_CH3I] -
        k[958]*y[IDX_CHI] - k[959]*y[IDX_CNI] - k[960]*y[IDX_NI] -
        k[961]*y[IDX_NH2I] - k[962]*y[IDX_NHI] - k[963]*y[IDX_O2I] -
        k[964]*y[IDX_O2I] - k[965]*y[IDX_OI] - k[966]*y[IDX_OHI] -
        k[1199]*y[IDX_CII] - k[1200]*y[IDX_CI] - k[1201]*y[IDX_CHI] -
        k[1202]*y[IDX_SiII] - k[1203]*y[IDX_SiHII] - k[1204]*y[IDX_SiH3II] +
        (-H2dissociation);
    data[2943] = 0.0 - k[10]*y[IDX_H2I] + k[128]*y[IDX_H2II] + k[614]*y[IDX_CHII] +
        k[615]*y[IDX_CH2II] + k[616]*y[IDX_CH3II] + k[617]*y[IDX_CH4II] +
        k[619]*y[IDX_SiHII] + k[967]*y[IDX_CH2I] + k[968]*y[IDX_CH3I] +
        k[969]*y[IDX_CH4I] + k[970]*y[IDX_CHI] + k[973]*y[IDX_H2CNI] +
        k[974]*y[IDX_H2COI] + k[975]*y[IDX_H2OI] + k[976]*y[IDX_HCNI] +
        k[977]*y[IDX_HCOI] + k[981]*y[IDX_HNOI] + k[983]*y[IDX_NH2I] +
        k[984]*y[IDX_NH3I] + k[985]*y[IDX_NHI] + k[991]*y[IDX_O2HI] +
        k[996]*y[IDX_OHI] + (H2formation);
    data[2944] = 0.0 + k[254] - k[973]*y[IDX_HI] + k[1135];
    data[2945] = 0.0 + k[310]*y[IDX_eM] + k[1296];
    data[2946] = 0.0 + k[675]*y[IDX_HeII] + k[1144] + k[1144];
    data[2947] = 0.0 + k[335]*y[IDX_eM];
    data[2948] = 0.0 + k[336]*y[IDX_eM] - k[618]*y[IDX_HI];
    data[2949] = 0.0 + k[92]*y[IDX_HII];
    data[2950] = 0.0 + k[93]*y[IDX_HII];
    data[2951] = 0.0 + k[361]*y[IDX_eM] + k[1248];
    data[2952] = 0.0 + k[359]*y[IDX_eM] + k[546]*y[IDX_H2I];
    data[2953] = 0.0 + k[281] - k[990]*y[IDX_HI] - k[991]*y[IDX_HI] - k[992]*y[IDX_HI] +
        k[1170];
    data[2954] = 0.0 + k[94]*y[IDX_HII];
    data[2955] = 0.0 - k[986]*y[IDX_HI];
    data[2956] = 0.0 + k[356]*y[IDX_eM];
    data[2957] = 0.0 + k[354]*y[IDX_eM] + k[354]*y[IDX_eM] + k[355]*y[IDX_eM] +
        k[833]*y[IDX_OI];
    data[2958] = 0.0 - k[993]*y[IDX_HI] - k[994]*y[IDX_HI] - k[995]*y[IDX_HI];
    data[2959] = 0.0 + k[95]*y[IDX_HII] + k[289] + k[704]*y[IDX_HeII] + k[1086]*y[IDX_OI]
        + k[1086]*y[IDX_OI] + k[1181];
    data[2960] = 0.0 + k[364]*y[IDX_eM];
    data[2961] = 0.0 + k[352]*y[IDX_eM] + k[394]*y[IDX_CI] - k[619]*y[IDX_HI] +
        k[832]*y[IDX_OI] + k[1179];
    data[2962] = 0.0 + k[97]*y[IDX_HII] + k[708]*y[IDX_HeII] + k[1186] + k[1187];
    data[2963] = 0.0 + k[98]*y[IDX_HII] + k[292] + k[381]*y[IDX_CII] + k[709]*y[IDX_HeII]
        + k[877]*y[IDX_CI] + k[1089]*y[IDX_OI] + k[1188];
    data[2964] = 0.0 + k[96]*y[IDX_HII] + k[290] + k[706]*y[IDX_HeII] + k[1087]*y[IDX_OI]
        + k[1182];
    data[2965] = 0.0 + k[547]*y[IDX_H2I];
    data[2966] = 0.0 + k[331]*y[IDX_eM] + k[332]*y[IDX_eM] + k[1287];
    data[2967] = 0.0 + k[264] + k[686]*y[IDX_HeII] - k[980]*y[IDX_HI] - k[981]*y[IDX_HI]
        - k[982]*y[IDX_HI] + k[1068]*y[IDX_OI] + k[1153];
    data[2968] = 0.0 + k[712]*y[IDX_NII] + k[714]*y[IDX_NII] + k[715]*y[IDX_NII] +
        k[820]*y[IDX_O2II] + k[1120];
    data[2969] = 0.0 + k[83]*y[IDX_HII] + k[591]*y[IDX_H3II];
    data[2970] = 0.0 + k[301]*y[IDX_eM] + k[301]*y[IDX_eM] + k[302]*y[IDX_eM] -
        k[617]*y[IDX_HI] + k[1123];
    data[2971] = 0.0 + k[99]*y[IDX_HII];
    data[2972] = 0.0 - k[126]*y[IDX_HI] + k[531]*y[IDX_H2I];
    data[2973] = 0.0 + k[327]*y[IDX_eM] + k[327]*y[IDX_eM] + k[328]*y[IDX_eM] +
        k[329]*y[IDX_eM] + k[1301];
    data[2974] = 0.0 + k[338]*y[IDX_eM] + k[1302];
    data[2975] = 0.0 + k[347]*y[IDX_eM];
    data[2976] = 0.0 + k[476]*y[IDX_CHI] + k[572]*y[IDX_H2OI] + k[859]*y[IDX_OHI] -
        k[1209]*y[IDX_HI];
    data[2977] = 0.0 + k[91]*y[IDX_HII] + k[1102]*y[IDX_OHI];
    data[2978] = 0.0 + k[262] + k[683]*y[IDX_HeII] + k[684]*y[IDX_HeII] -
        k[979]*y[IDX_HI] + k[979]*y[IDX_HI] + k[1151];
    data[2979] = 0.0 + k[334]*y[IDX_eM];
    data[2980] = 0.0 + k[456]*y[IDX_CH4I] + k[539]*y[IDX_H2I] + k[730]*y[IDX_H2COI];
    data[2981] = 0.0 + k[319]*y[IDX_eM] + k[320]*y[IDX_eM] + k[321]*y[IDX_eM] +
        k[321]*y[IDX_eM];
    data[2982] = 0.0 + k[77]*y[IDX_HII] + k[456]*y[IDX_N2II] + k[511]*y[IDX_H2II] +
        k[658]*y[IDX_HeII] + k[660]*y[IDX_HeII] + k[716]*y[IDX_NII] +
        k[717]*y[IDX_NII] + k[718]*y[IDX_NII] + k[718]*y[IDX_NII] -
        k[969]*y[IDX_HI] + k[1125] + k[1127];
    data[2983] = 0.0 - k[127]*y[IDX_HI] + k[532]*y[IDX_H2I] + k[533]*y[IDX_H2I];
    data[2984] = 0.0 - k[128]*y[IDX_HI] + k[305]*y[IDX_eM] + k[305]*y[IDX_eM] +
        k[509]*y[IDX_CI] + k[510]*y[IDX_CH2I] + k[511]*y[IDX_CH4I] +
        k[512]*y[IDX_CHI] + k[513]*y[IDX_CNI] + k[514]*y[IDX_CO2I] +
        k[515]*y[IDX_COI] + k[516]*y[IDX_H2I] + k[517]*y[IDX_H2COI] +
        k[518]*y[IDX_H2OI] + k[520]*y[IDX_HeI] + k[521]*y[IDX_N2I] +
        k[522]*y[IDX_NI] + k[523]*y[IDX_NHI] + k[524]*y[IDX_NOI] +
        k[525]*y[IDX_O2I] + k[526]*y[IDX_OI] + k[527]*y[IDX_OHI] + k[1134];
    data[2985] = 0.0 + k[85]*y[IDX_HII] + k[271] + k[692]*y[IDX_HeII] - k[984]*y[IDX_HI]
        + k[1159];
    data[2986] = 0.0 + k[76]*y[IDX_HII] + k[244] + k[915]*y[IDX_OI] + k[916]*y[IDX_OI] +
        k[957]*y[IDX_H2I] - k[968]*y[IDX_HI] + k[1008]*y[IDX_NI] +
        k[1010]*y[IDX_NI] + k[1010]*y[IDX_NI] + k[1116];
    data[2987] = 0.0 + k[514]*y[IDX_H2II] - k[971]*y[IDX_HI];
    data[2988] = 0.0 + k[468]*y[IDX_CHI] + k[538]*y[IDX_H2I] + k[712]*y[IDX_CH3OHI] +
        k[714]*y[IDX_CH3OHI] + k[715]*y[IDX_CH3OHI] + k[716]*y[IDX_CH4I] +
        k[717]*y[IDX_CH4I] + k[718]*y[IDX_CH4I] + k[718]*y[IDX_CH4I] +
        k[726]*y[IDX_NHI];
    data[2989] = 0.0 - k[131]*y[IDX_HI] + k[472]*y[IDX_CHI] + k[543]*y[IDX_H2I] +
        k[803]*y[IDX_NHI] + k[819]*y[IDX_OHI];
    data[2990] = 0.0 - k[129]*y[IDX_HI] + k[326]*y[IDX_eM] + k[535]*y[IDX_H2I];
    data[2991] = 0.0 + k[341]*y[IDX_eM] + k[341]*y[IDX_eM] + k[342]*y[IDX_eM] +
        k[542]*y[IDX_H2I] + k[741]*y[IDX_NI] + k[827]*y[IDX_OI];
    data[2992] = 0.0 + k[340]*y[IDX_eM] + k[541]*y[IDX_H2I] + k[740]*y[IDX_NI];
    data[2993] = 0.0 + k[551]*y[IDX_H2COI] + k[820]*y[IDX_CH3OHI];
    data[2994] = 0.0 + k[298]*y[IDX_eM] + k[300]*y[IDX_eM] + k[300]*y[IDX_eM] +
        k[443]*y[IDX_OI] - k[616]*y[IDX_HI] + k[1115];
    data[2995] = 0.0 + k[84]*y[IDX_HII] + k[270] + k[373]*y[IDX_CII] + k[690]*y[IDX_HeII]
        + k[866]*y[IDX_CI] + k[867]*y[IDX_CI] + k[961]*y[IDX_H2I] -
        k[983]*y[IDX_HI] + k[1030]*y[IDX_NOI] + k[1072]*y[IDX_OI] + k[1158];
    data[2996] = 0.0 + k[296]*y[IDX_eM] + k[296]*y[IDX_eM] + k[297]*y[IDX_eM] +
        k[418]*y[IDX_H2OI] + k[421]*y[IDX_OI] + k[530]*y[IDX_H2I] -
        k[615]*y[IDX_HI] + k[736]*y[IDX_NI] + k[1110];
    data[2997] = 0.0 + k[313]*y[IDX_eM] + k[313]*y[IDX_eM] + k[314]*y[IDX_eM] +
        k[534]*y[IDX_H2I] + k[738]*y[IDX_NI] + k[1140];
    data[2998] = 0.0 + k[343]*y[IDX_eM] + k[344]*y[IDX_eM] + k[344]*y[IDX_eM];
    data[2999] = 0.0 + k[322]*y[IDX_eM] + k[323]*y[IDX_eM] + k[325]*y[IDX_eM] +
        k[325]*y[IDX_eM] + k[1286];
    data[3000] = 0.0 + k[521]*y[IDX_H2II];
    data[3001] = 0.0 + k[370]*y[IDX_H2OI] + k[371]*y[IDX_H2OI] + k[373]*y[IDX_NH2I] +
        k[375]*y[IDX_NHI] + k[379]*y[IDX_OHI] + k[381]*y[IDX_SiHI] +
        k[528]*y[IDX_H2I] - k[1205]*y[IDX_HI];
    data[3002] = 0.0 + k[81]*y[IDX_HII] + k[259] + k[676]*y[IDX_HeII] +
        k[678]*y[IDX_HeII] - k[976]*y[IDX_HI] + k[1065]*y[IDX_OI] + k[1147];
    data[3003] = 0.0 + k[241] + k[294]*y[IDX_eM] + k[402]*y[IDX_H2OI] + k[408]*y[IDX_NI]
        + k[414]*y[IDX_OI] + k[529]*y[IDX_H2I] - k[614]*y[IDX_HI];
    data[3004] = 0.0 + k[75]*y[IDX_HII] + k[243] + k[510]*y[IDX_H2II] +
        k[654]*y[IDX_HeII] + k[888]*y[IDX_NOI] + k[890]*y[IDX_O2I] +
        k[890]*y[IDX_O2I] + k[895]*y[IDX_OI] + k[895]*y[IDX_OI] +
        k[896]*y[IDX_OI] + k[898]*y[IDX_OHI] + k[956]*y[IDX_H2I] -
        k[967]*y[IDX_HI] + k[1005]*y[IDX_NI] + k[1006]*y[IDX_NI] + k[1113];
    data[3005] = 0.0 + k[308]*y[IDX_eM] + k[308]*y[IDX_eM] + k[309]*y[IDX_eM];
    data[3006] = 0.0 + k[86]*y[IDX_HII] + k[274] + k[375]*y[IDX_CII] + k[523]*y[IDX_H2II]
        + k[693]*y[IDX_HeII] + k[726]*y[IDX_NII] + k[803]*y[IDX_OII] +
        k[869]*y[IDX_CI] + k[962]*y[IDX_H2I] - k[985]*y[IDX_HI] +
        k[1018]*y[IDX_NI] + k[1039]*y[IDX_NHI] + k[1039]*y[IDX_NHI] +
        k[1039]*y[IDX_NHI] + k[1039]*y[IDX_NHI] + k[1042]*y[IDX_NOI] +
        k[1046]*y[IDX_OI] + k[1049]*y[IDX_OHI] + k[1162];
    data[3007] = 0.0 + k[348]*y[IDX_eM] + k[545]*y[IDX_H2I] + k[743]*y[IDX_NI] +
        k[830]*y[IDX_OI] + k[1173];
    data[3008] = 0.0 + k[513]*y[IDX_H2II] + k[959]*y[IDX_H2I] + k[1091]*y[IDX_OHI];
    data[3009] = 0.0 + k[79]*y[IDX_HII] + k[497]*y[IDX_HII] + k[517]*y[IDX_H2II] +
        k[551]*y[IDX_O2II] + k[671]*y[IDX_HeII] + k[730]*y[IDX_N2II] -
        k[974]*y[IDX_HI] + k[1137] + k[1137] + k[1139];
    data[3010] = 0.0 + k[82]*y[IDX_HII] + k[260] + k[680]*y[IDX_HeII] - k[977]*y[IDX_HI]
        - k[978]*y[IDX_HI] + k[1016]*y[IDX_NI] + k[1066]*y[IDX_OI] + k[1149];
    data[3011] = 0.0 - k[130]*y[IDX_HI] + k[536]*y[IDX_H2I] + k[654]*y[IDX_CH2I] +
        k[658]*y[IDX_CH4I] + k[660]*y[IDX_CH4I] + k[662]*y[IDX_CHI] +
        k[671]*y[IDX_H2COI] + k[673]*y[IDX_H2OI] + k[675]*y[IDX_H2SiOI] +
        k[676]*y[IDX_HCNI] + k[678]*y[IDX_HCNI] + k[680]*y[IDX_HCOI] +
        k[683]*y[IDX_HNCI] + k[684]*y[IDX_HNCI] + k[686]*y[IDX_HNOI] +
        k[690]*y[IDX_NH2I] + k[692]*y[IDX_NH3I] + k[693]*y[IDX_NHI] +
        k[699]*y[IDX_OHI] + k[704]*y[IDX_SiH2I] + k[706]*y[IDX_SiH3I] +
        k[708]*y[IDX_SiH4I] + k[709]*y[IDX_SiHI];
    data[3012] = 0.0 + k[2]*y[IDX_H2I] - k[9]*y[IDX_HI] + k[9]*y[IDX_HI] + k[9]*y[IDX_HI]
        + k[78]*y[IDX_HII] + k[250] + k[468]*y[IDX_NII] + k[472]*y[IDX_OII] +
        k[476]*y[IDX_SiII] + k[512]*y[IDX_H2II] + k[662]*y[IDX_HeII] +
        k[928]*y[IDX_NI] + k[932]*y[IDX_NOI] + k[933]*y[IDX_O2I] +
        k[934]*y[IDX_O2I] + k[939]*y[IDX_OI] + k[941]*y[IDX_OHI] +
        k[958]*y[IDX_H2I] - k[970]*y[IDX_HI] + k[1128];
    data[3013] = 0.0 + k[315]*y[IDX_eM] + k[316]*y[IDX_eM] + k[316]*y[IDX_eM] +
        k[316]*y[IDX_eM] + k[591]*y[IDX_MgI] + k[598]*y[IDX_OI] + k[1145];
    data[3014] = 0.0 + k[520]*y[IDX_H2II];
    data[3015] = 0.0 + k[87]*y[IDX_HII] + k[524]*y[IDX_H2II] + k[888]*y[IDX_CH2I] +
        k[932]*y[IDX_CHI] - k[987]*y[IDX_HI] - k[988]*y[IDX_HI] +
        k[1030]*y[IDX_NH2I] + k[1042]*y[IDX_NHI] + k[1099]*y[IDX_OHI];
    data[3016] = 0.0 + k[408]*y[IDX_CHII] + k[522]*y[IDX_H2II] + k[736]*y[IDX_CH2II] +
        k[738]*y[IDX_H2OII] + k[740]*y[IDX_NHII] + k[741]*y[IDX_NH2II] +
        k[743]*y[IDX_OHII] + k[928]*y[IDX_CHI] + k[960]*y[IDX_H2I] +
        k[1005]*y[IDX_CH2I] + k[1006]*y[IDX_CH2I] + k[1008]*y[IDX_CH3I] +
        k[1010]*y[IDX_CH3I] + k[1010]*y[IDX_CH3I] + k[1016]*y[IDX_HCOI] +
        k[1018]*y[IDX_NHI] + k[1025]*y[IDX_OHI];
    data[3017] = 0.0 + k[7]*y[IDX_H2I] - k[13]*y[IDX_HI] + k[13]*y[IDX_HI] +
        k[13]*y[IDX_HI] + k[90]*y[IDX_HII] + k[284] + k[379]*y[IDX_CII] +
        k[527]*y[IDX_H2II] + k[699]*y[IDX_HeII] + k[819]*y[IDX_OII] +
        k[855]*y[IDX_HCOII] + k[859]*y[IDX_SiII] + k[875]*y[IDX_CI] +
        k[898]*y[IDX_CH2I] + k[941]*y[IDX_CHI] + k[966]*y[IDX_H2I] -
        k[996]*y[IDX_HI] + k[1025]*y[IDX_NI] + k[1049]*y[IDX_NHI] +
        k[1080]*y[IDX_OI] + k[1091]*y[IDX_CNI] + k[1092]*y[IDX_COI] +
        k[1099]*y[IDX_NOI] + k[1102]*y[IDX_SiI] + k[1174] - k[1208]*y[IDX_HI];
    data[3018] = 0.0 - k[12]*y[IDX_HI] + k[12]*y[IDX_HI] + k[88]*y[IDX_HII] +
        k[525]*y[IDX_H2II] + k[890]*y[IDX_CH2I] + k[890]*y[IDX_CH2I] +
        k[933]*y[IDX_CHI] + k[934]*y[IDX_CHI] + k[963]*y[IDX_H2I] -
        k[989]*y[IDX_HI];
    data[3019] = 0.0 + k[394]*y[IDX_SiHII] + k[509]*y[IDX_H2II] + k[866]*y[IDX_NH2I] +
        k[867]*y[IDX_NH2I] + k[869]*y[IDX_NHI] + k[875]*y[IDX_OHI] +
        k[877]*y[IDX_SiHI] + k[955]*y[IDX_H2I] - k[1206]*y[IDX_HI];
    data[3020] = 0.0 + k[75]*y[IDX_CH2I] + k[76]*y[IDX_CH3I] + k[77]*y[IDX_CH4I] +
        k[78]*y[IDX_CHI] + k[79]*y[IDX_H2COI] + k[80]*y[IDX_H2OI] +
        k[81]*y[IDX_HCNI] + k[82]*y[IDX_HCOI] + k[83]*y[IDX_MgI] +
        k[84]*y[IDX_NH2I] + k[85]*y[IDX_NH3I] + k[86]*y[IDX_NHI] +
        k[87]*y[IDX_NOI] + k[88]*y[IDX_O2I] + k[89]*y[IDX_OI] + k[90]*y[IDX_OHI]
        + k[91]*y[IDX_SiI] + k[92]*y[IDX_SiC2I] + k[93]*y[IDX_SiC3I] +
        k[94]*y[IDX_SiCI] + k[95]*y[IDX_SiH2I] + k[96]*y[IDX_SiH3I] +
        k[97]*y[IDX_SiH4I] + k[98]*y[IDX_SiHI] + k[99]*y[IDX_SiOI] +
        k[497]*y[IDX_H2COI] - k[1197]*y[IDX_HI] + k[1216]*y[IDX_eM];
    data[3021] = 0.0 + k[330]*y[IDX_eM] + k[855]*y[IDX_OHI] + k[1148];
    data[3022] = 0.0 + k[4]*y[IDX_H2I] - k[11]*y[IDX_HI] + k[11]*y[IDX_HI] +
        k[11]*y[IDX_HI] + k[80]*y[IDX_HII] + k[256] + k[370]*y[IDX_CII] +
        k[371]*y[IDX_CII] + k[402]*y[IDX_CHII] + k[418]*y[IDX_CH2II] +
        k[518]*y[IDX_H2II] + k[572]*y[IDX_SiII] + k[673]*y[IDX_HeII] -
        k[975]*y[IDX_HI] + k[1142];
    data[3023] = 0.0 + k[89]*y[IDX_HII] + k[414]*y[IDX_CHII] + k[421]*y[IDX_CH2II] +
        k[443]*y[IDX_CH3II] + k[526]*y[IDX_H2II] + k[598]*y[IDX_H3II] +
        k[827]*y[IDX_NH2II] + k[830]*y[IDX_OHII] + k[832]*y[IDX_SiHII] +
        k[833]*y[IDX_SiH2II] + k[895]*y[IDX_CH2I] + k[895]*y[IDX_CH2I] +
        k[896]*y[IDX_CH2I] + k[915]*y[IDX_CH3I] + k[916]*y[IDX_CH3I] +
        k[939]*y[IDX_CHI] + k[965]*y[IDX_H2I] + k[1046]*y[IDX_NHI] +
        k[1065]*y[IDX_HCNI] + k[1066]*y[IDX_HCOI] + k[1068]*y[IDX_HNOI] +
        k[1072]*y[IDX_NH2I] + k[1080]*y[IDX_OHI] + k[1086]*y[IDX_SiH2I] +
        k[1086]*y[IDX_SiH2I] + k[1087]*y[IDX_SiH3I] + k[1089]*y[IDX_SiHI] -
        k[1207]*y[IDX_HI];
    data[3024] = 0.0 + k[8]*y[IDX_H2I] + k[8]*y[IDX_H2I] + k[294]*y[IDX_CHII] +
        k[296]*y[IDX_CH2II] + k[296]*y[IDX_CH2II] + k[297]*y[IDX_CH2II] +
        k[298]*y[IDX_CH3II] + k[300]*y[IDX_CH3II] + k[300]*y[IDX_CH3II] +
        k[301]*y[IDX_CH4II] + k[301]*y[IDX_CH4II] + k[302]*y[IDX_CH4II] +
        k[305]*y[IDX_H2II] + k[305]*y[IDX_H2II] + k[308]*y[IDX_H2COII] +
        k[308]*y[IDX_H2COII] + k[309]*y[IDX_H2COII] + k[310]*y[IDX_H2NOII] +
        k[313]*y[IDX_H2OII] + k[313]*y[IDX_H2OII] + k[314]*y[IDX_H2OII] +
        k[315]*y[IDX_H3II] + k[316]*y[IDX_H3II] + k[316]*y[IDX_H3II] +
        k[316]*y[IDX_H3II] + k[319]*y[IDX_H3COII] + k[320]*y[IDX_H3COII] +
        k[321]*y[IDX_H3COII] + k[321]*y[IDX_H3COII] + k[322]*y[IDX_H3OII] +
        k[323]*y[IDX_H3OII] + k[325]*y[IDX_H3OII] + k[325]*y[IDX_H3OII] +
        k[326]*y[IDX_HCNII] + k[327]*y[IDX_HCNHII] + k[327]*y[IDX_HCNHII] +
        k[328]*y[IDX_HCNHII] + k[329]*y[IDX_HCNHII] + k[330]*y[IDX_HCOII] +
        k[331]*y[IDX_HCO2II] + k[332]*y[IDX_HCO2II] + k[334]*y[IDX_HNOII] +
        k[335]*y[IDX_HOCII] + k[336]*y[IDX_HeHII] + k[338]*y[IDX_N2HII] +
        k[340]*y[IDX_NHII] + k[341]*y[IDX_NH2II] + k[341]*y[IDX_NH2II] +
        k[342]*y[IDX_NH2II] + k[343]*y[IDX_NH3II] + k[344]*y[IDX_NH3II] +
        k[344]*y[IDX_NH3II] + k[347]*y[IDX_O2HII] + k[348]*y[IDX_OHII] +
        k[352]*y[IDX_SiHII] + k[354]*y[IDX_SiH2II] + k[354]*y[IDX_SiH2II] +
        k[355]*y[IDX_SiH2II] + k[356]*y[IDX_SiH3II] + k[359]*y[IDX_SiH4II] +
        k[361]*y[IDX_SiH5II] + k[364]*y[IDX_SiOHII] + k[1216]*y[IDX_HII];
    data[3025] = 0.0 + k[515]*y[IDX_H2II] - k[972]*y[IDX_HI] + k[1092]*y[IDX_OHI];
    data[3026] = 0.0 + k[2]*y[IDX_CHI] + k[3]*y[IDX_H2I] + k[3]*y[IDX_H2I] +
        k[3]*y[IDX_H2I] + k[3]*y[IDX_H2I] + k[4]*y[IDX_H2OI] + k[7]*y[IDX_OHI] +
        k[8]*y[IDX_eM] + k[8]*y[IDX_eM] - k[10]*y[IDX_HI] + k[10]*y[IDX_HI] +
        k[10]*y[IDX_HI] + k[10]*y[IDX_HI] + k[233] + k[235] + k[235] +
        k[516]*y[IDX_H2II] + k[528]*y[IDX_CII] + k[529]*y[IDX_CHII] +
        k[530]*y[IDX_CH2II] + k[531]*y[IDX_CNII] + k[532]*y[IDX_COII] +
        k[533]*y[IDX_COII] + k[534]*y[IDX_H2OII] + k[535]*y[IDX_HCNII] +
        k[536]*y[IDX_HeII] + k[538]*y[IDX_NII] + k[539]*y[IDX_N2II] +
        k[541]*y[IDX_NHII] + k[542]*y[IDX_NH2II] + k[543]*y[IDX_OII] +
        k[545]*y[IDX_OHII] + k[546]*y[IDX_SiH4II] + k[547]*y[IDX_SiOII] +
        k[955]*y[IDX_CI] + k[956]*y[IDX_CH2I] + k[957]*y[IDX_CH3I] +
        k[958]*y[IDX_CHI] + k[959]*y[IDX_CNI] + k[960]*y[IDX_NI] +
        k[961]*y[IDX_NH2I] + k[962]*y[IDX_NHI] + k[963]*y[IDX_O2I] +
        k[965]*y[IDX_OI] + k[966]*y[IDX_OHI] + (2.0 * H2dissociation);
    data[3027] = 0.0 - k[9]*y[IDX_CHI] + k[9]*y[IDX_CHI] + k[9]*y[IDX_CHI] -
        k[10]*y[IDX_H2I] + k[10]*y[IDX_H2I] + k[10]*y[IDX_H2I] +
        k[10]*y[IDX_H2I] - k[11]*y[IDX_H2OI] + k[11]*y[IDX_H2OI] +
        k[11]*y[IDX_H2OI] - k[12]*y[IDX_O2I] + k[12]*y[IDX_O2I] -
        k[13]*y[IDX_OHI] + k[13]*y[IDX_OHI] + k[13]*y[IDX_OHI] -
        k[126]*y[IDX_CNII] - k[127]*y[IDX_COII] - k[128]*y[IDX_H2II] -
        k[129]*y[IDX_HCNII] - k[130]*y[IDX_HeII] - k[131]*y[IDX_OII] - k[236] -
        k[258] - k[614]*y[IDX_CHII] - k[615]*y[IDX_CH2II] - k[616]*y[IDX_CH3II]
        - k[617]*y[IDX_CH4II] - k[618]*y[IDX_HeHII] - k[619]*y[IDX_SiHII] -
        k[967]*y[IDX_CH2I] - k[968]*y[IDX_CH3I] - k[969]*y[IDX_CH4I] -
        k[970]*y[IDX_CHI] - k[971]*y[IDX_CO2I] - k[972]*y[IDX_COI] -
        k[973]*y[IDX_H2CNI] - k[974]*y[IDX_H2COI] - k[975]*y[IDX_H2OI] -
        k[976]*y[IDX_HCNI] - k[977]*y[IDX_HCOI] - k[978]*y[IDX_HCOI] -
        k[979]*y[IDX_HNCI] + k[979]*y[IDX_HNCI] - k[980]*y[IDX_HNOI] -
        k[981]*y[IDX_HNOI] - k[982]*y[IDX_HNOI] - k[983]*y[IDX_NH2I] -
        k[984]*y[IDX_NH3I] - k[985]*y[IDX_NHI] - k[986]*y[IDX_NO2I] -
        k[987]*y[IDX_NOI] - k[988]*y[IDX_NOI] - k[989]*y[IDX_O2I] -
        k[990]*y[IDX_O2HI] - k[991]*y[IDX_O2HI] - k[992]*y[IDX_O2HI] -
        k[993]*y[IDX_OCNI] - k[994]*y[IDX_OCNI] - k[995]*y[IDX_OCNI] -
        k[996]*y[IDX_OHI] - k[1197]*y[IDX_HII] - k[1205]*y[IDX_CII] -
        k[1206]*y[IDX_CI] - k[1207]*y[IDX_OI] - k[1208]*y[IDX_OHI] -
        k[1209]*y[IDX_SiII] + (-2.0 * H2formation);
    
    // clang-format on

    /* */

    return NAUNET_SUCCESS;
}