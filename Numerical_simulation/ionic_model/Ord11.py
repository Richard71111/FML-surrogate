import torch
from torch import Tensor
from typing import Dict, Tuple

@torch.no_grad()
def Ord11_model(t:float, 
                state:Tensor, 
                parameters: Dict, 
                S: Tensor,
                device:torch.device,
                dtype: torch.dtype = torch.float32) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute the derivatives for the Ord11 ionic current model.
        https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002061
        This is the vectorized code for ORd11 model, the goal of this model is to simulate human ventricular cardiomyocyte action potential.
        Will return the total ionic current at time t for all compartments(N). shape (N,)
        the compartments can be ID cleft compartment of cell.

        Units:

        time in ms
        voltage in mV
        current in uA
        concentration in mM
        conductance in mS
        capacitance in uF (uF*mV/ms = uA)

        Basic terminology:
            N_cells:     number of cells
            N_int:       number of internal compartments per cell
            N_junctions: number of junctions over the cable (N_junctions = N_cells - 1)
            M:           number of the single FEM mesh partitions
            N:           total number of axial compartments (N = N_cells * N_int)
            N_total:     total number of compartments including junctions (N_total = N_cells * N_int + 2*M*N_junctions+1)
                         Here we +1 for ground compartment at the right end of the cable
        If we use NN to replace FEM, then we do not need to use N_total

        Args:
            t[Tensor]: current time in ms for stimulating current
            state: Optional[Tensor] of axial voltages/gating variables shape (2*N,)
            parameters: dictionary of parameters, for example:
                {
                    'dt': time step in ms,
                    'L': cell length in um    
                }
            S: Optional[Tensor]: Ionic concentrations shape (3*N,)
                in Ord model we have 3 [Na, K, Ca] concentrations per compartment
        Returns:
            Iion [Tensor]: ionic current shape (N,)
            current_matrix [Tensor]: matrix of all individual ionic currents shape (N, num_currents)
            G [Tensor]: gating variables shape (num_gating_variables, N)
    """

    # Basic parameters
    dt = parameters.get('dt', 0.01)  # time step in ms
    # ``dt`` is a Python scalar in eager mode and a scalar Tensor in the
    # compiled path. ``as_tensor`` handles both without copying an existing
    # Tensor (and avoids torch.compile's tensor-copy warning).
    dt = torch.as_tensor(dt, device=device, dtype=dtype)
    N  = parameters.get('N', 1)      # number of compartments
    dX = torch.zeros_like(state, device=device, dtype=dtype)  # placeholder for derivatives
    # Physical constants
    R = 8314.0     # J/(mol·K)
    T = 310.0      # K
    F = 96485.0    # C/mol
    # Cell geometry
    L = parameters['L'] / 1e4  # cm
    rad = parameters['r'] / 1e4  # cm
    vcell = 1000 * torch.pi * rad * rad * L  # uL
    Ageo = 2 * torch.pi * rad * rad + 2 * torch.pi * rad * L  # cm^2
    Rcg = 2
    Acap = Rcg * Ageo          # uF
    vmyo = 0.68 * vcell        # uL
    vnsr = 0.0552 * vcell      # uL
    vjsr = 0.0048 * vcell      # uL
    vss = 0.02 * vcell         # uL
    # Additional scaling factors
    fSERCA = parameters.get('fSERCA', 1.0)
    fRyR = parameters.get('fRyR', 1.0)
    ftauhL = parameters.get('ftauhL', 1.0)
    fCaMKa = parameters.get('fCaMKa', 1.0)
    fIleak = parameters.get('fIleak', 1.0)
    fJrel = parameters.get('fJrel', 1.0)
    celltype = parameters.get('celltype', 0)  # 0: endo, 1: epi, 2: M cell
    f_I = parameters.get('f_I')
    # Concentrations we only have 3 concentrations: [Na, K, Ca]
    nao = S[0:N]                  # extracellular Na concentration in mM shape (N,)
    ko  = S[N:2*N]                # extracellular K concentration in mM shape (N,)
    cao = S[2*N:3*N]               # extracellular Ca concentration in mM shape (N,)

    # assign state variables
    v  = state[0:N]                  # voltage in mV shape (N,)
    G  = state[N:]                   # gating variables
    nai   = state[1 * N : 2 * N]
    nass  = state[2 * N : 3 * N]
    ki    = state[3 * N : 4 * N]
    kss   = state[4 * N : 5 * N]
    cai   = state[5 * N : 6 * N]
    cass  = state[6 * N : 7 * N]
    cansr = state[7 * N : 8 * N]
    cajsr = state[8 * N : 9 * N]
    m     = state[9 * N : 10 * N]
    hf    = state[10 * N : 11 * N]
    hs    = state[11 * N : 12 * N]
    j     = state[12 * N : 13 * N]
    hsp   = state[13 * N : 14 * N]
    jp    = state[14 * N : 15 * N]
    mL    = state[15 * N : 16 * N]
    hL    = state[16 * N : 17 * N]
    hLp   = state[17 * N : 18 * N]
    a     = state[18 * N : 19 * N]
    iF    = state[19 * N : 20 * N]
    iS    = state[20 * N : 21 * N]
    ap    = state[21 * N : 22 * N]
    iFp   = state[22 * N : 23 * N]
    iSp   = state[23 * N : 24 * N]
    d     = state[24 * N : 25 * N]
    ff    = state[25 * N : 26 * N]
    fs    = state[26 * N : 27 * N]
    fcaf  = state[27 * N : 28 * N]
    fcas  = state[28 * N : 29 * N]
    jca   = state[29 * N : 30 * N]
    nca   = state[30 * N : 31 * N]
    ffp   = state[31 * N : 32 * N]
    fcafp = state[32 * N : 33 * N]
    xrf   = state[33 * N : 34 * N]
    xrs   = state[34 * N : 35 * N]
    xs1   = state[35 * N : 36 * N]
    xs2   = state[36 * N : 37 * N]
    xk1   = state[37 * N : 38 * N]
    Jrelnp = state[38 * N : 39 * N] 
    Jrelp  = state[39 * N : 40 * N]
    CaMKt  = state[40 * N : 41 * N]

    # CaMk constants
    KmCaMK = 0.15
    aCaMK = 0.05
    bCaMK = 0.00068
    CaMKo = 0.05
    KmCaM = 0.0015
    CaMKb = CaMKo * (1.0 - CaMKt) / (1.0 + KmCaM / cass)
    CaMKa = fCaMKa * (CaMKb + CaMKt)
    dCaMKt = aCaMK * CaMKb * (CaMKb + CaMKt) - bCaMK * CaMKt

    # Nernst potentials
    ENa = (R * T / F) * torch.log(nao / nai)
    EK  = (R * T / F) * torch.log(ko / ki)
    PKNa = 0.01833
    EKs = (R * T / F) * torch.log((ko + PKNa * nao) / (ki + PKNa * nai))
    
    # convenient shorthand calculations
    vffrt = v * F * F / (R * T)
    vfrt  = v * F / (R * T)


    # Calculate ionic currents in uA/uF

    # Calculate INa fast sodium current
    GNa = 75.0
    ftauhL = parameters.get("ftauhL", 1.0)
    mss = 1.0 / (1.0 + torch.exp((-(v + 39.57)) / 9.871))
    tm  = 1.0 / (6.765 * torch.exp((v + 11.64) / 34.77) +
                 8.552 * torch.exp(-(v + 77.42) / 5.955))

    hss = 1.0 / (1.0 + torch.exp((v + 82.90) / 6.086))
    thf = 1.0 / (1.432e-5 * torch.exp(-(v + 1.196) / 6.285) +
                 6.149 * torch.exp((v + 0.5096) / 20.27))
    ths = 1.0 / (0.009794 * torch.exp(-(v + 17.95) / 28.05) +
                 0.3343 * torch.exp((v + 5.730) / 56.66))
    Ahf = 0.99
    Ahs = 1.0 - Ahf
    h = Ahf * hf + Ahs * hs

    jss = hss
    tj = 2.038 + 1.0 / (0.02136 * torch.exp(-(v + 100.6) / 8.281) +
                        0.3052 * torch.exp((v + 0.9941) / 38.45))

    hssp = 1.0 / (1.0 + torch.exp((v + 89.1) / 6.086))
    thsp = 3.0 * ths
    hp = Ahf * hf + Ahs * hsp
    tjp = 1.46 * tj
    fINap = 1.0 / (1.0 + KmCaMK / CaMKa)
    INa = f_I[parameters.get('iina')]*(
        GNa
        * (v - ENa)
        * m**3
        * ((1.0 - fINap) * h * j + fINap * hp * jp)
    )

    # INaL Late sodium current
    mLss = 1.0 / (1.0 + torch.exp((-(v + 42.85)) / 5.264))
    tmL = tm

    hLss = 1.0 / (1.0 + torch.exp((v + 87.61) / 7.488))
    thL = ftauhL * 200.0

    hLssp = 1.0 / (1.0 + torch.exp((v + 93.81) / 7.488))
    thLp = 3.0 * thL
    torch.full
    GNaL = 0.0075

    fINaLp = 1.0 / (1.0 + KmCaMK / CaMKa)
    INaL = f_I[parameters.get('iinal')]*(
        GNaL
        * (v - ENa)
        * mL
        * ((1.0 - fINaLp) * hL + fINaLp * hLp)
    )

    # --------------------------------------------------
    # Transient Outward Potassium Current (Ito)
    # --------------------------------------------------
    ass = 1.0 / (1.0 + torch.exp((-(v - 14.34)) / 14.82))
    ta = 1.0515 / (
        1.0 / (1.2089 * (1.0 + torch.exp(-(v - 18.4099) / 29.3814))) +
        3.5 / (1.0 + torch.exp((v + 100.0) / 29.3814))
    )

    iss = 1.0 / (1.0 + torch.exp((v + 43.94) / 5.711))


    tiF = 4.562 + 1.0 / (
        0.3933 * torch.exp((-(v + 100.0)) / 100.0) +
        0.08004 * torch.exp((v + 50.0) / 16.59)
    )
    tiS = 23.62 + 1.0 / (
        0.001416 * torch.exp((-(v + 96.52)) / 59.05) +
        1.780e-8 * torch.exp((v + 114.1) / 8.079)
    )
    tiF = tiF 
    tiS = tiS 

    AiF = 1.0 / (1.0 + torch.exp((v - 213.6) / 151.2))
    AiS = 1.0 - AiF
    i = AiF * iF + AiS * iS

    assp = 1.0 / (1.0 + torch.exp((-(v - 24.34)) / 14.82))

    dti_develop = 1.354 + 1.0e-4 / (
        torch.exp((v - 167.4) / 15.89) +
        torch.exp(-(v - 12.23) / 0.2154)
    )
    dti_recover = 1.0 - 0.5 / (1.0 + torch.exp((v + 70.0) / 20.0))
    tiFp = dti_develop * dti_recover * tiF
    tiSp = dti_develop * dti_recover * tiS
    ip = AiF * iFp + AiS * iSp

    Gto = 0.02

    fItop = 1.0 / (1.0 + KmCaMK / CaMKa)
    Ito = f_I[parameters.get('iito')]*(
        Gto
        * (v - EK)
        * ((1.0 - fItop) * a * i + fItop * ap * ip)
    )

    # --------------------------------------------------
    # 9. ICaL, ICaNa, ICaK
    # --------------------------------------------------
    dss = 1.0 / (1.0 + torch.exp((-(v + 3.940)) / 4.230))
    td = 0.6 + 1.0 / (torch.exp(-0.05 * (v + 6.0)) +
                      torch.exp(0.09 * (v + 14.0)))

    fss = 1.0 / (1.0 + torch.exp((v + 19.58) / 3.696))
    tff = 7.0 + 1.0 / (0.0045 * torch.exp(-(v + 20.0) / 10.0) +
                       0.0045 * torch.exp((v + 20.0) / 10.0))
    tfs = 1000.0 + 1.0 / (0.000035 * torch.exp(-(v + 5.0) / 4.0) +
                          0.000035 * torch.exp((v + 5.0) / 6.0))
    Aff = 0.6
    Afs = 1.0 - Aff
    f = Aff * ff + Afs * fs

    fcass = fss
    tfcaf = 7.0 + 1.0 / (0.04 * torch.exp(-(v - 4.0) / 7.0) +
                         0.04 * torch.exp((v - 4.0) / 7.0))
    tfcas = 100.0 + 1.0 / (0.00012 * torch.exp(-v / 3.0) +
                           0.00012 * torch.exp(v / 7.0))
    Afcaf = 0.3 + 0.6 / (1.0 + torch.exp((v - 10.0) / 10.0))
    Afcas = 1.0 - Afcaf
    fca = Afcaf * fcaf + Afcas * fcas

    tjca = 75.0

    tffp = 2.5 * tff
    fp = Aff * ffp + Afs * fs
    tfcafp = 2.5 * tfcaf
    fcap = Afcaf * fcafp + Afcas * fcas

    Kmn = 0.002
    k2n = 1000.0
    km2n = jca * 1.0
    anca = 1.0 / (k2n / km2n + (1.0 + Kmn / cass) ** 4.0)
    dnca = anca * k2n - nca * km2n

    PhiCaL = (
        4.0
        * vffrt
        * (cass * torch.exp(2.0 * vfrt) - 0.341 * cao)
        / (torch.exp(2.0 * vfrt) - 1.0)
    )
    PhiCaNa = (
        vffrt
        * (0.75 * nass * torch.exp(vfrt) - 0.75 * nao)
        / (torch.exp(vfrt) - 1.0)
    )
    PhiCaK = (
        vffrt
        * (0.75 * kss * torch.exp(vfrt) - 0.75 * ko)
        / (torch.exp(vfrt) - 1.0)
    )

    PCa = 0.0001

    PCap  = 1.1 * PCa
    PCaNa = 0.00125 * PCa
    PCaK  = 3.574e-4 * PCa
    PCaNap = 0.00125 * PCap
    PCaKp  = 3.574e-4 * PCap

    fICaLp = 1.0 / (1.0 + KmCaMK / CaMKa)

    # ICaL
    ICaL = f_I[parameters.get('iical')]*(
        (1.0 - fICaLp)
        * PCa
        * PhiCaL
        * d
        * (f * (1.0 - nca) + jca * fca * nca)
        +
        fICaLp
        * PCap
        * PhiCaL
        * d
        * (fp * (1.0 - nca) + jca * fcap * nca)
    )

    # ICaNa
    ICaNa = f_I[parameters.get('iical')]*(
        (1.0 - fICaLp)
        * PCaNa
        * PhiCaNa
        * d
        * (f * (1.0 - nca) + jca * fca * nca)
        +
        fICaLp
        * PCaNap
        * PhiCaNa
        * d
        * (fp * (1.0 - nca) + jca * fcap * nca)
    )

    # ICaK
    ICaK = f_I[parameters.get('iical')]*(
        (1.0 - fICaLp)
        * PCaK
        * PhiCaK
        * d
        * (f * (1.0 - nca) + jca * fca * nca)
        +
        fICaLp
        * PCaKp
        * PhiCaK
        * d
        * (fp * (1.0 - nca) + jca * fcap * nca)
    )

    # --------------------------------------------------
    # 10. IKr
    # --------------------------------------------------
    xrss = 1.0 / (1.0 + torch.exp((-(v + 8.337)) / 6.789))
    txrf = 12.98 + 1.0 / (
        0.3652 * torch.exp((v - 31.66) / 3.869) +
        4.123e-5 * torch.exp((-(v - 47.78)) / 20.38)
    )
    txrs = 1.865 + 1.0 / (
        0.06629 * torch.exp((v - 34.70) / 7.355) +
        1.128e-5 * torch.exp((-(v - 29.74)) / 25.94)
    )
    Axrf = 1.0 / (1.0 + torch.exp((v + 54.81) / 38.21))
    Axrs = 1.0 - Axrf
    xr = Axrf * xrf + Axrs * xrs

    rkr = 1.0 / (1.0 + torch.exp((v + 55.0) / 75.0)) * \
          1.0 / (1.0 + torch.exp((v - 10.0) / 30.0))

    GKr = 0.046

    IKr = f_I[parameters.get('iikr')]*(
        GKr
        * torch.sqrt(ko / 5.4)
        * xr
        * rkr
        * (v - EK)
    )

    # --------------------------------------------------
    # 11. IKs
    # --------------------------------------------------
    xs1ss = 1.0 / (1.0 + torch.exp((-(v + 11.60)) / 8.932))
    txs1  = 817.3 + 1.0 / (
        2.326e-4 * torch.exp((v + 48.28) / 17.80) +
        0.001292 * torch.exp((-(v + 210.0)) / 230.0)
    )
    xs2ss = xs1ss
    txs2  = 1.0 / (
        0.01 * torch.exp((v - 50.0) / 20.0) +
        0.0193 * torch.exp((-(v + 66.54)) / 31.0)
    )

    KsCa = 1.0 + 0.6 / (1.0 + (3.8e-5 / cai) ** 1.4)

    GKs = 0.0034

    IKs = f_I[parameters.get('iiks')]*(
        GKs
        * KsCa
        * xs1
        * xs2
        * (v - EKs)
    )

    # --------------------------------------------------
    # 12. IK1
    # --------------------------------------------------
    xk1ss = 1.0 / (1.0 + torch.exp(
        -(v + 2.5538 * ko + 144.59) / (1.5692 * ko + 3.8115)
    ))
    txk1 = 122.2 / (
        torch.exp((-(v + 127.2)) / 20.36) +
        torch.exp((v + 236.8) / 69.33)
    )
    rk1 = 1.0 / (1.0 + torch.exp((v + 105.8 - 2.6 * ko) / 9.493))

    GK1 = 0.1908
    IK1 = f_I[parameters.get('iik1')]*(
        GK1
        * torch.sqrt(ko)
        * rk1
        * xk1
        * (v - EK)
    )

    # --------------------------------------------------
    # INaCa_i / INaCa_ss (NCX)
    # --------------------------------------------------
    kna1 = 15.0
    kna2 = 5.0
    kna3 = 88.12
    kasymm = 12.5
    wna = 6.0e4
    wca = 6.0e4
    wnaca = 5.0e3
    kcaon = 1.5e6
    kcaoff = 5.0e3
    qna = 0.5224
    qca = 0.1670

    hca = torch.exp((qca * v * F) / (R * T))
    hna = torch.exp((qna * v * F) / (R * T))

    h1 = 1 + nai / kna3 * (1 + hna)
    h2 = (nai * hna) / (kna3 * h1)
    h3 = 1.0 / h1
    h4 = 1.0 + nai / kna1 * (1 + nai / kna2)
    h5 = nai * nai / (h4 * kna1 * kna2)
    h6 = 1.0 / h4
    h7 = 1.0 + nao / kna3 * (1.0 + 1.0 / hna)
    h8 = nao / (kna3 * hna * h7)
    h9 = 1.0 / h7
    h10 = kasymm + 1.0 + nao / kna1 * (1.0 + nao / kna2)
    h11 = nao * nao / (h10 * kna1 * kna2)
    h12 = 1.0 / h10

    k1 = h12 * cao * kcaon
    k2 = kcaoff
    k3p = h9 * wca
    k3pp = h8 * wnaca
    k3 = k3p + k3pp
    k4p = h3 * wca / hca
    k4pp = h2 * wnaca
    k4 = k4p + k4pp
    k5 = kcaoff
    k6 = h6 * cai * kcaon
    k7 = h5 * h2 * wna
    k8 = h8 * h11 * wna

    x1 = k2 * k4 * (k7 + k6) + k5 * k7 * (k2 + k3)
    x2 = k1 * k7 * (k4 + k5) + k4 * k6 * (k1 + k8)
    x3 = k1 * k3 * (k7 + k6) + k8 * k6 * (k2 + k3)
    x4 = k2 * k8 * (k4 + k5) + k3 * k5 * (k1 + k8)

    E1 = x1 / (x1 + x2 + x3 + x4)
    E2 = x2 / (x1 + x2 + x3 + x4)
    E3 = x3 / (x1 + x2 + x3 + x4)
    E4 = x4 / (x1 + x2 + x3 + x4)

    KmCaAct = 150.0e-6
    allo = 1.0 / (1.0 + (KmCaAct / cai) ** 2.0)
    zna = 1.0

    JncxNa = 3.0 * (E4 * k7 - E1 * k8) + E3 * k4pp - E2 * k3pp
    JncxCa = E2 * k2 - E1 * k1

    Gncx = 0.0008
    INaCa_i = f_I[parameters.get('iinaca_i')]*(0.8 * Gncx * allo * (zna * JncxNa + 2.0 * JncxCa))

    h1 = 1 + nass / kna3 * (1 + hna)
    h2 = (nass * hna) / (kna3 * h1)
    h3 = 1.0 / h1
    h4 = 1.0 + nass / kna1 * (1 + nass / kna2)
    h5 = nass * nass / (h4 * kna1 * kna2)
    h6 = 1.0 / h4
    h7 = 1.0 + nao / kna3 * (1.0 + 1.0 / hna)
    h8 = nao / (kna3 * hna * h7)
    h9 = 1.0 / h7
    h10 = kasymm + 1.0 + nao / kna1 * (1.0 + nao / kna2)
    h11 = nao * nao / (h10 * kna1 * kna2)
    h12 = 1.0 / h10

    k1 = h12 * cao * kcaon
    k2 = kcaoff
    k3p = h9 * wca
    k3pp = h8 * wnaca
    k3 = k3p + k3pp
    k4p = h3 * wca / hca
    k4pp = h2 * wnaca
    k4 = k4p + k4pp
    k5 = kcaoff
    k6 = h6 * cass * kcaon
    k7 = h5 * h2 * wna
    k8 = h8 * h11 * wna

    x1 = k2 * k4 * (k7 + k6) + k5 * k7 * (k2 + k3)
    x2 = k1 * k7 * (k4 + k5) + k4 * k6 * (k1 + k8)
    x3 = k1 * k3 * (k7 + k6) + k8 * k6 * (k2 + k3)
    x4 = k2 * k8 * (k4 + k5) + k3 * k5 * (k1 + k8)

    E1 = x1 / (x1 + x2 + x3 + x4)
    E2 = x2 / (x1 + x2 + x3 + x4)
    E3 = x3 / (x1 + x2 + x3 + x4)
    E4 = x4 / (x1 + x2 + x3 + x4)

    KmCaAct = 150.0e-6
    allo = 1.0 / (1.0 + (KmCaAct / cass) ** 2.0)
    zna = 1.0

    JncxNa = 3.0 * (E4 * k7 - E1 * k8) + E3 * k4pp - E2 * k3pp
    JncxCa = E2 * k2 - E1 * k1

    INaCa_ss = f_I[parameters.get('iinaca_ss')]*(0.2 * Gncx * allo * (zna * JncxNa + 2.0 * JncxCa))

    # --------------------------------------------------
    # 14. INaK
    # --------------------------------------------------
    kna1 = 949.5
    k1m = 182.4
    k2p = 687.2
    k2m = 39.4
    k3p = 1899.0
    k3m = 79300.0
    k4p = 639.0
    k4m = 40.0

    Knai0 = 9.073
    Knao0 = 27.78
    delta = -0.1550
    Knai = Knai0 * torch.exp((delta * v * F) / (3.0 * R * T))
    Knao = Knao0 * torch.exp(((1.0 - delta) * v * F) / (3.0 * R * T))

    Kki = 0.5
    Kko = 0.3582
    MgADP = 0.05
    MgATP = 9.8
    Kmgatp = 1.698e-7
    H = 1.0e-7
    eP = 4.2
    Khp = 1.698e-7
    Knap = 224.0
    Kxkur = 292.0

    P = eP / (1.0 + H / Khp + nai / Knap + ki / Kxkur)

    a1 = (kna1 * (nai / Knai) ** 3.0) / (
        (1.0 + nai / Knai) ** 3.0 + (1.0 + ki / Kki) ** 2.0 - 1.0
    )
    b1 = k1m * MgADP
    a2 = k2p
    b2 = (k2m * (nao / Knao) ** 3.0) / (
        (1.0 + nao / Knao) ** 3.0 + (1.0 + ko / Kko) ** 2.0 - 1.0
    )
    a3 = (k3p * (ko / Kko) ** 2.0) / (
        (1.0 + nao / Knao) ** 3.0 + (1.0 + ko / Kko) ** 2.0 - 1.0
    )
    b3 = (k3m * P * H) / (1.0 + MgATP / Kmgatp)
    a4 = (k4p * MgATP / Kmgatp) / (1.0 + MgATP / Kmgatp)
    b4 = (k4m * (ki / Kki) ** 2.0) / (
        (1.0 + nai / Knai) ** 3.0 + (1.0 + ki / Kki) ** 2.0 - 1.0
    )

    x1 = a4 * a1 * a2 + b2 * b4 * b3 + a2 * b4 * b3 + b3 * a1 * a2
    x2 = b2 * b1 * b4 + a1 * a2 * a3 + a3 * b1 * b4 + a2 * a3 * b4
    x3 = a2 * a3 * a4 + b3 * b2 * b1 + b2 * b1 * a4 + a3 * a4 * b1
    x4 = b4 * b3 * b2 + a3 * a4 * a1 + b2 * a4 * a1 + b3 * b2 * a1

    E1 = x1 / (x1 + x2 + x3 + x4)
    E2 = x2 / (x1 + x2 + x3 + x4)
    E3 = x3 / (x1 + x2 + x3 + x4)
    E4 = x4 / (x1 + x2 + x3 + x4)

    zna = 1.0
    zk = 1.0

    JnakNa = 3.0 * (E1 * a3 - E2 * b3)
    JnakK  = 2.0 * (E4 * b1 - E3 * a1)

    Pnak = 30.0

    INaK = f_I[parameters.get('iinak')]*(
        Pnak
        * (zna * JnakNa + zk * JnakK)
    )
    # --------------------------------------------------
    # 15. IKb, INab, ICab, IpCa
    # --------------------------------------------------
    # IKb
    xkb = 1.0 / (1.0 + torch.exp(-(v - 14.48) / 18.34))
    GKb = 0.003

    IKb = f_I[parameters.get('iikb')]*GKb * xkb * (v - EK)

    # INab
    PNab = 3.75e-10
    INab = f_I[parameters.get('iinab')]*(
        PNab
        * vffrt
        * (nai * torch.exp(vfrt) - nao)
        / (torch.exp(vfrt) - 1.0)
    )

    # ICab
    PCab = 2.5e-8
    ICab = f_I[parameters.get('iicab')]*(
        PCab
        * 4.0
        * vffrt
        * (cai * torch.exp(2.0 * vfrt) - 0.341 * cao)
        / (torch.exp(2.0 * vfrt) - 1.0)
    )

    # IpCa
    GpCa = 0.0005
    IpCa = f_I[parameters.get('iipca')]*(
        GpCa
        * cai
        / (0.0005 + cai)
    )

    # --------------------------------------------------
    # Istim
    # --------------------------------------------------
    # ``%`` is identical to np.mod for the non-negative simulation times used
    # here and remains traceable when ``t`` is a scalar tensor in the optional
    # torch.compile execution path.
    t_mod_bcl = t % parameters.get('bcl', 1000.0) # ms
    is_stimulated = (t_mod_bcl < parameters.get('stim_dur', 1.0))  # ms
    stim_factor = is_stimulated
    Istim = parameters.get('stim_amp', 0.0) * stim_factor * parameters.get('indstim', 0.0)
    scaled_factor = Rcg * parameters.get('Ctot', 1.0)
    Iion = scaled_factor*(
        (INa
        + INaL
        + Ito
        + ICaL
        + ICaNa
        + ICaK
        + IKr
        + IKs
        + IK1
        + INaCa_i
        + INaCa_ss
        + INaK
        + INab
        + IKb
        + IpCa
        + ICab)
        - Istim
    )

    # Saving individual currents
    scaled_INa = scaled_factor * INa
    scaled_INaL = scaled_factor * INaL
    scaled_Ito = scaled_factor * Ito
    scaled_ICaL = scaled_factor * ICaL
    scaled_IKr = scaled_factor * IKr
    scaled_IKs = scaled_factor * IKs
    scaled_IK1 = scaled_factor * IK1
    scaled_INaCa_i = scaled_factor * INaCa_i
    scaled_INaCa_ss = scaled_factor * INaCa_ss
    scaled_INaK = scaled_factor * INaK
    scaled_IKb = scaled_factor * IKb
    scaled_INab = scaled_factor * INab
    scaled_ICab = scaled_factor * ICab
    scaled_IpCa = scaled_factor * IpCa

    current_matrix = torch.stack([
        scaled_INa.squeeze(),
        scaled_INaL.squeeze(),
        scaled_Ito.squeeze(),
        scaled_ICaL.squeeze(),
        scaled_IKr.squeeze(),
        scaled_IKs.squeeze(),
        scaled_IK1.squeeze(),
        scaled_INaCa_i.squeeze(),
        scaled_INaCa_ss.squeeze(),
        scaled_INaK.squeeze(),
        scaled_IKb.squeeze(),
        scaled_INab.squeeze(),
        scaled_ICab.squeeze(),
        scaled_IpCa.squeeze(),
    ], dim=0) # shape: (Ncurrents, Nm)

    # Updating gating variables

    # --- calculate diffusion fluxes ---
    JdiffNa = (nass - nai) / 2.0
    JdiffK = (kss - ki) / 2.0
    Jdiff = (cass - cai) / 0.2

    # --- calculate ryanodione receptor calcium induced calcium release from the jsr ---
    bt = 4.75
    a_rel = 0.5 * bt
    Jrel_inf = fJrel * a_rel * (-ICaL) / (1.0 + (1.5 / cajsr)**8.0)
    tau_rel = bt / (1.0 + 0.0123 / cajsr)
    tau_rel[tau_rel < 0.001] = 0.001
    btp = 1.25 * bt
    a_relp = 0.5 * btp
    Jrel_infp = a_relp * (-ICaL) / (1.0 + (1.5 / cajsr)**8.0)
    tau_relp = btp / (1.0 + 0.0123 / cajsr)
    tau_relp[tau_relp < 0.001] = 0.001
    fJrelp = 1.0 / (1.0 + KmCaMK / CaMKa)
    Jrel = fRyR * ((1.0 - fJrelp) * Jrelnp + fJrelp * Jrelp)

    # --- calculate serca pump, ca uptake flux ---
    Jupnp = 0.004375 * cai / (cai + 0.00092)
    Jupp = 2.75 * 0.004375 * cai / (cai + 0.00092 - 0.00017)
    fJupp = 1.0 / (1.0 + KmCaMK / CaMKa)
    Jleak = fIleak * 0.0039375 * cansr / 15.0
    Jup = fSERCA * ((1.0 - fJupp) * Jupnp + fJupp * Jupp - Jleak)

    # --- calculate tranlocation flux ---
    Jtr = (cansr - cajsr) / 100.0

    # --- calcium buffer constants ---
    cmdnmax = 0.05
    kmcmdn = 0.00238
    trpnmax = 0.07
    kmtrpn = 0.0005
    BSRmax = 0.047
    KmBSR = 0.00087
    BSLmax = 1.124
    KmBSL = 0.0087
    csqnmax = 10.0
    kmcsqn = 0.8

    # --- Concentration Time Derivatives (dnai, dnass, dki, dkss) ---
    dnai = -(INa + INaL + 3.0 * INaCa_i + 3.0 * INaK + INab) * Acap / (F * vmyo) + JdiffNa * vss / vmyo + 0.0

    dnass = -(ICaNa + 3.0 * INaCa_ss) * Acap / (F * vss) - JdiffNa

    dki = -(Ito + IKr + IKs + IK1 + IKb + Istim - 2.0 * INaK) * Acap / (F * vmyo) + JdiffK * vss / vmyo + 0.0

    dkss = -(ICaK) * Acap / (F * vss) - JdiffK

    # --- Cai Buffer and Derivative ---
    Bcai = 1.0 / (1.0 + cmdnmax * kmcmdn / (kmcmdn + cai)**2.0 + trpnmax * kmtrpn / (kmtrpn + cai)**2.0)

    dcai = Bcai * (-(IpCa + ICab - 2.0 * INaCa_i) * Acap / (2.0 * F * vmyo) - Jup * vnsr / vmyo + Jdiff * vss / vmyo + 0.0)

    # --- Cass Buffer and Derivative ---
    Bcass = 1.0 / (1.0 + BSRmax * KmBSR / (KmBSR + cass)**2.0 + BSLmax * KmBSL / (KmBSL + cass)**2.0)

    dcass = Bcass * (-(ICaL - 2.0 * INaCa_ss) * Acap / (2.0 * F * vss) + Jrel * vjsr / vss - Jdiff)

    # --- Cansr Derivative ---
    dcansr = Jup - Jtr * vjsr / vnsr

    # --- Cajsr Buffer and Derivative ---
    Bcajsr = 1.0 / (1.0 + csqnmax * kmcsqn / (kmcsqn + cajsr)**2.0)

    dcajsr = Bcajsr * (Jtr - Jrel)
    
    # --- Forward Euler for Concentrations ---
    G_new = G.clone()
    G_new[0*N:1*N] = nai + dt * dnai
    G_new[1*N:2*N] = nass + dt * dnass
    G_new[2*N:3*N] = ki + dt * dki
    G_new[3*N:4*N] = kss + dt * dkss
    G_new[4*N:5*N] = cai + dt * dcai
    G_new[5*N:6*N] = cass + dt * dcass
    G_new[6*N:7*N] = cansr + dt * dcansr
    G_new[7*N:8*N] = cajsr + dt * dcajsr

    # --- Rush-Larsen for Gating Variables ---

    # m
    G_new[8*N:9*N] = mss - (mss - m) * torch.exp(-dt / tm)

    # hf, hs
    G_new[9*N:10*N] = hss - (hss - hf) * torch.exp(-dt / thf)
    G_new[10*N:11*N] = hss - (hss - hs) * torch.exp(-dt / ths)
    # j
    G_new[11*N:12*N] = jss - (jss - j) * torch.exp(-dt / tj)

    # hsp, jp
    G_new[12*N:13*N] = hssp - (hssp - hsp) * torch.exp(-dt / thsp)
    G_new[13*N:14*N] = jss - (jss - jp) * torch.exp(-dt / tjp)

    # mL, hL, hLp
    G_new[14*N:15*N] = mLss - (mLss - mL) * torch.exp(-dt / tmL)
    G_new[15*N:16*N] = hLss - (hLss - hL) * torch.exp(-dt / thL)
    G_new[16*N:17*N] = hLssp - (hLssp - hLp) * torch.exp(-dt / thLp)

    # a, iF, iS
    G_new[17*N:18*N] = ass - (ass - a) * torch.exp(-dt / ta)
    G_new[18*N:19*N] = iss - (iss - iF) * torch.exp(-dt / tiF)
    G_new[19*N:20*N] = iss - (iss - iS) * torch.exp(-dt / tiS)

    # ap, iFp, iSp
    G_new[20*N:21*N] = assp - (assp - ap) * torch.exp(-dt / ta)
    G_new[21*N:22*N] = iss - (iss - iFp) * torch.exp(-dt / tiFp)
    G_new[22*N:23*N] = iss - (iss - iSp) * torch.exp(-dt / tiSp)

    # d, ff, fs
    G_new[23*N:24*N] = dss - (dss - d) * torch.exp(-dt / td)
    G_new[24*N:25*N] = fss - (fss - ff) * torch.exp(-dt / tff)
    G_new[25*N:26*N] = fss - (fss - fs) * torch.exp(-dt / tfs)

    # fcaf, fcas, jca
    G_new[26*N:27*N] = fcass - (fcass - fcaf) * torch.exp(-dt / tfcaf)
    G_new[27*N:28*N] = fcass - (fcass - fcas) * torch.exp(-dt / tfcas)
    G_new[28*N:29*N] = fcass - (fcass - jca) * torch.exp(-dt / tjca)

    # nca (Forward Euler)
    G_new[29*N:30*N] = nca + dt * dnca

    # ffp, fcafp
    G_new[30*N:31*N] = fss - (fss - ffp) * torch.exp(-dt / tffp)
    G_new[31*N:32*N] = fcass - (fcass - fcafp) * torch.exp(-dt / tfcafp)
    # xrf, xrs
    G_new[32*N:33*N] = xrss - (xrss - xrf) * torch.exp(-dt / txrf)
    G_new[33*N:34*N] = xrss - (xrss - xrs) * torch.exp(-dt / txrs)

    # xs1, xs2, xk1
    G_new[34*N:35*N] = xs1ss - (xs1ss - xs1) * torch.exp(-dt / txs1)
    G_new[35*N:36*N] = xs2ss - (xs2ss - xs2) * torch.exp(-dt / txs2)
    G_new[36*N:37*N] = xk1ss - (xk1ss - xk1) * torch.exp(-dt / txk1)

    # Jrelnp, Jrelp
    G_new[37*N:38*N] = Jrel_inf - (Jrel_inf - Jrelnp) * torch.exp(-dt / tau_rel)
    G_new[38*N:39*N] = Jrel_infp - (Jrel_infp - Jrelp) * torch.exp(-dt / tau_relp)
    # CaMKt (Forward Euler)
    G_new[39*N:40*N] = CaMKt + dt * dCaMKt

    if N == 1:
        dX[0:N] = -Iion / parameters['Ctot']
        dX[N:2*N] = dnai
        dX[2*N:3*N] = dnass
        dX[3*N:4*N] = dki
        dX[4*N:5*N] = dkss
        dX[5*N:6*N] = dcai
        dX[6*N:7*N] = dcass
        dX[7*N:8*N] = dcansr
        dX[8*N:9*N] = dcajsr
        dX[9*N:10*N] = (mss - m) / tm
        dX[10*N:11*N] = (hss - hf) / thf
        dX[11*N:12*N] = (hss - hs) / ths
        dX[12*N:13*N] = (jss - j) / tj
        dX[13*N:14*N] = (hssp - hsp) / thsp
        dX[14*N:15*N] = (jss - jp) / tjp
        dX[15*N:16*N] = (mLss - mL) / tmL
        dX[16*N:17*N] = (hLss - hL) / thL
        dX[17*N:18*N] = (hLssp - hLp) / thLp
        dX[18*N:19*N] = (ass - a) / ta
        dX[19*N:20*N] = (iss - iF) / tiF
        dX[20*N:21*N] = (iss - iS) / tiS
        dX[21*N:22*N] = (assp - ap) / ta
        dX[22*N:23*N] = (iss - iFp) / tiFp
        dX[23*N:24*N] = (iss - iSp) / tiSp
        dX[24*N:25*N] = (dss - d) / td
        dX[25*N:26*N] = (fss - ff) / tff
        dX[26*N:27*N] = (fss - fs) / tfs
        dX[27*N:28*N] = (fcass - fcaf) / tfcaf
        dX[28*N:29*N] = (fcass - fcas) / tfcas
        dX[29*N:30*N] = (fcass - jca) / tjca
        dX[30*N:31*N] = dnca
        dX[31*N:32*N] = (fss - ffp) / tffp
        dX[32*N:33*N] = (fcass - fcafp) / tfcafp
        dX[33*N:34*N] = (xrss - xrf) / txrf
        dX[34*N:35*N] = (xrss - xrs) / txrs
        dX[35*N:36*N] = (xs1ss - xs1) / txs1
        dX[36*N:37*N] = (xs2ss - xs2) / txs2
        dX[37*N:38*N] = (xk1ss - xk1) / txk1
        dX[38*N:39*N] = (Jrel_inf - Jrelnp) / tau_rel
        dX[39*N:40*N] = (Jrel_infp - Jrelp) / tau_relp
        dX[40*N:41*N] = dCaMKt
    return Iion, current_matrix, G_new, dX
