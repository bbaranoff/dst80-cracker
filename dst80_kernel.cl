#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// DST-80 OpenCL kernel: exact Python dst80_py.py transpose

#define BIT(x,n)            (((x)>>(n)) & 1UL)
#define BIT_SLICE(x,msb,lsb) (((x)&(((1UL<<((msb)+1))-1UL)))>>(lsb))
#define BV2I5(b4,b3,b2,b1,b0) (((b4)<<4)|((b3)<<3)|((b2)<<2)|((b1)<<1)|(b0))
#define BV2I4(b3,b2,b1,b0)    (((b3)<<3)|((b2)<<2)|((b1)<<1)|(b0))
#define KEY_MASK             ((1UL<<39) - 1UL)

static inline uint fa(uint x){ return BIT(0x3A35ACC5UL, x); }
static inline uint fb(uint x){ return BIT(0xAC35742EUL, x); }
static inline uint fc(uint x){ return BIT(0xB81D8BD1UL, x); }
static inline uint fd(uint x){ return BIT(0x5ACC335AUL, x); }
static inline uint fe(uint x){ return BIT(0xE247UL,     x); }
static inline uint fg(uint x){ return BIT(0x4E72UL,    x); }
static inline uint h(uint x) { return (0x1F9826F4UL >> (2*(x))) & 3UL; }

static inline void fn_bits(ulong s, ulong k, uint fx[16]){
    uint fs[16];
    fs[ 0]=fd(BV2I5(BIT(s,32),BIT(k,32),BIT(s,24),BIT(k,24),BIT(s,16)));
    fs[ 1]=fe(BV2I4(BIT(k,16),BIT(s,8),BIT(k,8),BIT(k,0)));
    fs[ 2]=fb(BV2I5(BIT(s,33),BIT(k,33),BIT(s,25),BIT(k,25),BIT(s,17)));
    fs[ 3]=fe(BV2I4(BIT(k,17),BIT(s,9),BIT(k,9),BIT(k,1)));
    fs[ 4]=fd(BV2I5(BIT(s,34),BIT(k,34),BIT(s,26),BIT(k,26),BIT(s,18)));
    fs[ 5]=fc(BV2I5(BIT(k,18),BIT(s,10),BIT(k,10),BIT(s,2),BIT(k,2)));
    fs[ 6]=fb(BV2I5(BIT(s,35),BIT(k,35),BIT(s,27),BIT(k,27),BIT(s,19)));
    fs[ 7]=fa(BV2I5(BIT(k,19),BIT(s,11),BIT(k,11),BIT(s,3),BIT(k,3)));
    fs[ 8]=fd(BV2I5(BIT(s,36),BIT(k,36),BIT(s,28),BIT(k,28),BIT(s,20)));
    fs[ 9]=fc(BV2I5(BIT(k,20),BIT(s,12),BIT(k,12),BIT(s,4),BIT(k,4)));
    fs[10]=fb(BV2I5(BIT(s,37),BIT(k,37),BIT(s,29),BIT(k,29),BIT(s,21)));
    fs[11]=fa(BV2I5(BIT(k,21),BIT(s,13),BIT(k,13),BIT(s,5),BIT(k,5)));
    fs[12]=fd(BV2I5(BIT(s,38),BIT(k,38),BIT(s,30),BIT(k,30),BIT(s,22)));
    fs[13]=fc(BV2I5(BIT(k,22),BIT(s,14),BIT(k,14),BIT(s,6),BIT(k,6)));
    fs[14]=fb(BV2I5(BIT(s,39),BIT(k,39),BIT(s,31),BIT(k,31),BIT(s,23)));
    fs[15]=fa(BV2I5(BIT(k,23),BIT(s,15),BIT(k,15),BIT(s,7),BIT(k,7)));
    for(int i=0;i<16;i++) fx[i]=fs[15-i];
}

static inline void g_bits(ulong s, ulong k, uint out[4]){
    uint fx[16]; fn_bits(s,k,fx);
    out[0]=fg(BV2I4(fx[12],fx[13],fx[14],fx[15]));
    out[1]=fg(BV2I4(fx[ 8],fx[ 9],fx[10],fx[11]));
    out[2]=fg(BV2I4(fx[ 4],fx[ 5],fx[ 6],fx[ 7]));
    out[3]=fg(BV2I4(fx[ 0],fx[ 1],fx[ 2],fx[ 3]));
}

static inline uint f_func(ulong k, ulong s){
    uint gb[4]; g_bits(s,k,gb);
    return h(BV2I4(gb[0],gb[1],gb[2],gb[3]));
}

static inline uchar p1(uchar x){
    uchar o=x&0xA5;
    o|=((x>>6)&1)<<3; o|=((x>>4)&1)<<1;
    o|=((x>>3)&1)<<6; o|=((x>>1)&1)<<4;
    return o;
}

static inline ulong p2(ulong x){
    ulong o=0UL;
    for(int i=0;i<5;i++) o|=((ulong)p1((x>>(8*i))&0xFF))<<(8*i);
    return o;
}

static inline ulong dst80_merge(ulong keyl, ulong keyr){
    keyl=p2(keyl); if(BIT(keyl,39)) keyl^=KEY_MASK;
    keyr=p2(keyr); if(BIT(keyr,39)) keyr^=KEY_MASK;
    return ((BIT_SLICE(keyl,39,20)<<20)|BIT_SLICE(keyr,39,20));
}

static inline ulong lfsr_round(ulong x){
    uint fb=(BIT(x,0)^BIT(x,2)^BIT(x,19)^BIT(x,21))&1UL;
    return (x>>1)|((ulong)fb<<39);
}

__kernel void dst80_kernel(
    __global const ulong* keyl_buf,
    __global const ulong* keyr_buf,
    __global const ulong* chal_buf,
    __global       uint*  out_buf
){
    size_t gid=get_global_id(0);
    ulong keyl=keyl_buf[gid], keyr=keyr_buf[gid], s=chal_buf[gid];
    for(int r=0;r<200;r++){
        ulong k=dst80_merge(keyl,keyr);
        uint t=f_func(k,s)^(s&3UL);
        s=(s>>2)|((ulong)t<<38);
        keyr=lfsr_round(keyr);
        keyl=lfsr_round(keyl);
    }
    out_buf[gid]=(uint)(s&0xFFFFFFUL);
}
