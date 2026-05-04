/*==================================================================
 * AMAZONS ENGINE — bitboard.h
 * 128-bit bitboard for 10×10 board (100 squares).
 * Uses two uint64_t: lo covers sq 0-63, hi covers sq 64-99.
 *
 * PEXT Attack Tables: O(1) queen attack lookup via BMI2 PEXT.
 * BETWEEN_BB / LINE_BB: precomputed path masks.
 *==================================================================*/
#pragma once
#include <cstdint>

#ifdef _MSC_VER
#include <intrin.h>
#include <immintrin.h>
#pragma intrinsic(_BitScanForward64, _BitScanReverse64, __popcnt64)
#else
#include <x86intrin.h>
#endif

// ── Bitboard128 ──────────────────────────────────────────────────
struct Bitboard128 {
    uint64_t lo, hi;

    // ── Construction ─────────────────────────────────────────────
    static constexpr Bitboard128 zero() { return {0ULL, 0ULL}; }

    static Bitboard128 from_sq(int sq) {
        if (sq < 64) return {1ULL << sq, 0ULL};
        return {0ULL, 1ULL << (sq - 64)};
    }

    // ── Bit manipulation ─────────────────────────────────────────
    void set(int sq)   { if (sq < 64) lo |= (1ULL << sq); else hi |= (1ULL << (sq - 64)); }
    void clear(int sq) { if (sq < 64) lo &= ~(1ULL << sq); else hi &= ~(1ULL << (sq - 64)); }
    bool test(int sq) const {
        if (sq < 64) return (lo >> sq) & 1;
        return (hi >> (sq - 64)) & 1;
    }

    // ── Bitwise operators ────────────────────────────────────────
    Bitboard128 operator&(Bitboard128 b) const { return {lo & b.lo, hi & b.hi}; }
    Bitboard128 operator|(Bitboard128 b) const { return {lo | b.lo, hi | b.hi}; }
    Bitboard128 operator^(Bitboard128 b) const { return {lo ^ b.lo, hi ^ b.hi}; }
    Bitboard128 operator~() const { return {~lo, ~hi}; }
    Bitboard128& operator&=(Bitboard128 b) { lo &= b.lo; hi &= b.hi; return *this; }
    Bitboard128& operator|=(Bitboard128 b) { lo |= b.lo; hi |= b.hi; return *this; }
    Bitboard128& operator^=(Bitboard128 b) { lo ^= b.lo; hi ^= b.hi; return *this; }
    bool operator==(Bitboard128 b) const { return lo == b.lo && hi == b.hi; }
    bool operator!=(Bitboard128 b) const { return lo != b.lo || hi != b.hi; }
    explicit operator bool() const { return lo || hi; }

    // ── Population count (hardware accelerated) ──────────────────
    int popcount() const {
#ifdef _MSC_VER
        return (int)__popcnt64(lo) + (int)__popcnt64(hi);
#else
        return __builtin_popcountll(lo) + __builtin_popcountll(hi);
#endif
    }

    // ── Lowest set bit index (-1 if empty) ───────────────────────
    int lsb() const {
        if (lo) {
#ifdef _MSC_VER
            unsigned long idx; _BitScanForward64(&idx, lo); return (int)idx;
#else
            return __builtin_ctzll(lo);
#endif
        }
        if (hi) {
#ifdef _MSC_VER
            unsigned long idx; _BitScanForward64(&idx, hi); return 64 + (int)idx;
#else
            return 64 + __builtin_ctzll(hi);
#endif
        }
        return -1;
    }

    // ── Pop lowest set bit (return index and clear it) ───────────
    int pop_lsb() {
        if (lo) {
#ifdef _MSC_VER
            unsigned long idx; _BitScanForward64(&idx, lo);
            lo &= lo - 1; // clear lowest bit
            return (int)idx;
#else
            int idx = __builtin_ctzll(lo);
            lo &= lo - 1;
            return idx;
#endif
        }
        if (hi) {
#ifdef _MSC_VER
            unsigned long idx; _BitScanForward64(&idx, hi);
            hi &= hi - 1;
            return 64 + (int)idx;
#else
            int idx = __builtin_ctzll(hi);
            hi &= hi - 1;
            return 64 + idx;
#endif
        }
        return -1;
    }
};

// ── Shift operations (for flood fill / king attacks) ─────────────
inline Bitboard128 bb_shl(Bitboard128 b, int n) {
    if (n == 0)   return b;
    if (n >= 128) return Bitboard128::zero();
    if (n >= 64)  return {0ULL, b.lo << (n - 64)};
    return {b.lo << n, (b.hi << n) | (b.lo >> (64 - n))};
}

inline Bitboard128 bb_shr(Bitboard128 b, int n) {
    if (n == 0)   return b;
    if (n >= 128) return Bitboard128::zero();
    if (n >= 64)  return {b.hi >> (n - 64), 0ULL};
    return {(b.lo >> n) | (b.hi << (64 - n)), b.hi >> n};
}

// ── Precomputed tables ───────────────────────────────────────────
extern Bitboard128 SQUARE_BB[100];   // single-bit bitboard per square
extern Bitboard128 NEIGHBOR_BB[100]; // 8-neighbor king attacks per square
extern Bitboard128 FILE_A_BB;        // all squares on column 0
extern Bitboard128 FILE_J_BB;        // all squares on column 9
extern Bitboard128 BOARD_MASK_BB;    // all 100 valid squares

// ── PEXT Attack Tables ──────────────────────────────────────────
// Direction indices for attack table lookups
enum Dir4 : int { DIR_RANK = 0, DIR_FILE = 1, DIR_DIAG = 2, DIR_ADIAG = 3 };

// Line masks: all squares on the line through sq in direction dir, EXCLUDING sq
extern Bitboard128 LINE_MASK[100][4];

// Precomputed popcount of the lo half of each LINE_MASK
extern int LINE_MASK_LO_POPCNT[100][4];

// Attack table: [sq][dir][occupancy_index] → reachable squares bitboard
// Max 9 bits per line (10 squares minus sq itself) → 512 entries
extern Bitboard128 ATTACK_TABLE[100][4][512];

// ── BETWEEN_BB: squares strictly between sq1 and sq2 ─────────────
extern Bitboard128 BETWEEN_BB[100][100];

// ── LINE_BB: full line through sq1 and sq2 (if on same line) ─────
extern Bitboard128 LINE_BB[100][100];

// Initialize all bitboard tables (call once at startup)
void init_bitboards();

// Initialize PEXT attack tables (call once at startup, after init_bitboards)
void init_attack_tables();

// Initialize BETWEEN_BB and LINE_BB (call once at startup, after init_bitboards)
void init_between_line_bb();

// ── PEXT helper: extract 128-bit occupancy to compact index ──────
inline unsigned pext128(Bitboard128 occ, Bitboard128 mask, int lo_popcnt) {
#if defined(__BMI2__) || defined(_MSC_VER)
    unsigned lo_bits = (unsigned)_pext_u64(occ.lo, mask.lo);
    unsigned hi_bits = (unsigned)_pext_u64(occ.hi, mask.hi);
    return lo_bits | (hi_bits << lo_popcnt);
#else
    // Software fallback for PEXT
    unsigned result = 0;
    unsigned bit = 0;
    // Process lo
    uint64_t m = mask.lo;
    uint64_t o = occ.lo;
    while (m) {
        uint64_t lsb = m & (-m);  // isolate lowest set bit
        if (o & lsb) result |= (1u << bit);
        bit++;
        m &= m - 1;  // clear lowest bit
    }
    // Process hi
    m = mask.hi;
    o = occ.hi;
    while (m) {
        uint64_t lsb = m & (-m);
        if (o & lsb) result |= (1u << bit);
        bit++;
        m &= m - 1;
    }
    return result;
#endif
}

// ── O(1) Queen attacks from sq given occupied bitboard ───────────
// 4 PEXT lookups + 3 ORs — no loops!
inline Bitboard128 get_queen_attacks(int sq, Bitboard128 occupied) {
    return ATTACK_TABLE[sq][DIR_RANK] [pext128(occupied, LINE_MASK[sq][DIR_RANK],  LINE_MASK_LO_POPCNT[sq][DIR_RANK])]
         | ATTACK_TABLE[sq][DIR_FILE] [pext128(occupied, LINE_MASK[sq][DIR_FILE],  LINE_MASK_LO_POPCNT[sq][DIR_FILE])]
         | ATTACK_TABLE[sq][DIR_DIAG] [pext128(occupied, LINE_MASK[sq][DIR_DIAG],  LINE_MASK_LO_POPCNT[sq][DIR_DIAG])]
         | ATTACK_TABLE[sq][DIR_ADIAG][pext128(occupied, LINE_MASK[sq][DIR_ADIAG], LINE_MASK_LO_POPCNT[sq][DIR_ADIAG])];
}

// ── Path clearance check (O(1) via BETWEEN_BB) ──────────────────
// Returns true if the path from sq1 to sq2 is clear (no occupied squares between them)
inline bool is_path_clear(int sq1, int sq2, Bitboard128 occupied) {
    return (BETWEEN_BB[sq1][sq2] & occupied) == Bitboard128::zero();
}

// ── King-attack expansion (for bitboard flood fill) ──────────────
// Returns all squares attacked by kings on every set bit of 'b'.
// Used for fast partition detection and territory computation.
inline Bitboard128 king_attacks_all(Bitboard128 b) {
    Bitboard128 not_a = ~FILE_A_BB;
    Bitboard128 not_j = ~FILE_J_BB;

    Bitboard128 atk = Bitboard128::zero();
    atk |= bb_shl(b, 10);               // North
    atk |= bb_shr(b, 10);               // South
    atk |= bb_shl(b, 1)  & not_a;       // East
    atk |= bb_shr(b, 1)  & not_j;       // West
    atk |= bb_shl(b, 11) & not_a;       // NE
    atk |= bb_shl(b, 9)  & not_j;       // NW
    atk |= bb_shr(b, 9)  & not_a;       // SE
    atk |= bb_shr(b, 11) & not_j;       // SW
    return atk & BOARD_MASK_BB;
}

// ── Bitboard flood fill ─────────────────────────────────────────
// Flood from 'seed' through squares in 'passable'. Returns all
// reachable squares (including seed). O(diameter) iterations.
inline Bitboard128 bb_flood_fill(Bitboard128 seed, Bitboard128 passable) {
    Bitboard128 filled = seed & passable;
    while (true) {
        Bitboard128 expanded = king_attacks_all(filled) & passable & ~filled;
        if (!expanded) break;
        filled |= expanded;
    }
    return filled;
}
