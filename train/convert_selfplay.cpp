#include "../bitboard.h"
#include "../evaluate.h"
#include "../line_pattern.h"
#include "../position.h"

#include <cstdint>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {

constexpr int BOARD_SIZE = 100;
constexpr char FEATURE_MAGIC[8] = {'A', 'M', 'Z', 'F', 'V', '2', '\0', '\0'};

#pragma pack(push, 1)
struct GameRecord {
    int8_t side;
    char board[BOARD_SIZE];
    int8_t outcome;
    int16_t score;
};

struct FeatureHeader {
    char magic[8];
    uint32_t version;
    uint32_t board_size;
    uint32_t global_feature_size;
};

struct FeatureRecordV2 {
    int8_t side;
    char board[BOARD_SIZE];
    int8_t outcome;
    int16_t score;
    int16_t classical;
    uint8_t phase_bucket;
    int8_t global[NNUE::GLOBAL_FEATURE_SIZE];
};
#pragma pack(pop)

static_assert(sizeof(GameRecord) == 104, "GameRecord size mismatch");

void init_engine_tables() {
    Zobrist::init();
    init_rays();
    init_bitboards();
    init_attack_tables();
    init_between_line_bb();
    init_line_patterns();
}

bool build_position(const GameRecord& src, Position& pos) {
    std::memset(pos.board, EMPTY, sizeof(pos.board));
    int counts[2] = {0, 0};
    pos.bb_white = Bitboard128::zero();
    pos.bb_black = Bitboard128::zero();
    pos.bb_arrow = Bitboard128::zero();
    pos.side_to_move = src.side == 0 ? WHITE : BLACK;
    pos.ply = 0;
    pos.key = 0;

    for (int sq = 0; sq < BOARD_SIZE; ++sq) {
        const char ch = src.board[sq];
        int8_t cell = EMPTY;
        if (ch == '1')
            cell = WHITE_AMAZON;
        else if (ch == '2')
            cell = BLACK_AMAZON;
        else if (ch == '3')
            cell = ARROW;
        else if (ch != '0')
            return false;

        pos.board[sq] = cell;
        if (cell == WHITE_AMAZON) {
            if (counts[WHITE] >= NUM_AMAZONS)
                return false;
            pos.amazons[WHITE][counts[WHITE]++] = sq;
            pos.bb_white.set(sq);
        } else if (cell == BLACK_AMAZON) {
            if (counts[BLACK] >= NUM_AMAZONS)
                return false;
            pos.amazons[BLACK][counts[BLACK]++] = sq;
            pos.bb_black.set(sq);
        } else if (cell == ARROW) {
            pos.bb_arrow.set(sq);
        }

        if (cell != EMPTY)
            pos.key ^= Zobrist::piece_sq[cell - 1][sq];
    }

    if (counts[WHITE] != NUM_AMAZONS || counts[BLACK] != NUM_AMAZONS)
        return false;

    if (pos.side_to_move == BLACK)
        pos.key ^= Zobrist::side;

    pos.bb_occupied = pos.bb_white | pos.bb_black | pos.bb_arrow;
    return true;
}

void encode_record(const GameRecord& src, FeatureRecordV2& dst, Position& pos) {
    std::memset(&dst, 0, sizeof(dst));
    dst.side = src.side;
    std::memcpy(dst.board, src.board, BOARD_SIZE);
    dst.outcome = src.outcome;
    dst.score = src.score;

    if (!build_position(src, pos))
        return;

    EvalInfo info;
    get_eval_info(pos, info);
    dst.classical = static_cast<int16_t>(std::clamp(info.breakdown.classical, -32768, 32767));
    dst.phase_bucket = static_cast<uint8_t>(info.phase_bucket);
    std::memcpy(dst.global, info.global.v, sizeof(dst.global));
}

void print_usage(const char* exe) {
    std::cout << "Usage:\n"
              << "  " << exe << " [input_bin] [output_bin] [--append]\n\n"
              << "Defaults:\n"
              << "  input_bin  = selfplay_data.bin\n"
              << "  output_bin = selfplay_features_v2.bin\n"
              << "  output_bin = -  (write binary feature stream to stdout)\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string input_path = "selfplay_data.bin";
    std::string output_path = "selfplay_features_v2.bin";
    bool append = false;

    if (argc >= 2) {
        std::string arg = argv[1];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        input_path = arg;
    }
    if (argc >= 3)
        output_path = argv[2];
    if (argc >= 4)
        append = std::string(argv[3]) == "--append";

    init_engine_tables();

    std::ifstream in(input_path, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open input file: " << input_path << "\n";
        return 1;
    }

    const bool use_stdout = output_path == "-";
    std::unique_ptr<std::ofstream> out_file;
    std::ostream* out = nullptr;

    if (use_stdout) {
#ifdef _WIN32
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        out = &std::cout;
    } else {
        std::ios::openmode mode = std::ios::binary | std::ios::out;
        mode |= append ? std::ios::app : std::ios::trunc;
        out_file = std::make_unique<std::ofstream>(output_path, mode);
        if (!(*out_file)) {
            std::cerr << "Failed to open output file: " << output_path << "\n";
            return 1;
        }
        out = out_file.get();
    }

    if (!append) {
        FeatureHeader header{};
        std::memcpy(header.magic, FEATURE_MAGIC, sizeof(header.magic));
        header.version = 2;
        header.board_size = BOARD_SIZE;
        header.global_feature_size = NNUE::GLOBAL_FEATURE_SIZE;
        out->write(reinterpret_cast<const char*>(&header), sizeof(header));
    }

    constexpr std::size_t CHUNK_RECORDS = 1 << 15;
    std::vector<GameRecord> input_buf(CHUNK_RECORDS);
    std::vector<FeatureRecordV2> output_buf(CHUNK_RECORDS);
    auto pos = std::make_unique<Position>();

    std::uint64_t total_records = 0;
    while (true) {
        in.read(reinterpret_cast<char*>(input_buf.data()),
                static_cast<std::streamsize>(input_buf.size() * sizeof(GameRecord)));
        const std::streamsize bytes = in.gcount();
        if (bytes <= 0)
            break;
        if (bytes % static_cast<std::streamsize>(sizeof(GameRecord)) != 0)
            std::cerr << "Warning: trailing bytes not aligned to GameRecord, ignored.\n";

        const std::size_t records = static_cast<std::size_t>(bytes / sizeof(GameRecord));
        for (std::size_t i = 0; i < records; ++i)
            encode_record(input_buf[i], output_buf[i], *pos);

        out->write(reinterpret_cast<const char*>(output_buf.data()),
                   static_cast<std::streamsize>(records * sizeof(FeatureRecordV2)));
        if (!(*out)) {
            std::cerr << "Write failed while writing output records.\n";
            return 1;
        }

        total_records += records;
        if (!use_stdout && (total_records % 1000000ULL) < records)
            std::cerr << "Converted " << total_records << " records...\n";
    }

    std::cerr << "Done. Converted " << total_records
              << " records to: " << output_path << "\n";
    return 0;
}
