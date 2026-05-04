#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cstdio>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

namespace {

constexpr int BOARD_SIZE = 100;
constexpr int FEATURE_SIZE = 300;

#pragma pack(push, 1)
struct GameRecord {
    int8_t side;             // 0 = WHITE, 1 = BLACK
    char board[BOARD_SIZE];  // '0', '1', '2', '3'
    int8_t outcome;          // -1 or 1
    int16_t score;           // search score (side-to-move perspective)
};

struct FeatureRecord {
    uint8_t us[FEATURE_SIZE];
    uint8_t them[FEATURE_SIZE];
    uint8_t target;          // 0 or 1
    int16_t score;           // search score (passed through)
};
#pragma pack(pop)

static_assert(sizeof(GameRecord) == 104, "GameRecord size mismatch");
static_assert(sizeof(FeatureRecord) == 603, "FeatureRecord size mismatch");

void encode_features(const GameRecord& src, FeatureRecord& dst) {
    std::memset(dst.us, 0, FEATURE_SIZE);
    std::memset(dst.them, 0, FEATURE_SIZE);
    dst.target = static_cast<uint8_t>((src.outcome + 1) / 2); // -1/1 -> 0/1
    dst.score = src.score; // pass through search score for blended loss

    for (int sq = 0; sq < BOARD_SIZE; ++sq) {
        const char cell = src.board[sq];
        if (cell == '1') { // White amazon
            if (src.side == 0) {
                dst.us[sq] = 1;
                dst.them[100 + sq] = 1;
            } else {
                dst.us[100 + sq] = 1;
                dst.them[sq] = 1;
            }
        } else if (cell == '2') { // Black amazon
            if (src.side == 1) {
                dst.us[sq] = 1;
                dst.them[100 + sq] = 1;
            } else {
                dst.us[100 + sq] = 1;
                dst.them[sq] = 1;
            }
        } else if (cell == '3') { // Arrow
            dst.us[200 + sq] = 1;
            dst.them[200 + sq] = 1;
        }
    }
}

void print_usage(const char* exe) {
    std::cout << "Usage:\n"
              << "  " << exe << " [input_bin] [output_bin] [--append]\n\n"
              << "Defaults:\n"
              << "  input_bin  = selfplay_data.bin\n"
              << "  output_bin = selfplay_features.bin\n"
              << "  output_bin = -  (write binary feature stream to stdout)\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string input_path = "selfplay_data.bin";
    std::string output_path = "selfplay_features.bin";
    bool append = false;

    if (argc >= 2) {
        std::string arg1 = argv[1];
        if (arg1 == "-h" || arg1 == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        input_path = arg1;
    }
    if (argc >= 3) {
        output_path = argv[2];
    }
    if (argc >= 4) {
        append = (std::string(argv[3]) == "--append");
    }

    std::ifstream in(input_path, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open input file: " << input_path << "\n";
        return 1;
    }

    const bool use_stdout = (output_path == "-");
    std::unique_ptr<std::ofstream> out_file;
    std::ostream* out = nullptr;

    if (use_stdout) {
#ifdef _WIN32
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        out = &std::cout;
    } else {
        std::ios::openmode out_mode = std::ios::binary | std::ios::out;
        out_mode |= append ? std::ios::app : std::ios::trunc;
        out_file = std::make_unique<std::ofstream>(output_path, out_mode);
        if (!(*out_file)) {
            std::cerr << "Failed to open output file: " << output_path << "\n";
            return 1;
        }
        out = out_file.get();
    }

    constexpr std::size_t CHUNK_RECORDS = 1 << 16; // 65536
    std::vector<GameRecord> input_buf(CHUNK_RECORDS);
    std::vector<FeatureRecord> output_buf(CHUNK_RECORDS);

    std::uint64_t total_records = 0;
    while (true) {
        in.read(reinterpret_cast<char*>(input_buf.data()),
                static_cast<std::streamsize>(input_buf.size() * sizeof(GameRecord)));
        const std::streamsize bytes = in.gcount();

        if (bytes <= 0) {
            break;
        }
        if (bytes % static_cast<std::streamsize>(sizeof(GameRecord)) != 0) {
            std::cerr << "Warning: trailing bytes not aligned to GameRecord, ignored.\n";
        }

        const std::size_t records = static_cast<std::size_t>(bytes / sizeof(GameRecord));
        for (std::size_t i = 0; i < records; ++i) {
            encode_features(input_buf[i], output_buf[i]);
        }

        out->write(reinterpret_cast<const char*>(output_buf.data()),
                   static_cast<std::streamsize>(records * sizeof(FeatureRecord)));
        if (!(*out)) {
            std::cerr << "Write failed while writing output records.\n";
            return 1;
        }

        total_records += records;
        if (!use_stdout && (total_records % 1000000ULL) < records) {
            std::cerr << "Converted " << total_records << " records...\n";
        }

        if (!in && !in.eof()) {
            std::cerr << "Read failed on input stream.\n";
            return 1;
        }
    }

    std::cerr << "Done. Converted " << total_records
              << " records to: " << output_path << "\n";
    return 0;
}
