
#include <fstream>
#include <iomanip>
#include <map>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <dirent.h>
#include <inttypes.h>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/wav/wav_io.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

// These are all common classes it's handy to reference with no namespace.
using std::get;
using std::map;
using std::tuple;
using tensorflow::Flag;
using tensorflow::int32;
using tensorflow::int64;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::uint16;
using tensorflow::uint32;

string wav = "";
string graph = "";
string labels = "";

string input_data_name = "decoded_sample_data:0";
string input_rate_name = "decoded_sample_data:1";
string output_name = "labels_softmax";
int32 clip_duration_ms = 500;
int32 clip_stride_ms = 20;
float detection_threshold = 0.7f;

static std::vector<string> words = {"вредно",         "запрещено",  "исключено",
                                    "не_делай",       "не_надо",    "не_нужно",
                                    "не_следует",     "не_стоит",   "нельзя",
                                    "плохая_примета", "разве_можно"};

static std::vector<string> models = {"cnn",    "crnn", "dnn",
                                     "ds_cnn", "gru",  "lstm"};

namespace {

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(),
                                               graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_file_name, "'");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
}

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings.
Status ReadLabelsFile(const string& file_name, std::vector<string>* result) {
    std::ifstream file(file_name);
    if (!file) {
        return tensorflow::errors::NotFound("Labels file '", file_name,
                                            "' not found.");
    }
    result->clear();
    string line;
    while (std::getline(file, line)) {
        result->push_back(line);
    }
    return Status::OK();
}

} // namespace

std::vector<string> labels_list;
std::unique_ptr<tensorflow::Session> session;

map<string, int> apply_model() {

    map<string, int> result;

    // decode wav
    string wav_string;
    Status read_wav_status = tensorflow::ReadFileToString(
        tensorflow::Env::Default(), wav, &wav_string);
    if (!read_wav_status.ok()) {
        LOG(ERROR) << read_wav_status;
        return result;
    }
    std::vector<float> audio_data;
    uint32 sample_count;
    uint16 channel_count;
    uint32 sample_rate;
    Status decode_wav_status = tensorflow::wav::DecodeLin16WaveAsFloatVector(
        wav_string, &audio_data, &sample_count, &channel_count, &sample_rate);
    if (!decode_wav_status.ok()) {
        LOG(ERROR) << decode_wav_status;
        return result;
    }
    if (channel_count != 1) {
        LOG(ERROR) << "Only mono .wav files can be used, but input has "
                   << channel_count << " channels.";
        return result;
    }

    // printf("%s\n", wav.c_str());
    // printf("\tsamples:  %d\n", sample_count);
    // printf("\trate:     %d\n", sample_rate);
    // printf("\tchannels: %d\n", channel_count);

    // setup run
    const int64 clip_duration_samples = (clip_duration_ms * sample_rate) / 1000;
    const int64 clip_stride_samples = (clip_stride_ms * sample_rate) / 1000;
    Tensor audio_data_tensor(
        tensorflow::DT_FLOAT,
        tensorflow::TensorShape({clip_duration_samples, 1}));

    Tensor sample_rate_tensor(tensorflow::DT_INT32,
                              tensorflow::TensorShape({}));
    sample_rate_tensor.scalar<int32>()() = sample_rate;

    const int64 audio_data_end = (sample_count - clip_duration_samples);
    for (int64 audio_data_offset = 0; audio_data_offset < audio_data_end;
         audio_data_offset += clip_stride_samples) {
        const float* input_start = &(audio_data[audio_data_offset]);
        const float* input_end = input_start + clip_duration_samples;
        std::copy(input_start, input_end,
                  audio_data_tensor.flat<float>().data());

        // Actually run the audio through the model.
        std::vector<Tensor> outputs;
        Status run_status =
            session->Run({{input_data_name, audio_data_tensor},
                          {input_rate_name, sample_rate_tensor}},
                         {output_name}, {}, &outputs);
        if (!run_status.ok()) {
            LOG(ERROR) << "Running model failed: " << run_status;
            return result;
        }

        const int64 current_time_ms = (audio_data_offset * 1000) / sample_rate;
        auto results = outputs[0].flat<float>();
        int id = 0;
        int r_max = 0;
        for (int i = 0; i < results.size(); ++i) {
            int r = (int)(results(i) * 100.0);
            if (r > r_max) {
                id = i;
                r_max = r;
            }
        }
        result[labels_list[id]] += 1;
    }
    return result;
}

int run_validation() {

    std::cout << "Модель\tСлово\tИП ко-во\tИП сумма\tЛП ко-во\tЛП сумма\n";
    char buf[256];
    for (string& model : models) {

        sprintf(buf, "train/%1$s/%1$s_frozen.pb", model.c_str());
        graph = buf;
        Status load_graph_status = LoadGraph(graph, &session);
        if (!load_graph_status.ok()) {
            LOG(ERROR) << load_graph_status;
            return -1;
        }

        sprintf(buf, "train/%1$s/training/%1$s_labels.txt", model.c_str());
        labels = buf;
        Status read_labels_status = ReadLabelsFile(labels, &labels_list);
        if (!read_labels_status.ok()) {
            LOG(ERROR) << read_labels_status;
            return -1;
        }

        for (string& word : words) {
            tuple<int, int, int, int, int> stat;
            DIR* dir;
            struct dirent* ent;
            sprintf(buf, "data/validation/%s/", word.c_str());
            if ((dir = opendir(buf)) != NULL) {
                while ((ent = readdir(dir)) != NULL) {
                    if (ent->d_name[0] != '.') {
                        sprintf(buf, "data/validation/%s/%s", word.c_str(),
                                ent->d_name);
                        wav = buf;
                        map<string, int> res = apply_model();
                        if (res.find(word) != res.end()) {
                            get<0>(stat)++; // found word
                            get<1>(stat) += res[word];
                        } else {
                            get<2>(stat)++; // err
                        }

                        // count false positive
                        for (auto const& w : words) {
                            if (w != word) {
                                if (res.find(w) != res.end()) {
                                    get<3>(stat)++;
                                    get<4>(stat) += res[w];
                                }
                            }
                        }
                    }
                }
                closedir(dir);
            }

            std::cout << model << "\t" << word << "\t" << get<0>(stat) << "\t"
                      << get<1>(stat) << "\t" << get<2>(stat) << "\t"
                      << get<3>(stat) << "\t" << get<4>(stat);
            std::cout << "\n" << std::flush;
        }
    }

    return 0;
}

int main(int argc, char* argv[]) {

    std::vector<Flag> flag_list = {
        Flag("input_data_name", &input_data_name,
             "name of input data node in model"),
        Flag("input_rate_name", &input_rate_name,
             "name of input sample rate node in model"),
        Flag("output_name", &output_name, "name of output node in model"),
        Flag("clip_duration_ms", &clip_duration_ms,
             "length of recognition window"),
        Flag("clip_stride_ms", &clip_stride_ms, "how often to run recognition"),
        Flag("detection_threshold", &detection_threshold,
             "what score is required to trigger detection of a word"),
    };
    string usage = tensorflow::Flags::Usage(argv[0], flag_list);
    const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
    if (!parse_result) {
        LOG(ERROR) << usage;
        return -1;
    }

    // We need to call this to set up global state for TensorFlow.
    tensorflow::port::InitMain(argv[0], &argc, &argv);
    if (argc > 1) {
        LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
        return -1;
    }

    run_validation();
}
