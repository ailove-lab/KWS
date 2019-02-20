
#include <fstream>
#include <iomanip>
#include <unordered_set>
#include <vector>

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

#define NRM  "\x1B[0m"
#define RED  "\x1B[31m"
#define GRN  "\x1B[32m"
#define YEL  "\x1B[33m"
#define BLU  "\x1B[34m"
#define MAG  "\x1B[35m"
#define CYN  "\x1B[36m"
#define WHT  "\x1B[37m"

 // These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::int32;
using tensorflow::int64;
using tensorflow::string;
using tensorflow::uint16;
using tensorflow::uint32;

string wav    = "";
string graph  = "";
string labels = "";

string input_data_name  = "decoded_sample_data:0";
string input_rate_name  = "decoded_sample_data:1";
string output_name      = "labels_softmax";
int32 clip_duration_ms  =  500;
int32 clip_stride_ms    =   20;
float detection_threshold = 0.7f;

namespace {

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
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

}  // namespace


std::vector<string> labels_list;
std::unique_ptr<tensorflow::Session> session;
  
int apply_model() {


  // decode wav
  string wav_string;
  Status read_wav_status = tensorflow::ReadFileToString(
      tensorflow::Env::Default(), wav, &wav_string);
  if (!read_wav_status.ok()) {
    LOG(ERROR) << read_wav_status;
    return -1;
  }
  std::vector<float> audio_data;
  uint32 sample_count;
  uint16 channel_count;
  uint32 sample_rate;
  Status decode_wav_status = tensorflow::wav::DecodeLin16WaveAsFloatVector(
      wav_string, &audio_data, &sample_count, &channel_count, &sample_rate);
  if (!decode_wav_status.ok()) {
    LOG(ERROR) << decode_wav_status;
    return -1;
  }
  if (channel_count != 1) {
    LOG(ERROR) << "Only mono .wav files can be used, but input has "
               << channel_count << " channels.";
    return -1;
  }

  printf("%s\n", wav.c_str());
  printf("\tsamples:  %d\n", sample_count); 
  printf("\trate:     %d\n", sample_rate);
  printf("\tchannels: %d\n", channel_count);

  // setup run
  const int64 clip_duration_samples = (clip_duration_ms * sample_rate) / 1000;
  const int64 clip_stride_samples   = (clip_stride_ms   * sample_rate) / 1000;
  Tensor audio_data_tensor(tensorflow::DT_FLOAT,
                           tensorflow::TensorShape({clip_duration_samples, 1}));

  Tensor sample_rate_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
  sample_rate_tensor.scalar<int32>()() = sample_rate;

  const int64 audio_data_end = (sample_count - clip_duration_samples);
  for (int64 audio_data_offset = 0; audio_data_offset < audio_data_end;
       audio_data_offset += clip_stride_samples) {
    const float* input_start = &(audio_data[audio_data_offset]);
    const float* input_end = input_start + clip_duration_samples;
    std::copy(input_start, input_end, audio_data_tensor.flat<float>().data());

    // Actually run the audio through the model.
    std::vector<Tensor> outputs;
    Status run_status = session->Run({{input_data_name, audio_data_tensor},
                                      {input_rate_name, sample_rate_tensor}},
                                      {output_name}, {}, &outputs);
    if (!run_status.ok()) {
      LOG(ERROR) << "Running model failed: " << run_status;
      return -1;
    }

    const int64 current_time_ms = (audio_data_offset * 1000) / sample_rate;
    auto results = outputs[0].flat<float>();
    printf("%*d\t", 5, audio_data_offset);
    int id = 0;
    int r_max=0;
    for (int i = 0; i < results.size(); ++i) {
        int r = (int)(results(i)*100.0);
        if     (30 < r && r < 50) printf(CYN "%*d" NRM, 3, r);
        else if(50 < r && r < 70) printf(YEL "%*d" NRM, 3, r);
        else if(70 <r           ) printf(GRN "%*d" NRM, 3, r);
        else                      printf(BLU "%*d" NRM, 3, r);
        if(r>r_max) { id = i; r_max = r; }
    }
    printf(" %s\n", labels_list[id].c_str());
    
    // float score;
    // bool is_new_command;
    // Status recognize_status = recognize_commands.ProcessLatestResults(
    //     outputs[0], current_time_ms, &found_command, &score, &is_new_command);
    // if (!recognize_status.ok()) {
    //   LOG(ERROR) << "Recognition processing failed: " << recognize_status;
    //   return -1;
    // }

    // if (is_new_command && (found_command != "_silence_")) {
    // all_found_words.push_back({found_command, current_time_ms});
    // printf("%" PRId64 " sample %" PRId64 "ms:\t%s %f\n", audio_data_offset, current_time_ms, found_command.c_str(), score);
  }
  return 0;
}


int main(int argc, char* argv[]) {

  std::vector<Flag> flag_list = {
      Flag("wav"                , &wav                , "audio file to be identified"                          ),
      Flag("graph"              , &graph              , "model to be executed"                                 ),
      Flag("labels"             , &labels             , "path to file containing labels"                       ),
      Flag("input_data_name"    , &input_data_name    , "name of input data node in model"                     ),
      Flag("input_rate_name"    , &input_rate_name    , "name of input sample rate node in model"              ),
      Flag("output_name"        , &output_name        , "name of output node in model"                         ),
      Flag("clip_duration_ms"   , &clip_duration_ms   , "length of recognition window"                         ),
      Flag("clip_stride_ms"     , &clip_stride_ms     , "how often to run recognition"                         ),
      Flag("detection_threshold", &detection_threshold, "what score is required to trigger detection of a word"),
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

  // First we load and initialize the model.
  Status load_graph_status = LoadGraph(graph, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  Status read_labels_status = ReadLabelsFile(labels, &labels_list);
  if (!read_labels_status.ok()) {
    LOG(ERROR) << read_labels_status;
    return -1;
  }

  return apply_model();

}
