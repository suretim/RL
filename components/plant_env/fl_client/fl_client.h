#pragma once
#include <vector>
#include <string>

bool fetch_seq_from_server(std::vector<std::vector<float>>& seq_input, const std::string& url);
