#pragma once

#include "app/Ports.hpp"

namespace gigaam::infra::datasets {

class RussianLibrispeechIndex : public app::IDatasetIndex {
public:
    explicit RussianLibrispeechIndex(const app::IHttpClient &http_client);

    std::vector<domain::DatasetSample> ListSamples(size_t limit) const override;

private:
    const app::IHttpClient &http_client_;
};

}  // namespace gigaam::infra::datasets
