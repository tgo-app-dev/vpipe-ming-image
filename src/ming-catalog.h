#ifndef VPIPE_MING_CATALOG_H
#define VPIPE_MING_CATALOG_H

#include "stages/model-catalog.h"

#include <vector>

namespace ming {

// The checkpoints this plugin can load, for `model-fetch` and the model
// pickers. A row lives with the loader: offering a fetch the host
// cannot run ends in a stage going inert.
std::vector<vpipe::ModelCatalogEntry> catalog_entries();

}  // namespace ming

#endif
