#define BEGIN_SINGLE_CORE if (pi_core_id() == 0) {
#define END_SINGLE_CORE }
#define SINGLE_CORE if (pi_core_id() == 0)