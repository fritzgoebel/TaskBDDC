#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <inttypes.h>
#include <omp.h>
#include <ompt.h>

static const char* ompt_thread_type_t_values[] = {
  NULL,
  "ompt_thread_initial",
  "ompt_thread_worker",
  "ompt_thread_other"
};

static const char* ompt_task_status_t_values[] = {
  NULL,
  "ompt_task_complete",
  "ompt_task_yield",
  "ompt_task_cancel",
  "ompt_task_others"
};
static const char* ompt_cancel_flag_t_values[] = {
  "ompt_cancel_parallel",
  "ompt_cancel_sections",
  "ompt_cancel_do",
  "ompt_cancel_taskgroup",
  "ompt_cancel_activated",
  "ompt_cancel_detected",
  "ompt_cancel_discarded_task"
};

static void format_task_type(int type, char *buffer) {
  char *progress = buffer;
  if (type & ompt_task_initial)
    progress += sprintf(progress, "ompt_task_initial");
  if (type & ompt_task_implicit)
    progress += sprintf(progress, "ompt_task_implicit");
  if (type & ompt_task_explicit)
    progress += sprintf(progress, "ompt_task_explicit");
  if (type & ompt_task_target)
    progress += sprintf(progress, "ompt_task_target");
  if (type & ompt_task_undeferred)
    progress += sprintf(progress, "|ompt_task_undeferred");
  if (type & ompt_task_untied)
    progress += sprintf(progress, "|ompt_task_untied");
  if (type & ompt_task_final)
    progress += sprintf(progress, "|ompt_task_final");
  if (type & ompt_task_mergeable)
    progress += sprintf(progress, "|ompt_task_mergeable");
  if (type & ompt_task_merged)
    progress += sprintf(progress, "|ompt_task_merged");
}

static ompt_set_callback_t ompt_set_callback;
static ompt_get_callback_t ompt_get_callback;
static ompt_get_state_t ompt_get_state;
static ompt_get_task_info_t ompt_get_task_info;
static ompt_get_thread_data_t ompt_get_thread_data;
static ompt_get_parallel_info_t ompt_get_parallel_info;
static ompt_get_unique_id_t ompt_get_unique_id;
static ompt_get_num_procs_t ompt_get_num_procs;
static ompt_get_num_places_t ompt_get_num_places;
static ompt_get_place_proc_ids_t ompt_get_place_proc_ids;
static ompt_get_place_num_t ompt_get_place_num;
static ompt_get_partition_place_nums_t ompt_get_partition_place_nums;
static ompt_get_proc_id_t ompt_get_proc_id;
static ompt_enumerate_states_t ompt_enumerate_states;
static ompt_enumerate_mutex_impls_t ompt_enumerate_mutex_impls;

typedef struct {
    int size;
    int capacity;
    long unsigned int* ids;
    int* locations;
} bucket;

bucket* task_buckets=NULL;

typedef struct {
    long unsigned int task_id;
    long unsigned int parent_task_id;
    long unsigned int* in_dependencies;
    long unsigned int* out_dependencies;
    int is_wait;
} task_data_t;

task_data_t* tasks=NULL;

ompt_task_dependence_t* dep_list=NULL;
int* dep_offsets=NULL;
long unsigned int* task_ids=NULL;
long unsigned int* parent_ids=NULL;
int dep_index=0;
int dep_size=0;
int task_index=0;
int task_size=0;
bool inside_parallel = false;

static void print_ids(int level)
{
  int task_type, thread_num;
  omp_frame_t *frame;
  ompt_data_t *task_parallel_data;
  ompt_data_t *task_data;
  int exists_task = ompt_get_task_info(level, &task_type, &task_data, &frame,
                                       &task_parallel_data, &thread_num);
  char buffer[2048];
  format_task_type(task_type, buffer);
  /* if (frame) */
    /* printf("%" PRIu64 ": task level %d: parallel_id=%" PRIu64 */
    /*        ", task_id=%" PRIu64 ", exit_frame=%p, reenter_frame=%p, " */
    /*        "task_type=%s=%d, thread_num=%d\n", */
    /*        ompt_get_thread_data()->value, level, */
    /*        exists_task ? task_parallel_data->value : 0, */
    /*        exists_task ? task_data->value : 0, frame->exit_frame, */
    /*        frame->enter_frame, buffer, task_type, thread_num); */
}

#define get_frame_address(level) __builtin_frame_address(level)

#define print_frame(level)                                                     \
  printf("%" PRIu64 ": __builtin_frame_address(%d)=%p\n",                      \
         ompt_get_thread_data()->value, level, get_frame_address(level))

// clang (version 5.0 and above) adds an intermediate function call with debug flag (-g)
#if defined(TEST_NEED_PRINT_FRAME_FROM_OUTLINED_FN) || true
  #if defined(DEBUG) && defined(__clang__) && __clang_major__ >= 5
    #define print_frame_from_outlined_fn(level) print_frame(level+1)
  #else
    #define print_frame_from_outlined_fn(level) print_frame(level)
  #endif

  #if defined(__clang__) && __clang_major__ >= 5
    #warning "Clang 5.0 and later add an additional wrapper for outlined functions when compiling with debug information."
    #warning "Please define -DDEBUG iff you manually pass in -g to make the tests succeed!"
  #endif
#endif

// This macro helps to define a label at the current position that can be used
// to get the current address in the code.
//
// For print_current_address():
//   To reliably determine the offset between the address of the label and the
//   actual return address, we insert a NOP instruction as a jump target as the
//   compiler would otherwise insert an instruction that we can't control. The
//   instruction length is target dependent and is explained below.
//
// (The empty block between "#pragma omp ..." and the __asm__ statement is a
// workaround for a bug in the Intel Compiler.)
#define define_ompt_label(id) \
  {} \
  __asm__("nop"); \
ompt_label_##id:

// This macro helps to get the address of a label that is inserted by the above
// macro define_ompt_label(). The address is obtained with a GNU extension
// (&&label) that has been tested with gcc, clang and icc.
#define get_ompt_label_address(id) (&& ompt_label_##id)

// This macro prints the exact address that a previously called runtime function
// returns to.
#define print_current_address(id) \
  define_ompt_label(id) \
  print_possible_return_addresses(get_ompt_label_address(id))

// On X86 the NOP instruction is 1 byte long. In addition, the comiler inserts
// a MOV instruction for non-void runtime functions which is 3 bytes long.
#define print_possible_return_addresses(addr) \
  printf("%" PRIu64 ": current_address=%p or %p for non-void functions\n", \
         ompt_get_thread_data()->value, ((char *)addr) - 1, ((char *)addr) - 4)


// This macro performs a somewhat similar job to print_current_address(), except
// that it discards a certain number of nibbles from the address and only prints
// the most significant bits / nibbles. This can be used for cases where the
// return address can only be approximated.
//
// To account for overflows (ie the most significant bits / nibbles have just
// changed as we are a few bytes above the relevant power of two) the addresses
// of the "current" and of the "previous block" are printed.
#define print_fuzzy_address(id) \
  define_ompt_label(id) \
  print_fuzzy_address_blocks(get_ompt_label_address(id))

// If you change this define you need to adapt all capture patterns in the tests
// to include or discard the new number of nibbles!
#define FUZZY_ADDRESS_DISCARD_NIBBLES 2
#define FUZZY_ADDRESS_DISCARD_BYTES (1 << ((FUZZY_ADDRESS_DISCARD_NIBBLES) * 4))
#define print_fuzzy_address_blocks(addr) \
  printf("%" PRIu64 ": fuzzy_address=0x%" PRIx64 " or 0x%" PRIx64 " (%p)\n", \
  ompt_get_thread_data()->value, \
  ((uint64_t)addr) / FUZZY_ADDRESS_DISCARD_BYTES - 1, \
  ((uint64_t)addr) / FUZZY_ADDRESS_DISCARD_BYTES, addr)

static void
on_ompt_callback_mutex_acquire(
  ompt_mutex_kind_t kind,
  unsigned int hint,
  unsigned int impl,
  omp_wait_id_t wait_id,
  const void *codeptr_ra)
{
  switch(kind)
  {
    case ompt_mutex_lock:
      /* printf("%" PRIu64 ": ompt_event_wait_lock: wait_id=%" PRIu64 ", hint=%" PRIu32 ", impl=%" PRIu32 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, hint, impl, codeptr_ra); */
      break;
    case ompt_mutex_nest_lock:
      /* printf("%" PRIu64 ": ompt_event_wait_nest_lock: wait_id=%" PRIu64 ", hint=%" PRIu32 ", impl=%" PRIu32 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, hint, impl, codeptr_ra); */
      break;
    case ompt_mutex_critical:
      /* printf("%" PRIu64 ": ompt_event_wait_critical: wait_id=%" PRIu64 ", hint=%" PRIu32 ", impl=%" PRIu32 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, hint, impl, codeptr_ra); */
      break;
    case ompt_mutex_atomic:
      /* printf("%" PRIu64 ": ompt_event_wait_atomic: wait_id=%" PRIu64 ", hint=%" PRIu32 ", impl=%" PRIu32 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, hint, impl, codeptr_ra); */
      break;
    case ompt_mutex_ordered:
      /* printf("%" PRIu64 ": ompt_event_wait_ordered: wait_id=%" PRIu64 ", hint=%" PRIu32 ", impl=%" PRIu32 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, hint, impl, codeptr_ra); */
      break;
    default:
      break;
  }
}

static void
on_ompt_callback_mutex_acquired(
  ompt_mutex_kind_t kind,
  omp_wait_id_t wait_id,
  const void *codeptr_ra)
{
  switch(kind)
  {
    case ompt_mutex_lock:
      /* printf("%" PRIu64 ": ompt_event_acquired_lock: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra); */
      break;
    case ompt_mutex_nest_lock:
      /* printf("%" PRIu64 ": ompt_event_acquired_nest_lock_first: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra); */
      break;
    case ompt_mutex_critical:
      /* printf("%" PRIu64 ": ompt_event_acquired_critical: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra); */
      break;
    case ompt_mutex_atomic:
      /* printf("%" PRIu64 ": ompt_event_acquired_atomic: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra); */
      break;
    case ompt_mutex_ordered:
      /* printf("%" PRIu64 ": ompt_event_acquired_ordered: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra); */
      break;
    default:
      break;
  }
}

static void
on_ompt_callback_mutex_released(
  ompt_mutex_kind_t kind,
  omp_wait_id_t wait_id,
  const void *codeptr_ra)
{
  switch(kind)
  {
    case ompt_mutex_lock:
      /* printf("%" PRIu64 ": ompt_event_release_lock: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra); */
      break;
    case ompt_mutex_nest_lock:
      /* printf("%" PRIu64 ": ompt_event_release_nest_lock_last: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra); */
      break;
    case ompt_mutex_critical:
      /* printf("%" PRIu64 ": ompt_event_release_critical: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra); */
      break;
    case ompt_mutex_atomic:
      /* printf("%" PRIu64 ": ompt_event_release_atomic: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra); */
      break;
    case ompt_mutex_ordered:
      /* printf("%" PRIu64 ": ompt_event_release_ordered: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra); */
      break;
    default:
      break;
  }
}

static void
on_ompt_callback_nest_lock(
    ompt_scope_endpoint_t endpoint,
    omp_wait_id_t wait_id,
    const void *codeptr_ra)
{
  switch(endpoint)
  {
    case ompt_scope_begin:
      /* printf("%" PRIu64 ": ompt_event_acquired_nest_lock_next: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra); */
      break;
    case ompt_scope_end:
      /* printf("%" PRIu64 ": ompt_event_release_nest_lock_prev: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra); */
      break;
  }
}

static void
on_ompt_callback_sync_region(
  ompt_sync_region_kind_t kind,
  ompt_scope_endpoint_t endpoint,
  ompt_data_t *parallel_data,
  ompt_data_t *task_data,
  const void *codeptr_ra)
{
  switch(endpoint)
  {
    case ompt_scope_begin:
      switch(kind)
      {
        case ompt_sync_region_barrier:
          /* printf("%" PRIu64 ": ompt_event_barrier_begin: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra); */
          print_ids(0);
          break;
        case ompt_sync_region_taskwait:
          printf("%" PRIu64 ": ompt_event_taskwait_begin: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra);
          break;
        case ompt_sync_region_taskgroup:
          printf("%" PRIu64 ": ompt_event_taskgroup_begin: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra);
          break;
      }
      break;
    case ompt_scope_end:
      switch(kind)
      {
        case ompt_sync_region_barrier:
          printf("%" PRIu64 ": ompt_event_barrier_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, (parallel_data)?parallel_data->value:0, task_data->value, codeptr_ra);
          break;
        case ompt_sync_region_taskwait:
          printf("%" PRIu64 ": ompt_event_taskwait_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, (parallel_data)?parallel_data->value:0, task_data->value, codeptr_ra);
          break;
        case ompt_sync_region_taskgroup:
          printf("%" PRIu64 ": ompt_event_taskgroup_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, (parallel_data)?parallel_data->value:0, task_data->value, codeptr_ra);
          break;
      }
      break;
  }
}

static void
on_ompt_callback_sync_region_wait(
  ompt_sync_region_kind_t kind,
  ompt_scope_endpoint_t endpoint,
  ompt_data_t *parallel_data,
  ompt_data_t *task_data,
  const void *codeptr_ra)
{
  if (!inside_parallel)
    return;
  switch(endpoint)
  {
    case ompt_scope_begin:
      switch(kind)
      {
        case ompt_sync_region_barrier:
          printf("%" PRIu64 ": ompt_event_wait_barrier_begin: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra);
          break;
        case ompt_sync_region_taskwait:
#pragma omp critical
          {
          if (dep_index == dep_size) {
            dep_size *= 2;
            dep_list = (ompt_task_dependence_t *)realloc(dep_list, dep_size * sizeof(ompt_task_dependence_t));
          }
          // NOTE: assuming the only dependencies we actually get are 2 and 3
          dep_list[dep_index].dependence_flags = 1;
          dep_index++;
          if (task_index == task_size - 1) {
            task_size *= 2;
            dep_offsets = (int *)realloc(dep_offsets, task_size * sizeof(int));
            tasks = (task_data_t *)realloc(tasks, task_size * sizeof(task_data_t));
          }
          dep_offsets[task_index + 1] = dep_offsets[task_index] + 1;

          // Create data for the wait
          task_data_t new_task;
          new_task.task_id = ompt_get_unique_id();
          new_task.parent_task_id = task_data->value;
          new_task.in_dependencies = NULL;
          new_task.out_dependencies = NULL;
          new_task.is_wait = 1;
          tasks[task_index] = new_task;

          // Put wait into bucket
          int bucket_idx = new_task.task_id % 1024;
          if (task_buckets[bucket_idx].size == task_buckets[bucket_idx].capacity) {
            task_buckets[bucket_idx].capacity *= 2;
            task_buckets[bucket_idx].ids = (uint64_t *)realloc(task_buckets[bucket_idx].ids, task_buckets[bucket_idx].capacity * sizeof(uint64_t));
            task_buckets[bucket_idx].locations = (int *)realloc(task_buckets[bucket_idx].locations, task_buckets[bucket_idx].capacity * sizeof(int));
          }
          int bucket_size = task_buckets[bucket_idx].size;
          task_buckets[bucket_idx].ids[bucket_size] = new_task.task_id;
          task_buckets[bucket_idx].locations[bucket_size] = task_index;
          task_buckets[bucket_idx].size++;

          task_index++;
          }
          printf("%" PRIu64 ": ompt_event_wait_taskwait_begin: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra);
          break;
        case ompt_sync_region_taskgroup:
          printf("%" PRIu64 ": ompt_event_wait_taskgroup_begin: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra);
          break;
      }
      break;
    case ompt_scope_end:
      switch(kind)
      {
        case ompt_sync_region_barrier:
          printf("%" PRIu64 ": ompt_event_wait_barrier_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, (parallel_data)?parallel_data->value:0, task_data->value, codeptr_ra);
          break;
        case ompt_sync_region_taskwait:
          printf("%" PRIu64 ": ompt_event_wait_taskwait_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, (parallel_data)?parallel_data->value:0, task_data->value, codeptr_ra);
          break;
        case ompt_sync_region_taskgroup:
          printf("%" PRIu64 ": ompt_event_wait_taskgroup_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, (parallel_data)?parallel_data->value:0, task_data->value, codeptr_ra);
          break;
      }
      break;
  }
}

static void
on_ompt_callback_flush(
    ompt_data_t *thread_data,
    const void *codeptr_ra)
{
  printf("%" PRIu64 ": ompt_event_flush: codeptr_ra=%p\n", thread_data->value, codeptr_ra);
}

static void
on_ompt_callback_cancel(
    ompt_data_t *task_data,
    int flags,
    const void *codeptr_ra)
{
  const char* first_flag_value;
  const char* second_flag_value;
  if(flags & ompt_cancel_parallel)
    first_flag_value = ompt_cancel_flag_t_values[0];
  else if(flags & ompt_cancel_sections)
    first_flag_value = ompt_cancel_flag_t_values[1];
  else if(flags & ompt_cancel_do)
    first_flag_value = ompt_cancel_flag_t_values[2];
  else if(flags & ompt_cancel_taskgroup)
    first_flag_value = ompt_cancel_flag_t_values[3];

  if(flags & ompt_cancel_activated)
    second_flag_value = ompt_cancel_flag_t_values[4];
  else if(flags & ompt_cancel_detected)
    second_flag_value = ompt_cancel_flag_t_values[5];
  else if(flags & ompt_cancel_discarded_task)
    second_flag_value = ompt_cancel_flag_t_values[6];

  printf("%" PRIu64 ": ompt_event_cancel: task_data=%" PRIu64 ", flags=%s|%s=%" PRIu32 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, task_data->value, first_flag_value, second_flag_value, flags,  codeptr_ra);
}

static void
on_ompt_callback_idle(
  ompt_scope_endpoint_t endpoint)
{
  switch(endpoint)
  {
    case ompt_scope_begin:
      printf("%" PRIu64 ": ompt_event_idle_begin:\n", ompt_get_thread_data()->value);
      break;
    case ompt_scope_end:
      printf("%" PRIu64 ": ompt_event_idle_end:\n", ompt_get_thread_data()->value);
      break;
  }
}

static void
on_ompt_callback_implicit_task(
    ompt_scope_endpoint_t endpoint,
    ompt_data_t *parallel_data,
    ompt_data_t *task_data,
    unsigned int team_size,
    unsigned int thread_num)
{
  switch(endpoint)
  {
    case ompt_scope_begin:
      if(task_data->ptr)
        printf("%s\n", "0: task_data initially not null");
      task_data->value = ompt_get_unique_id();
      printf("%" PRIu64 ": ompt_event_implicit_task_begin: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", team_size=%" PRIu32 ", thread_num=%" PRIu32 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, team_size, thread_num);
      break;
    case ompt_scope_end:
      printf("%" PRIu64 ": ompt_event_implicit_task_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", team_size=%" PRIu32 ", thread_num=%" PRIu32 "\n", ompt_get_thread_data()->value, (parallel_data)?parallel_data->value:0, task_data->value, team_size, thread_num);
      break;
  }
}

static void
on_ompt_callback_lock_init(
  ompt_mutex_kind_t kind,
  unsigned int hint,
  unsigned int impl,
  omp_wait_id_t wait_id,
  const void *codeptr_ra)
{
  switch(kind)
  {
    case ompt_mutex_lock:
      printf("%" PRIu64 ": ompt_event_init_lock: wait_id=%" PRIu64 ", hint=%" PRIu32 ", impl=%" PRIu32 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, hint, impl, codeptr_ra);
      break;
    case ompt_mutex_nest_lock:
      printf("%" PRIu64 ": ompt_event_init_nest_lock: wait_id=%" PRIu64 ", hint=%" PRIu32 ", impl=%" PRIu32 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, hint, impl, codeptr_ra);
      break;
    default:
      break;
  }
}

static void
on_ompt_callback_lock_destroy(
  ompt_mutex_kind_t kind,
  omp_wait_id_t wait_id,
  const void *codeptr_ra)
{
  switch(kind)
  {
    case ompt_mutex_lock:
      printf("%" PRIu64 ": ompt_event_destroy_lock: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra);
      break;
    case ompt_mutex_nest_lock:
      printf("%" PRIu64 ": ompt_event_destroy_nest_lock: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra);
      break;
    default:
      break;
  }
}

static void
on_ompt_callback_work(
  ompt_work_type_t wstype,
  ompt_scope_endpoint_t endpoint,
  ompt_data_t *parallel_data,
  ompt_data_t *task_data,
  uint64_t count,
  const void *codeptr_ra)
{
  switch(endpoint)
  {
    case ompt_scope_begin:
      switch(wstype)
      {
        case ompt_work_loop:
          printf("%" PRIu64 ": ompt_event_loop_begin: parallel_id=%" PRIu64 ", parent_task_id=%" PRIu64 ", codeptr_ra=%p, count=%" PRIu64 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra, count);
          break;
        case ompt_work_sections:
          printf("%" PRIu64 ": ompt_event_sections_begin: parallel_id=%" PRIu64 ", parent_task_id=%" PRIu64 ", codeptr_ra=%p, count=%" PRIu64 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra, count);
          break;
        case ompt_work_single_executor:
          printf("%" PRIu64 ": ompt_event_single_in_block_begin: parallel_id=%" PRIu64 ", parent_task_id=%" PRIu64 ", codeptr_ra=%p, count=%" PRIu64 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra, count);
          break;
        case ompt_work_single_other:
          printf("%" PRIu64 ": ompt_event_single_others_begin: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p, count=%" PRIu64 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra, count);
          break;
        case ompt_work_workshare:
          //impl
          break;
        case ompt_work_distribute:
          printf("%" PRIu64 ": ompt_event_distribute_begin: parallel_id=%" PRIu64 ", parent_task_id=%" PRIu64 ", codeptr_ra=%p, count=%" PRIu64 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra, count);
          break;
        case ompt_work_taskloop:
          //impl
          printf("%" PRIu64 ": ompt_event_taskloop_begin: parallel_id=%" PRIu64 ", parent_task_id=%" PRIu64 ", codeptr_ra=%p, count=%" PRIu64 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra, count);
          break;
      }
      break;
    case ompt_scope_end:
      switch(wstype)
      {
        case ompt_work_loop:
          printf("%" PRIu64 ": ompt_event_loop_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p, count=%" PRIu64 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra, count);
          break;
        case ompt_work_sections:
          printf("%" PRIu64 ": ompt_event_sections_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p, count=%" PRIu64 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra, count);
          break;
        case ompt_work_single_executor:
          printf("%" PRIu64 ": ompt_event_single_in_block_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p, count=%" PRIu64 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra, count);
          break;
        case ompt_work_single_other:
          printf("%" PRIu64 ": ompt_event_single_others_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p, count=%" PRIu64 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra, count);
          break;
        case ompt_work_workshare:
          //impl
          break;
        case ompt_work_distribute:
          printf("%" PRIu64 ": ompt_event_distribute_end: parallel_id=%" PRIu64 ", parent_task_id=%" PRIu64 ", codeptr_ra=%p, count=%" PRIu64 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra, count);
          break;
        case ompt_work_taskloop:
          //impl
          printf("%" PRIu64 ": ompt_event_taskloop_end: parallel_id=%" PRIu64 ", parent_task_id=%" PRIu64 ", codeptr_ra=%p, count=%" PRIu64 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra, count);
          break;
      }
      break;
  }
}

static void
on_ompt_callback_master(
  ompt_scope_endpoint_t endpoint,
  ompt_data_t *parallel_data,
  ompt_data_t *task_data,
  const void *codeptr_ra)
{
  switch(endpoint)
  {
    case ompt_scope_begin:
      printf("%" PRIu64 ": ompt_event_master_begin: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra);
      break;
    case ompt_scope_end:
      printf("%" PRIu64 ": ompt_event_master_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra);
      break;
  }
}

static void
on_ompt_callback_parallel_begin(
  ompt_data_t *encountering_task_data,
  const omp_frame_t *encountering_task_frame,
  ompt_data_t* parallel_data,
  uint32_t requested_team_size,
  ompt_invoker_t invoker,
  const void *codeptr_ra)
{
  if(parallel_data->ptr)
    printf("0: parallel_data initially not null\n");
  parallel_data->value = ompt_get_unique_id();
  printf("%" PRIu64 ": ompt_event_parallel_begin: parent_task_id=%" PRIu64 ", parent_task_frame.exit=%p, parent_task_frame.reenter=%p, parallel_id=%" PRIu64 ", requested_team_size=%" PRIu32 ", codeptr_ra=%p, invoker=%d\n", ompt_get_thread_data()->value, encountering_task_data->value, encountering_task_frame->exit_frame, encountering_task_frame->enter_frame, parallel_data->value, requested_team_size, codeptr_ra, invoker);
  inside_parallel = true;
}

static void
on_ompt_callback_parallel_end(
  ompt_data_t *parallel_data,
  ompt_data_t *encountering_task_data,
  ompt_invoker_t invoker,
  const void *codeptr_ra)
{
  inside_parallel = false;
  printf("%" PRIu64 ": ompt_event_parallel_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", invoker=%d, codeptr_ra=%p\n", ompt_get_thread_data()->value, parallel_data->value, encountering_task_data->value, invoker, codeptr_ra);
}

static void
on_ompt_callback_task_create(
    ompt_data_t *encountering_task_data,
    const omp_frame_t *encountering_task_frame,
    ompt_data_t* new_task_data,
    int type,
    int has_dependences,
    const void *codeptr_ra)
{
  if(new_task_data->ptr)
    printf("0: new_task_data initially not null\n");
  new_task_data->value = ompt_get_unique_id();
  char buffer[2048];

  format_task_type(type, buffer);

  //there is no parallel_begin callback for implicit parallel region
  //thus it is initialized in initial task
  if(type & ompt_task_initial)
  {
    ompt_data_t *parallel_data;
    ompt_get_parallel_info(0, &parallel_data, NULL);
    if(parallel_data->ptr)
      printf("%s\n", "0: parallel_data initially not null");
    parallel_data->value = ompt_get_unique_id();
  }

  printf("%" PRIu64 ": ompt_event_task_create: parent_task_id=%" PRIu64 ", parent_task_frame.exit=%p, parent_task_frame.reenter=%p, new_task_id=%" PRIu64 ", codeptr_ra=%p, task_type=%s=%d, has_dependences=%s\n", ompt_get_thread_data()->value, encountering_task_data ? encountering_task_data->value : 0, encountering_task_frame ? encountering_task_frame->exit_frame : NULL, encountering_task_frame ? encountering_task_frame->enter_frame : NULL, new_task_data->value, codeptr_ra, buffer, type, has_dependences ? "yes" : "no");

#pragma omp critical
  {
  if (task_index == task_size - 1) {
    task_size *= 2;
    dep_offsets = (int *)realloc(dep_offsets, task_size * sizeof(int));
    task_ids = (uint64_t *)realloc(task_ids, task_size * sizeof(uint64_t));
    parent_ids = (uint64_t *)realloc(parent_ids, task_size * sizeof(uint64_t));
    tasks = (task_data_t *)realloc(tasks, task_size * sizeof(task_data_t));
  }
  dep_offsets[task_index + 1] = dep_offsets[task_index];
  task_ids[task_index + 1] = new_task_data->value;
  parent_ids[task_index + 1] = encountering_task_data ? encountering_task_data->value : 0;

  // Add new task to bucket
  int bucket_idx = task_index % 1024;
  if (task_buckets[bucket_idx].size == task_buckets[bucket_idx].capacity) {
    task_buckets[bucket_idx].capacity *= 2;
    task_buckets[bucket_idx].ids = (uint64_t *)realloc(task_buckets[bucket_idx].ids, task_buckets[bucket_idx].capacity * sizeof(uint64_t));
    task_buckets[bucket_idx].locations = (int *)realloc(task_buckets[bucket_idx].locations, task_buckets[bucket_idx].capacity * sizeof(int));
  }
  int bucket_size = task_buckets[bucket_idx].size;
  task_buckets[bucket_idx].ids[bucket_size] = new_task_data->value;
  task_buckets[bucket_idx].locations[bucket_size] = task_index;
  task_buckets[bucket_idx].size++;

  // Create data for new task
  task_data_t new_task;
  new_task.task_id = new_task_data->value;
  new_task.parent_task_id = encountering_task_data ? encountering_task_data->value : 0;
  new_task.in_dependencies = NULL;
  new_task.out_dependencies = NULL;
  new_task.is_wait = 0;
  tasks[task_index + 1] = new_task;
  task_index++;
  }
}

static void
on_ompt_callback_task_schedule(
    ompt_data_t *first_task_data,
    ompt_task_status_t prior_task_status,
    ompt_data_t *second_task_data)
{
  printf("%" PRIu64 ": ompt_event_task_schedule: first_task_id=%" PRIu64 ", second_task_id=%" PRIu64 ", prior_task_status=%s=%d\n", ompt_get_thread_data()->value, first_task_data->value, second_task_data->value, ompt_task_status_t_values[prior_task_status], prior_task_status);
  if(prior_task_status == ompt_task_complete)
  {
    printf("%" PRIu64 ": ompt_event_task_end: task_id=%" PRIu64 "\n", ompt_get_thread_data()->value, first_task_data->value);
  }
}

static void
on_ompt_callback_task_dependences(
  ompt_data_t *task_data,
  const ompt_task_dependence_t *deps,
  int ndeps)
{
  if (!inside_parallel)
    return;
  printf("%" PRIu64 ": ompt_event_task_dependences: task_id=%" PRIu64 ", deps=%p, ndeps=%d\n", ompt_get_thread_data()->value, task_data->value, (void *)deps, ndeps);
  /* if (task_index == task_size - 1) { */
  /*   task_size *= 2; */
  /*   dep_offsets = (int *)realloc(dep_offsets, task_size * sizeof(int)); */
  /* } */
  dep_offsets[task_index] += ndeps;
  /* task_index++; */

  for (int i = 0; i < ndeps; i++)
  {
    printf("%" PRIu64 ": ompt_event_task_dependence %d:, flags=%d, variable address=%p\n", ompt_get_thread_data()->value, i, deps[i].dependence_flags, deps[i].variable_addr);
#pragma omp critical
    {
    if (dep_index == dep_size) {
        dep_size *= 2;
        dep_list = (ompt_task_dependence_t *)realloc(dep_list, dep_size * sizeof(ompt_task_dependence_t));
    }
    dep_list[dep_index] = deps[i];
    dep_index++;
    }
  }
}

static void
on_ompt_callback_task_dependence(
  ompt_data_t *first_task_data,
  ompt_data_t *second_task_data)
{
  printf("%" PRIu64 ": ompt_event_task_dependence_pair: first_task_id=%" PRIu64 ", second_task_id=%" PRIu64 "\n", ompt_get_thread_data()->value, first_task_data->value, second_task_data->value);
}

static void
on_ompt_callback_thread_begin(
  ompt_thread_type_t thread_type,
  ompt_data_t *thread_data)
{
  if(thread_data->ptr)
    printf("%s\n", "0: thread_data initially not null");
  thread_data->value = ompt_get_unique_id();
  printf("%" PRIu64 ": ompt_event_thread_begin: thread_type=%s=%d, thread_id=%" PRIu64 "\n", ompt_get_thread_data()->value, ompt_thread_type_t_values[thread_type], thread_type, thread_data->value);
}

static void
on_ompt_callback_thread_end(
  ompt_data_t *thread_data)
{
  printf("%" PRIu64 ": ompt_event_thread_end: thread_id=%" PRIu64 "\n", ompt_get_thread_data()->value, thread_data->value);
}

static int
on_ompt_callback_control_tool(
  uint64_t command,
  uint64_t modifier,
  void *arg,
  const void *codeptr_ra)
{
  omp_frame_t* omptTaskFrame;
  ompt_get_task_info(0, NULL, (ompt_data_t**) NULL, &omptTaskFrame, NULL, NULL);
  printf("%" PRIu64 ": ompt_event_control_tool: command=%" PRIu64 ", modifier=%" PRIu64 ", arg=%p, codeptr_ra=%p, current_task_frame.exit=%p, current_task_frame.reenter=%p \n", ompt_get_thread_data()->value, command, modifier, arg, codeptr_ra, omptTaskFrame->exit_frame, omptTaskFrame->enter_frame);
  return 0; //success
}

#define register_callback_t(name, type)                       \
do{                                                           \
  type f_##name = &on_##name;                                 \
  if (ompt_set_callback(name, (ompt_callback_t)f_##name) ==   \
      ompt_set_never)                                         \
    printf("0: Could not register callback '" #name "'\n");   \
}while(0)

#define register_callback(name) register_callback_t(name, name##_t)

int ompt_initialize(
  ompt_function_lookup_t lookup,
  ompt_data_t *tool_data)
{
  ompt_set_callback = (ompt_set_callback_t) lookup("ompt_set_callback");
  ompt_get_callback = (ompt_get_callback_t) lookup("ompt_get_callback");
  ompt_get_state = (ompt_get_state_t) lookup("ompt_get_state");
  ompt_get_task_info = (ompt_get_task_info_t) lookup("ompt_get_task_info");
  ompt_get_thread_data = (ompt_get_thread_data_t) lookup("ompt_get_thread_data");
  ompt_get_parallel_info = (ompt_get_parallel_info_t) lookup("ompt_get_parallel_info");
  ompt_get_unique_id = (ompt_get_unique_id_t) lookup("ompt_get_unique_id");

  ompt_get_num_procs = (ompt_get_num_procs_t) lookup("ompt_get_num_procs");
  ompt_get_num_places = (ompt_get_num_places_t) lookup("ompt_get_num_places");
  ompt_get_place_proc_ids = (ompt_get_place_proc_ids_t) lookup("ompt_get_place_proc_ids");
  ompt_get_place_num = (ompt_get_place_num_t) lookup("ompt_get_place_num");
  ompt_get_partition_place_nums = (ompt_get_partition_place_nums_t) lookup("ompt_get_partition_place_nums");
  ompt_get_proc_id = (ompt_get_proc_id_t) lookup("ompt_get_proc_id");
  ompt_enumerate_states = (ompt_enumerate_states_t) lookup("ompt_enumerate_states");
  ompt_enumerate_mutex_impls = (ompt_enumerate_mutex_impls_t) lookup("ompt_enumerate_mutex_impls");

  register_callback(ompt_callback_mutex_acquire);
  register_callback_t(ompt_callback_mutex_acquired, ompt_callback_mutex_t);
  register_callback_t(ompt_callback_mutex_released, ompt_callback_mutex_t);
  register_callback(ompt_callback_nest_lock);
  register_callback(ompt_callback_sync_region);
  register_callback_t(ompt_callback_sync_region_wait, ompt_callback_sync_region_t);
  register_callback(ompt_callback_control_tool);
  register_callback(ompt_callback_flush);
  register_callback(ompt_callback_cancel);
  register_callback(ompt_callback_idle);
  register_callback(ompt_callback_implicit_task);
  register_callback_t(ompt_callback_lock_init, ompt_callback_mutex_acquire_t);
  register_callback_t(ompt_callback_lock_destroy, ompt_callback_mutex_t);
  register_callback(ompt_callback_work);
  register_callback(ompt_callback_master);
  register_callback(ompt_callback_parallel_begin);
  register_callback(ompt_callback_parallel_end);
  register_callback(ompt_callback_task_create);
  register_callback(ompt_callback_task_schedule);
  register_callback(ompt_callback_task_dependences);
  register_callback(ompt_callback_task_dependence);
  register_callback(ompt_callback_thread_begin);
  register_callback(ompt_callback_thread_end);
  dep_list = (ompt_task_dependence_t *) calloc(1024, sizeof(ompt_task_dependence_t));
  dep_size = 1024;
  dep_offsets = (int *) calloc(1024, sizeof(int));
  task_ids = (uint64_t *) calloc(1024, sizeof(uint64_t));
  parent_ids = (uint64_t *) calloc(1024, sizeof(uint64_t));
  task_size = 1024;

  task_buckets = (bucket*) calloc(1024, sizeof(bucket));
  for (int i = 0; i < 1024; i++) {
    task_buckets[i].size = 0;
    task_buckets[i].capacity = 8;
    task_buckets[i].ids = (uint64_t*) calloc(8, sizeof(uint64_t));
    task_buckets[i].locations = (int*) calloc(8, sizeof(int));
  }
  tasks = (task_data_t*) calloc(1024, sizeof(task_data_t));
  printf("0: NULL_POINTER=%p\n", (void*)NULL);
  return 1; //success
}

void ompt_finalize(ompt_data_t *tool_data)
{
  printf("0: ompt_event_runtime_shutdown\n");
  /* printf("Tasks: %d, Dependencies: %d\n", task_index, dep_index); */
  for (int i = 0; i < dep_index; i++)
  {
    /* printf("ompt_event_task_dependence %d:, flags=%d, variable address=%p\n", i, dep_list[i].dependence_flags, dep_list[i].variable_addr); */
  }
  char* outfile = "taskgraph.dot";
  FILE *fp = fopen(outfile, "w");
  fprintf(fp, "digraph task_dependencies {\n");
  fprintf(fp, "  overlap = false;\n");
  fprintf(fp, "  splines = true;\n");
  fprintf(fp, "  node [shape=box];\n");
  fprintf(fp, "  start [shape=diamond];\n");
  fprintf(fp, "  end [shape=diamond];\n");
  bool outgoing[task_index];
  bool incoming[task_index];
  int wait_numbers[task_index];
  int waits = -1;
  for (int i = 0; i < task_index; i++)
  {
    outgoing[i] = false;
    incoming[i] = false;
    wait_numbers[i] = -1;
  }
  for (int i = 0; i < task_index; i++)
  {
    // If flag was set to 1 (taskwait), create a taskwait node with incoming edges from all previous tasks without outgoing edges
    if (dep_offsets[i] + 1 == dep_offsets[i+1] && dep_list[dep_offsets[i]].dependence_flags == 1)
    {
      waits++;
      fprintf(fp, "  taskwait_%d [shape=ellipse];\n", waits);
      for (int j = i-1; j >= 0; j--)
      {
        if (!outgoing[j])
        {
          fprintf(fp, "  task_%d_%" PRIu64 " -> taskwait_%d;\n", j, parent_ids[j], waits);
          outgoing[j] = true;
        }
      }
      incoming[i] = true;
      // Define outgoing to be true for taskwait such as not to create an extra task node for it in the end.
      outgoing[i] = true;
    } else {
      // Create node for each task, necessary for tasks without dependencies
      fprintf(fp, "  task_%d_%" PRIu64 ";\n", i, parent_ids[i]);
    }
    wait_numbers[i] = waits;
    for (int j = dep_offsets[i]; j < dep_offsets[i + 1]; j++)
    {
      if (dep_list[j].dependence_flags == ompt_task_dependence_type_in)
      {
        /* printf("%d, %d\n", i, j); */
        bool found_in = false;
        bool found_out = false;
        for (int k = i - 1; k >= 0; k--)
        {
          for (int l = dep_offsets[k]; l < dep_offsets[k + 1]; l++)
          {
            if ((!found_in || !found_out) && dep_list[l].dependence_flags == ompt_task_dependence_type_inout && dep_list[l].variable_addr == dep_list[j].variable_addr)
            {
              if (wait_numbers[k] != -1 && wait_numbers[k] < waits) {
                found_out = true;
                found_in = true;
                break;
              }
              /* printf("%d, %d, %d, %d\n", i, j, k, l); */
              fprintf(fp, "  task_%d_%" PRIu64 " -> task_%d_%" PRIu64 ";\n", k, parent_ids[k], i, parent_ids[i]);
              found_out = true;
              outgoing[k] = true;
              incoming[i] = true;
            }
          }
          for (int l = dep_offsets[k]; l < dep_offsets[k + 1]; l++)
          {
            if (found_out && dep_list[l].dependence_flags == ompt_task_dependence_type_in && dep_list[l].variable_addr == dep_list[j].variable_addr)
            {
              /* printf("%d, %d, %d, %d\n", i, j, k, l); */
              found_in = true;
            }
          }
        }
      }
    }
  }
  waits = -1;
  while (waits < wait_numbers[0]) {
    waits++;
    fprintf(fp, "  start -> taskwait_%d;\n", waits);
  }
  for (int i = 0; i < task_index; i++)
  {
    if (dep_list[dep_offsets[i]].dependence_flags == 1) {
      if (dep_list[dep_offsets[i] + 1].dependence_flags == 1) {
        fprintf(fp, "  taskwait_%d -> taskwait_%d;\n", wait_numbers[i], wait_numbers[i + 1]);
      }
    }
    if (!incoming[i])
    {
      if (wait_numbers[i] == -1)
      {
        fprintf(fp, "  start -> task_%d_%" PRIu64 ";\n", i, parent_ids[i]);
      } else {
        fprintf(fp, "  taskwait_%d -> task_%d_%" PRIu64 ";\n", wait_numbers[i], i, parent_ids[i]);
      }
    }
    if (!outgoing[i])
    {
      fprintf(fp, "  task_%d_%" PRIu64 " -> end;\n", i, parent_ids[i]);
    }
  }
  if (dep_list[dep_offsets[task_index - 1]].dependence_flags == 1)
  {
    fprintf(fp, "  taskwait_%d -> end;\n", wait_numbers[task_index - 1]);
  }
  fprintf(fp, "}\n");
  fclose(fp);

  for (int i = 0; i < 1024; i++) {
    if (task_buckets[i].size == 0)
      continue;
    printf("Bucket %d: %d / %d\n", i, task_buckets[i].size, task_buckets[i].capacity);
    for (int j = 0; j < task_buckets[i].size; j++) {
      if (tasks[task_buckets[i].locations[j]].is_wait == 1)
        printf("  %" PRIu64 " at %d, parent: %" PRIu64 " (wait)\n", task_buckets[i].ids[j], task_buckets[i].locations[j], tasks[task_buckets[i].locations[j]].parent_task_id);
      else
        printf("  %" PRIu64 " at %d, parent: %" PRIu64 "\n", task_buckets[i].ids[j], task_buckets[i].locations[j], tasks[task_buckets[i].locations[j]].parent_task_id);
    }
  }
  free(dep_list);
  free(dep_offsets);
  free(task_ids);
  free(parent_ids);
  free(task_buckets);
  free(tasks);
}

ompt_start_tool_result_t* ompt_start_tool(
  unsigned int omp_version,
  const char *runtime_version)
{
  static ompt_start_tool_result_t ompt_start_tool_result = {&ompt_initialize,&ompt_finalize, 0};
  return &ompt_start_tool_result;
}
