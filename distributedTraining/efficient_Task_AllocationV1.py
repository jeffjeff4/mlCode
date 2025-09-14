from collections import defaultdict
import copy


def allocate_tasks(tasks, models, capacity_req):
    """
    Allocates tasks to models in the most efficient way.

    The function uses a greedy approach to find the smallest suitable model for
    each task, ensuring that larger models are reserved for larger tasks.

    Args:
        tasks (list): A list of tuples, where each tuple represents a task in
                      the format (task_type, task_size).
                      Example: [('text', 'small'), ('image', 'large')]
        models (dict): A dictionary representing the available models, where the
                       key is a tuple (model_type, model_size) and the value is
                       the number of available models.
                       Example: {('text', 'small'): 3, ('image', 'medium'): 2}

    Returns:
        tuple: A tuple containing three items:
               - A dictionary of allocations: {task_id: (model_type, model_size)}
               - A list of tasks that could not be allocated.
               - The updated models dictionary showing remaining counts.
    """

    # Create a copy of the models dictionary to track availability.
    # We use a defaultdict to handle cases where a model type is not available.

    available_models = defaultdict(int, copy.deepcopy(models))
    model_capa = []
    for key,val in available_models.items():
        model_type = key[0]
        model_size  = key[1]
        model_size_num = capacity_req.get(model_size, -1)

        tmp = [model_type, model_size, model_size_num]
        model_capa.append(tmp)

    model_capa = sorted(model_capa, key=lambda x:x[2])

    # Sort tasks to handle larger tasks first. This prevents a large task that
    # needs a large model from being "starved" of resources by smaller tasks.
    new_tasks = []
    len_tasks = len(tasks)
    for idx in range(len_tasks):
        task = tasks[idx]
        task_type = task[0]
        task_size = task[1]
        task_size_num = capacity_req.get(task_size, -1)
        tmp = [task_type, task_size, task_size_num, idx]
        new_tasks.append(tmp)

    sorted_tasks = sorted(
        new_tasks,
        key=lambda x: x[2],
        reverse=True
    )

    # Dictionaries to store the results.
    task_allocations = []
    unallocated_tasks = []

    for idx, task_tup in enumerate(sorted_tasks):
        task_type = task_tup[0]
        task_size = task_tup[1]
        task_size_num = task_tup[2]
        ori_idx = task_tup[3]
        allocated = False

        for model_tup in model_capa:
            model_type = model_tup[0]
            model_size = model_tup[1]
            model_size_num = model_tup[2]
            if model_type != task_type:
                continue

            if model_size_num < task_size_num:
                continue

            model_key = tuple([model_type, model_size])
            if available_models[model_key] < 1:
                continue

            allocated = True
            available_models[model_key] -= 1

            tmp = [task_type, task_size, model_type, model_size, ori_idx]
            task_allocations.append(tmp)

        if allocated == False:
            tmp = [task_type, task_size, ori_idx]
            unallocated_tasks.append(tmp)

    return task_allocations, unallocated_tasks, available_models


# --- Example Usage ---

if __name__ == "__main__":
    # Define the list of tasks.
    tasks = [
        ('text', 'small'),
        ('text', 'medium'),
        ('image', 'small'),
        ('image', 'large'),
        ('text', 'small'),
        ('image', 'medium'),
        ('text', 'large'),
        ('video', 'medium')  # Task that cannot be allocated.
    ]

    # Define the available models and their counts.
    models = {
        ('text', 'small'): 1,
        ('text', 'medium'): 1,
        ('text', 'large'): 1,
        ('image', 'medium'): 2,
        ('image', 'large'): 1,
    }

    capacity_req = defaultdict(int)
    capacity_req['small'] = 0
    capacity_req['medium'] = 1
    capacity_req['large'] = 2

    print("Initial Tasks:")
    for i, task in enumerate(tasks):
        print(f"  Task {i}: {task}")

    print("\nAvailable Models:")
    for model, count in models.items():
        print(f"  {model}: {count}")

    # Perform the allocation.
    allocations, unallocated, remaining_models = allocate_tasks(tasks, models, capacity_req)

    allocations = sorted(allocations, key=lambda x:x[-1])
    print("\n--- Allocation Results ---")
    for (task_type, task_size, model_type, model_size, ori_idx) in allocations:
        print(f"Task {ori_idx}, {task_type}, {task_size} allocated to Model {model_type}, {model_size}")

    print("\n--- Unallocated Tasks ---")
    if unallocated:
        unallocated = sorted(unallocated, key=lambda x: x[-1])
        for (task_type, task_size, ori_idx) in unallocated:
            print(f"Could not allocate: {ori_idx}, {task_type}, {task_size}")
    else:
        print("All tasks were successfully allocated.")

    print("\n--- Remaining Models ---")
    for model, count in sorted(models.items()):
        # Use the returned `remaining_models` dictionary to get the final counts.
        remaining_count = remaining_models.get(model, 0)
        print(f"{model}: {remaining_count} remaining")
