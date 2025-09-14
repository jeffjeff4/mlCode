from collections import defaultdict
import copy


def allocate_tasks(tasks, models):
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
    # Define a canonical order for model sizes.
    size_order = {'small': 0, 'medium': 1, 'large': 2}

    # Create a copy of the models dictionary to track availability.
    # We use a defaultdict to handle cases where a model type is not available.
    available_models = defaultdict(int, copy.deepcopy(models))

    # Dictionaries to store the results.
    task_allocations = {}
    unallocated_tasks = []

    # Sort tasks to handle larger tasks first. This prevents a large task that
    # needs a large model from being "starved" of resources by smaller tasks.
    sorted_tasks = sorted(
        tasks,
        key=lambda x: size_order.get(x[1], -1),
        reverse=True
    )

    # Process each task one by one.
    for i, (task_type, task_size) in enumerate(sorted_tasks):
        allocated = False

        # Determine the minimum size required for the task.
        task_size_level = size_order.get(task_size, -1)

        # Iterate through model sizes from smallest to largest that can handle the task.
        for model_size in ['small', 'medium', 'large']:
            model_size_level = size_order.get(model_size)

            # Check if the model is large enough for the task.
            if model_size_level >= task_size_level:
                model_key = (task_type, model_size)

                # Check if this type and size of model is available.
                if available_models.get(model_key, 0) > 0:
                    # Allocate the task to this model.
                    available_models[model_key] -= 1
                    task_allocations[i] = model_key
                    allocated = True
                    break  # Move to the next task.

        # If no suitable model was found after checking all sizes.
        if not allocated:
            unallocated_tasks.append((task_type, task_size))

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

    print("Initial Tasks:")
    for i, task in enumerate(tasks):
        print(f"  Task {i}: {task}")

    print("\nAvailable Models:")
    for model, count in models.items():
        print(f"  {model}: {count}")

    # Perform the allocation.
    allocations, unallocated, remaining_models = allocate_tasks(tasks, models)

    print("\n--- Allocation Results ---")
    for task_id, model_assigned in allocations.items():
        # Map the task ID back to the original task definition for a clearer printout.
        original_task = tasks[task_id]
        print(f"Task {original_task} allocated to Model {model_assigned}")

    print("\n--- Unallocated Tasks ---")
    if unallocated:
        for task in unallocated:
            print(f"Could not allocate: {task}")
    else:
        print("All tasks were successfully allocated.")

    print("\n--- Remaining Models ---")
    for model, count in sorted(models.items()):
        # Use the returned `remaining_models` dictionary to get the final counts.
        remaining_count = remaining_models.get(model, 0)
        print(f"{model}: {remaining_count} remaining")
