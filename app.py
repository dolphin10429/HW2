from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

DISCOUNT_FACTOR = 0.95
MAX_ITERATIONS = 1000
CONVERGENCE_THRESHOLD = 1e-9
GOAL_REWARD = 10

ACTIONS = {
    '↑': (-1, 0),
    '↓': (1, 0),
    '←': (0, -1),
    '→': (0, 1)
}

# 檢查是否可走到終點的模擬函數
def simulate_path(policy, start, end, grid_size, obstacles):
    path = []
    visited = set()
    cur = start
    while cur not in visited:
        visited.add(cur)
        path.append(cur)
        if cur == end:
            return path
        a = policy[cur[0]][cur[1]]
        if a not in ACTIONS:
            break
        di, dj = ACTIONS[a]
        ni, nj = cur[0] + di, cur[1] + dj
        if not (0 <= ni < grid_size and 0 <= nj < grid_size) or (ni, nj) in obstacles:
            break
        cur = (ni, nj)
    return []

def value_iteration(grid_size, obstacles, start_point, end_point):
    V = np.zeros((grid_size, grid_size))
    policy = np.full((grid_size, grid_size), '', dtype=object)
    steps = []
    best_path = []

    for (i, j) in obstacles:
        V[i][j] = None
        policy[i][j] = '■'

    V[end_point[0]][end_point[1]] = GOAL_REWARD
    policy[start_point[0]][start_point[1]] = 'S'
    policy[end_point[0]][end_point[1]] = 'E'

    def is_valid(i, j):
        return 0 <= i < grid_size and 0 <= j < grid_size and (i, j) not in obstacles

    for _ in range(MAX_ITERATIONS):
        delta = 0
        new_V = V.copy()
        new_policy = policy.copy()
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) in obstacles or (i, j) == end_point:
                    continue
                best_value = float('-inf')
                best_action = ''
                for action, (di, dj) in ACTIONS.items():
                    ni, nj = i + di, j + dj
                    if not is_valid(ni, nj):
                        reward = -1
                        ni, nj = i, j
                    else:
                        reward = GOAL_REWARD if (ni, nj) == end_point else 0
                    next_value = 0 if V[ni][nj] is None else V[ni][nj]
                    value = reward + DISCOUNT_FACTOR * next_value
                    if value > best_value:
                        best_value = value
                        best_action = action
                new_V[i][j] = round(best_value, 2)
                new_policy[i][j] = best_action
                delta = max(delta, abs(V[i][j] - best_value))
        steps.append((new_V.copy(), new_policy.copy()))
        V = new_V
        policy = new_policy
        if delta < CONVERGENCE_THRESHOLD:
            break

    # 最終最佳路徑
    visited = set()
    cur = start_point
    while cur != end_point and cur not in visited:
        visited.add(cur)
        best_path.append(cur)
        a = policy[cur[0]][cur[1]]
        if a in ACTIONS:
            di, dj = ACTIONS[a]
            cur = (cur[0] + di, cur[1] + dj)
        else:
            break
    best_path.append(end_point)

    return steps, best_path

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        grid_size = int(request.form.get("n"))
        start_raw = request.form.get("start")
        end_raw = request.form.get("end")
        obstacles_data = request.form.get("obstacles")

        if not start_raw or not end_raw:
            return "請選擇起點與終點再提交。", 400

        start_point = tuple(map(int, start_raw.split(',')))
        end_point = tuple(map(int, end_raw.split(',')))
        obstacles = []
        if obstacles_data:
            obstacles = [tuple(map(int, obs.split(','))) for obs in obstacles_data.split()]

        steps, best_path = value_iteration(grid_size, obstacles, start_point, end_point)

        all_values = []
        all_policies = []
        all_paths = []

        for v, p in steps:
            value_grid = [[None if (i, j) in obstacles else round(v[i][j], 2)
                           for j in range(grid_size)] for i in range(grid_size)]
            policy_grid = [[p[i][j] for j in range(grid_size)] for i in range(grid_size)]
            all_values.append(value_grid)
            all_policies.append(policy_grid)
            all_paths.append(simulate_path(p, start_point, end_point, grid_size, obstacles))

        return render_template("result.html",
                               n=grid_size,
                               steps=len(steps),
                               value_matrices=all_values,
                               policy_matrices=all_policies,
                               all_paths=all_paths,
                               best_path=best_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
