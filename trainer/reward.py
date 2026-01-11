import numpy as np
from typing import Dict, Tuple, List, Optional, Any


class WeedRemovalRewardFunction:
    def __init__(self,
                 target_tool: str = 'scythe',
                 max_distance: float = 20.0,
                 time_penalty: float = 0.01,
                 stamina_penalty: float = 0.005,
                 use_range: float = 1.0):
        self.target_tool = target_tool
        self.max_distance = max_distance
        self.time_penalty = time_penalty
        self.stamina_penalty = stamina_penalty
        self.use_range = use_range

        self.all_tools = ['axe', 'pickaxe', 'hoe', 'watering_can', 'scythe']

        self.reward_weights = {
            'target_completion': 10.0,
            'distance_improvement': 0.2,
            'correct_direction': 0.3,
            'correct_distance': 0.5,
            'correct_tool': 0.3,
            'switched_to_correct': 0.3,
            'switched_to_wrong': -0.2,
            'wrong_tool_use': -0.5,
            'out_of_range_use': -0.3,
            'wrong_direction_use': -0.3,
        }

        self.reward_history = []
        self.last_reward_breakdown: Dict[str, Any] = {}

    def compute_reward(self,
                       current_state: Dict,
                       action: Dict,
                       next_state: Dict,
                       done: bool = False) -> Tuple[float, Dict]:
        total_reward = 0.0

        # 记录奖励的来源和数值
        reward_info = {
            'action_type': action.get('type', 'unknown'),
            'reward_components': {},
            'status': 'in_progress'
        }

        # 1. 时间惩罚
        time_penalty = self._compute_time_penalty()
        total_reward += time_penalty
        reward_info['reward_components']['time_penalty'] = time_penalty

        # 2. 体力惩罚
        stamina_penalty = self._compute_stamina_penalty(current_state, action, next_state)
        total_reward += stamina_penalty
        reward_info['reward_components']['stamina_penalty'] = stamina_penalty

        # 3. 移动奖励
        if action.get('type') == 'move':
            move_reward = self._compute_move_reward(current_state, action, next_state)
            total_reward += move_reward
            reward_info['reward_components']['move_reward'] = move_reward

        # 4. 工具奖励
        tool_reward = self._compute_tool_reward(current_state, action, next_state)
        total_reward += tool_reward
        reward_info['reward_components']['tool_reward'] = tool_reward

        # 5. 动作奖励
        if action.get('type') == 'use':
            use_reward = self._compute_use_reward(current_state, action, next_state)
            total_reward += use_reward
            reward_info['reward_components']['use_reward'] = use_reward

            if self._is_successful_weeding(current_state, action, next_state):
                target_reward = self.reward_weights['target_completion']
                total_reward += target_reward
                reward_info['reward_components']['target_completion'] = target_reward
                reward_info['status'] = 'success'

        # 6. episode 结束给一个较大奖励/失败惩罚
        if done:
            if reward_info['status'] == 'success':
                total_reward += 5.0
                reward_info['reward_components']['completion_bonus'] = 5.0
            else:
                total_reward -= 2.0
                reward_info['reward_components']['failure_penalty'] = -2.0
                reward_info['status'] = 'failure'

        reward_info['total_reward'] = total_reward
        self.reward_history.append({
            'total_reward': total_reward,
            'breakdown': reward_info['reward_components'].copy(),
            'action': action,
            'status': reward_info['status']
        })
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-1000:]

        self.last_reward_breakdown = reward_info.copy()
        return total_reward, reward_info

    def _compute_time_penalty(self) -> float:
        return -self.time_penalty

    def _compute_stamina_penalty(self,
                                 current_state: Dict,
                                 action: Dict,
                                 next_state: Dict) -> float:
        cur_stamina = current_state.get('stamina')
        next_stamina = next_state.get('stamina')
        if cur_stamina is None or next_stamina is None:
            return 0.0
        consumed = cur_stamina - next_stamina
        return -consumed * self.stamina_penalty if consumed > 0 else 0.0

    def _compute_move_reward(self,
                             current_state: Dict,
                             action: Dict,
                             next_state: Dict) -> float:
        reward = 0.0

        cur_pos = current_state.get('agent_position', (0.0, 0.0))
        next_pos = next_state.get('agent_position', cur_pos)
        target_pos = current_state.get('target_weed_position', cur_pos)

        cur_dist = abs(cur_pos[0] - target_pos[0]) + abs(cur_pos[1] - target_pos[1])
        next_dist = abs(next_pos[0] - target_pos[0]) + abs(next_pos[1] - target_pos[1])
        dist_change = cur_dist - next_dist   #距离变换

        moved = (next_pos != cur_pos)

        if moved:
            if dist_change > 0:
                reward += dist_change * self.reward_weights['distance_improvement']
                if next_dist <= 3:
                    reward += 0.3  

            elif dist_change < 0:
                reward -= 0.05   # 轻微惩罚即可

            # 3. 进入使用范围 大奖励（引导停下来准备 use）
            if next_dist <= self.use_range and cur_dist > self.use_range:
                reward += self.reward_weights.get('correct_distance', 0.5)

        else:
            # 移动失败（撞墙 / 有障碍物）
            reward -= 0.2  # 惩罚无效移动尝试

        if (current_state.get('equipped_tool') == self.target_tool and
            cur_dist <= self.use_range and
            moved):
            reward -= 0.3  

        # 5. 鼓励面向目标方向移动
        agent_dir = current_state.get('agent_direction', 0)
        correct_dir = self._calculate_correct_direction(cur_pos, target_pos)
        if moved and agent_dir == correct_dir:
            reward += 0.05   # 面向正确方向移动 → 小奖励

        return reward

    def _compute_tool_reward(self,
                             current_state: Dict,
                             action: Dict,
                             next_state: Dict) -> float:
        reward = 0.0
        cur_tool = current_state.get('equipped_tool')
        next_tool = next_state.get('equipped_tool', cur_tool)

        # 持有正确工具 持续奖励
        if cur_tool == self.target_tool:
            reward += self.reward_weights['correct_tool']

        if action.get('type') == 'switch_tool':
            new_tool = action.get('tool_id')
            if new_tool == self.target_tool:
                reward += self.reward_weights['switched_to_correct']
            else:
                reward += self.reward_weights['switched_to_wrong']
                if new_tool == cur_tool:          # 无意义切换
                    reward -= 0.1

            # 从正确工具切换走 惩罚
            if cur_tool == self.target_tool and new_tool != self.target_tool:
                reward -= 0.5

        return reward

    def _compute_use_reward(self,
                            current_state: Dict,
                            action: Dict,
                            next_state: Dict) -> float:
        reward = 0.0
        pos = current_state.get('agent_position', (0.0, 0.0))
        target = current_state.get('target_weed_position', (0.0, 0.0))
        direction = current_state.get('agent_direction', 0)
        tool = current_state.get('equipped_tool')

        dist = self._calculate_distance(pos, target)
        correct_dir = self._calculate_correct_direction(pos, target)

        has_tool = (tool == self.target_tool)
        in_range = (dist <= self.use_range)
        facing = (direction == correct_dir)

        # 小奖励鼓励满足每个条件
        if has_tool and in_range and facing:
            if next_state.get('weed_removed', False):      # 真正成功移除杂草
                reward += 20.0                              
            else:
                reward += 5.0                               

        # 2. 执行了 use 但任一条件不满足 → 重罚！（关键！！）
        else:
            if not has_tool:
                reward -= 3.0                                   # 错工具 use，血罚
            if not in_range:
                reward -= 2.0                                   # 不在范围 use，血罚
            if not facing:
                reward -= 1.5                                   # 朝向不对 use，罚

        # 额外：已经在范围内拿着正确工具却没对准朝向
        if has_tool and in_range and not facing:
            reward -= 3.0
        return reward

    def _is_successful_weeding(self,
                               current_state: Dict,
                               action: Dict,
                               next_state: Dict) -> bool:
        if action.get('type') != 'use':
            return False

        pos = current_state.get('agent_position', (0.0, 0.0))
        target = current_state.get('target_weed_position', (0.0, 0.0))
        direction = current_state.get('agent_direction', 0)
        tool = current_state.get('equipped_tool')

        dist = self._calculate_distance(pos, target)
        correct_dir = self._calculate_correct_direction(pos, target)

        return (tool == self.target_tool and
                dist <= self.use_range and
                direction == correct_dir and
                next_state.get('weed_removed', False))

    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def _calculate_correct_direction(self,
                                     agent_pos: Tuple[float, float],
                                     target_pos: Tuple[float, float]) -> int:
        dx = target_pos[0] - agent_pos[0]
        dy = target_pos[1] - agent_pos[1]
        
        if dx > 0:      return 1    
        if dx < 0:      return 3    
        if dy > 0:      return 2    
        if dy < 0:      return 0    
        return 0  # 重合时返回任意方向

    def get_reward_statistics(self, window_size: int = 100) -> Dict:
        if not self.reward_history:
            return {'average_reward': 0.0, 'std_reward': 0.0,
                    'success_rate': 0.0, 'recent_average': 0.0, 'total_episodes': 0}

        rewards = [h['total_reward'] for h in self.reward_history]
        recent = rewards[-window_size:]

        recent_hist = self.reward_history[-window_size:]
        success = sum(1 for h in recent_hist if h['status'] == 'success')

        return {
            'average_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'success_rate': success / len(recent_hist),
            'recent_average': float(np.mean(recent)),
            'total_episodes': len(self.reward_history)
        }


# ==================== 测试 ====================
def test_reward_functions():
    print("=== 测试除草任务奖励函数 ===\n")
    reward_fn = WeedRemovalRewardFunction()

    # 场景1：成功除草
    print("场景1: 正确使用镰刀除草")
    state1 = {
        'agent_position': (1.0, 0.0),
        'target_weed_position': (1.0, 1.0),
        'agent_direction': 2,
        'equipped_tool': 'scythe',
        'stamina': 100
    }
    action1 = {'type': 'use'}
    state2 = {
        'agent_position': (1.0, 0.0),
        'target_weed_position': (1.0, 1.0),
        'agent_direction': 2,
        'equipped_tool': 'scythe',
        'stamina': 95,
        'weed_removed': True
    }
    r, info = reward_fn.compute_reward(state1, action1, state2, done=True)
    print(f"奖励: {r:.3f}  状态: {info['status']}\n")

    # 场景2：错误工具
    print("场景2: 使用斧头尝试除草")
    state3 = {**state1, 'equipped_tool': 'axe'}
    state4 = {**state2, 'equipped_tool': 'axe', 'weed_removed': False}
    r, info = reward_fn.compute_reward(state3, {'type': 'use'}, state4)
    print(f"奖励: {r:.3f}  状态: {info['status']}\n")

    # 场景3：切换到正确工具
    print("场景3: 切换到镰刀")
    r, info = reward_fn.compute_reward(state3, {'type': 'switch_tool', 'tool_id': 'scythe'},
                                       {**state3, 'equipped_tool': 'scythe'})
    print(f"奖励: {r:.3f}\n")

    # 场景4：向杂草移动
    print("场景4: 向杂草移动")
    state7 = {
        'agent_position': (5.0, 5.0),
        'target_weed_position': (1.0, 1.0),
        'agent_direction': 3,
        'equipped_tool': 'scythe',
        'stamina': 100
    }
    state8 = {
        **state7,
        'agent_position': (4.0, 4.0),
        'stamina': 98
    }
    r, info = reward_fn.compute_reward(state7, {'type': 'move'}, state8)
    print(f"奖励: {r:.3f}\n")

    print("奖励统计:", reward_fn.get_reward_statistics())


if __name__ == "__main__":
    test_reward_functions()