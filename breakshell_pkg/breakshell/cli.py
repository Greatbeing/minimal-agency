# -*- coding: utf-8 -*-
"""
BreakShell CLI — Phase 1
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser(
        description='BreakShell — AI Agent 自我模型安全层',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  breakshell run "列出当前目录的所有文件" --provider mock
  breakshell train --env capability --episodes 500 --output my_agent
  breakshell evaluate --model my_agent --episodes 100
  breakshell compare --env capability --episodes 500
  breakshell session list
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # run (Phase 1: LLM Agent)
    run_parser = subparsers.add_parser('run', help='运行 LLM Agent')
    run_parser.add_argument('goal', type=str, help='任务目标')
    run_parser.add_argument('--provider', choices=['mock', 'profy', 'deepseek'], default='mock')
    run_parser.add_argument('--model', type=str, default='gpt-5.6-sol')
    run_parser.add_argument('--max-steps', type=int, default=10)
    run_parser.add_argument('--permission', choices=['read-only', 'workspace-write', 'network', 'system'], default='workspace-write')
    
    # train (RL Agent)
    train_parser = subparsers.add_parser('train', help='训练 RL Agent')
    train_parser.add_argument('--env', choices=['capability', 'energy', 'financial'], default='capability')
    train_parser.add_argument('--episodes', type=int, default=500)
    train_parser.add_argument('--output', type=str, default='my_agent')
    train_parser.add_argument('--lr', type=float, default=0.005)
    
    # evaluate
    eval_parser = subparsers.add_parser('evaluate', help='评估 Agent')
    eval_parser.add_argument('--model', type=str, required=True)
    eval_parser.add_argument('--env', choices=['capability', 'energy', 'financial'], default='capability')
    eval_parser.add_argument('--episodes', type=int, default=100)
    
    # compare
    compare_parser = subparsers.add_parser('compare', help='对比实验')
    compare_parser.add_argument('--env', choices=['capability', 'energy', 'financial'], default='capability')
    compare_parser.add_argument('--episodes', type=int, default=500)
    
    # session
    session_parser = subparsers.add_parser('session', help='会话管理')
    session_sub = session_parser.add_subparsers(dest='session_cmd')
    session_list = session_sub.add_parser('list', help='列出会话')
    session_list.add_argument('--limit', type=int, default=10)
    session_show = session_sub.add_parser('show', help='显示会话详情')
    session_show.add_argument('session_id', type=str)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.command == 'run':
        from breakshell import run_agent
        state = run_agent(args.goal, provider=args.provider, llm_model=args.model, max_steps=args.max_steps)
        print(f"\n{'='*50}")
        print(f"状态: {state.status}")
        print(f"步数: {state.step_count}")
        print(f"工具调用: {len(state.tool_calls)}")
        print(f"观察数: {len(state.observations)}")
        if state.error:
            print(f"错误: {state.error}")
        print(f"{'='*50}")
    
    elif args.command == 'train':
        from breakshell import BreakShell
        from breakshell.envs import CapabilityEnv, EnergyEnv, FinancialEnv
        
        envs = {'capability': CapabilityEnv, 'energy': EnergyEnv, 'financial': FinancialEnv}
        env = envs[args.env](seed=42)
        
        agent = BreakShell(action_dim=3, lr=args.lr)
        print(f"训练 BreakShell ({args.env}, {args.episodes} episodes)...")
        agent.train(env, num_episodes=args.episodes)
        agent.save(args.output)
        print(f"模型已保存: {args.output}.pt")
    
    elif args.command == 'evaluate':
        from breakshell import BreakShell
        from breakshell.envs import CapabilityEnv, EnergyEnv, FinancialEnv
        
        envs = {'capability': CapabilityEnv, 'energy': EnergyEnv, 'financial': FinancialEnv}
        env = envs[args.env](seed=999)
        
        agent = BreakShell(action_dim=3)
        agent.load(args.model)
        ret = agent.evaluate(env, num_episodes=args.episodes)
        print(f"平均奖励: {ret:+.4f}")
    
    elif args.command == 'compare':
        from breakshell import BreakShell, NormalAgent
        from breakshell.envs import CapabilityEnv, EnergyEnv, FinancialEnv
        
        envs = {'capability': CapabilityEnv, 'energy': EnergyEnv, 'financial': FinancialEnv}
        
        env_train_n = envs[args.env](seed=42)
        env_train_b = envs[args.env](seed=42)
        env_eval = envs[args.env](seed=999)
        
        normal = NormalAgent(obs_dim=4 if args.env != 'financial' else 5, action_dim=3)
        bs = BreakShell(action_dim=3)
        
        print(f"训练普通 Agent ({args.env}, {args.episodes} episodes)...")
        normal.train(env_train_n, num_episodes=args.episodes)
        
        print(f"训练 BreakShell ({args.env}, {args.episodes} episodes)...")
        bs.train(env_train_b, num_episodes=args.episodes)
        
        ne = normal.evaluate(env_eval)
        be = bs.evaluate(env_eval)
        
        print(f"\n结果:")
        print(f"  普通 Agent: {ne:+.4f}")
        print(f"  BreakShell: {be:+.4f}")
        print(f"  差异: {be-ne:+.4f}")
    
    elif args.command == 'session':
        from breakshell.llm_agent import SessionStore
        store = SessionStore()
        if args.session_cmd == 'list':
            sessions = store.list_sessions(limit=args.limit)
            for s in sessions:
                print(f"  {s['session_id']} | {s['created_at']} | {s['goal'][:50]}")
        elif args.session_cmd == 'show':
            data = store.load_session(args.session_id)
            if data:
                print(f"Session: {data['session_id']}")
                print(f"Goal: {data['goal']}")
                print(f"Events: {len(data['events'])}")
            else:
                print("会话不存在")


if __name__ == '__main__':
    main()
