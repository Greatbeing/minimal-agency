# -*- coding: utf-8 -*-
"""
BreakShell CLI — 命令行工具
"""

import argparse
import sys
import os

# 添加包路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser(
        description='BreakShell — AI Agent 自我模型安全层',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  breakshell train --env capability --episodes 500 --output my_agent
  breakshell evaluate --model my_agent --episodes 100
  breakshell compare --env capability --episodes 500
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # train
    train_parser = subparsers.add_parser('train', help='训练 Agent')
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
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.command == 'train':
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


if __name__ == '__main__':
    main()
