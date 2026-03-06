#!/usr/bin/env python3
"""
Скрипт для запуска функциональных тестов по сценарию из scenario.json
"""

import json
import requests
import sys
import time
from pathlib import Path

BASE_URL = "http://localhost:5000"
TIMEOUT = 30


def wait_for_service(url, timeout=TIMEOUT):
    """Ждём, пока сервис станет доступен"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print(f"Service is ready after {time.time() - start:.1f}s")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    print(f"Service not ready after {timeout}s")
    return False


def run_scenario(scenario):
    """Запуск одного сценария"""
    name = scenario['name']
    url = f"{BASE_URL}{scenario['endpoint']}"
    method = scenario['method'].upper()

    try:
        if method == 'GET':
            response = requests.get(url, timeout=10)
        elif method == 'POST':
            response = requests.post(url, json=scenario.get('input', {}), timeout=10)
        else:
            return {'name': name, 'status': 'SKIP', 'reason': f'Unknown method: {method}'}

        result = {
            'name': name,
            'status_code': response.status_code,
            'expected_status': scenario['expected_status'],
            'passed': response.status_code == scenario['expected_status']
        }

        # Проверка полей ответа
        if 'expected_fields' in scenario and result['passed']:
            try:
                data = response.json()
                for field in scenario['expected_fields']:
                    if field not in data:
                        result['passed'] = False
                        result['missing_field'] = field
                        break
            except:
                pass

        return result

    except Exception as e:
        return {
            'name': name,
            'status': 'ERROR',
            'error': str(e),
            'passed': False
        }


def main():
    # Загрузка сценариев
    scenario_file = Path('scenario.json')
    if not scenario_file.exists():
        print(f" Scenario file not found: {scenario_file}")
        return 1

    with open(scenario_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    scenarios = config.get('test_scenarios', [])
    print(f"🧪 Running {len(scenarios)} test scenarios...\n")

    # Ждём сервис
    if not wait_for_service(BASE_URL):
        return 1

    # Запуск тестов
    results = []
    for scenario in scenarios:
        result = run_scenario(scenario)
        results.append(result)
        status = "PASS" if result['passed'] else "FAIL"
        print(f"{status} {result['name']} (status: {result.get('status_code', 'N/A')})")

    # Итоги
    passed = sum(1 for r in results if r['passed'])
    total = len(results)

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"{'=' * 60}")

    # Сохранение результатов
    output = {
        'total': total,
        'passed': passed,
        'failed': total - passed,
        'results': results
    }

    Path('test_results').mkdir(exist_ok=True)
    with open('test_results/scenario_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Results saved to test_results/scenario_results.json")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())