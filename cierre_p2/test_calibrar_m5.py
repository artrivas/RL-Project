import unittest
from datetime import date, timedelta
from dataclasses import replace
import numpy as np
from calibrar_m5 import RetailConfig, calibrate, design, price_fit, BASE
from src.retail.data import RetailData
import json


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.n = 1941
        self.prices = np.repeat(self.rng.choice([1., 1.2, 1.5, 2.], 278), 7)[:self.n]
        self.calendar = tuple(dict(week=i//7, weekday=i%7, snap=0, event=0,
                                   month=(date(2011, 1, 29)+timedelta(days=i)).month)
                              for i in range(self.n))
        self.expected = 80 * self.prices**-1.8

    def test_elasticity_sign_and_absent_price_variation(self):
        x = design(self.calendar, self.prices)
        b = price_fit(x, self.expected)
        self.assertAlmostEqual(-b[-1], 1.8, places=5)
        self.assertIsNone(price_fit(design(self.calendar, np.ones(self.n)), self.expected))

    def test_future_data_cannot_change_calibration(self):
        y = tuple(map(int, self.rng.poisson(self.expected)))
        data = RetailData('SYNTHETIC', (y,)*3, (tuple(self.prices),)*3,
                          self.calendar, {'sha256': {}})
        cfg = RetailConfig(**json.loads((BASE/'docker/m5_final_assumptions.json').read_text()))
        report = calibrate(data, cfg)
        v = report['H0']['0.9']['validacion']
        central = next(r for r in v['sensibilidad_costes'] if r['factor_almacenamiento']==r['factor_penalizacion']==1)
        self.assertAlmostEqual(v['base_return'], central['retorno'], places=6)
        changed_test = y[:1576] + (999,)*(self.n-1576)
        other = calibrate(replace(data, sales=(changed_test,)*3), cfg)
        self.assertEqual(report, other)
        changed_validation = y[:1211] + (25,)*(self.n-1211)
        other = calibrate(replace(data, sales=(changed_validation,)*3), cfg)
        self.assertEqual(report['tiendas'], other['tiendas'])
        self.assertNotEqual(report['H0'], other['H0'])

if __name__ == '__main__':
    unittest.main()
