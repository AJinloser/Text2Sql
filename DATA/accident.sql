-- =========================================================
-- Table: insurance_products_accident  (Accident Insurance Products)
-- =========================================================
DROP TABLE IF EXISTS insurance_products_accident;

CREATE TABLE insurance_products_accident (
    product_id              SERIAL PRIMARY KEY,
    company_name            VARCHAR(100),   -- 保险公司
    product_name            VARCHAR(150),   -- 产品名称
    insurance_type          VARCHAR(50),    -- 保险类型

    coverage_content        JSONB,          -- 保险责任
    exclusion_clause        JSONB,          -- 责任免除
    renewable               BOOLEAN,        -- 是否可续保
    underwriting_rules      JSONB,          -- 核保规则

    entry_age               VARCHAR(50),    -- 投保年龄
    deductible              DECIMAL(12,2),  -- 免赔额

    premium                 JSONB,          -- 保费
    coverage_amount         JSONB,          -- 保额

    coverage_period         VARCHAR(50),    -- 保障期限
    occupation              VARCHAR(20),    -- 职业类别
    payment_period          VARCHAR(10),    -- 缴费期限

    hospital_scope          TEXT,           -- 就诊医院范围
    reimbursement_scope     TEXT,           -- 报销范围

    accidental_death        JSONB,          -- 意外身故保障
    accidental_disability   JSONB,          -- 意外伤残保障
    accidental_medical      VARCHAR(50),          -- 意外医疗保障

    total_score             DECIMAL(6,2)    -- 综合评分
);


INSERT INTO insurance_products_accident (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, hospital_scope, reimbursement_scope, accidental_death, accidental_disability, accidental_medical, total_score) VALUES ('少儿意外（性价比）【推荐】', '平安财险 – 平安小玩童6号尊贵版', '意外险', $${
  "其他责任": {
    "note": "其他责任:"
  },
  "预防接种意外身故及伤残保险金": {
    "amount": 200000,
    "unit": "元"
  },
  "意外烫伤导致的意外医疗": {
    "amount": 30000,
    "unit": "元",
    "note": "/0免赔/100%报销 (私立医院挂号费限200元/次, 床位费限200元/天)"
  },
  "误食异物导致的意外医疗": {
    "amount": 20000,
    "unit": "元"
  },
  "航空意外身故伤残": {
    "amount": 1000000,
    "unit": "元"
  },
  "意外骨折/关节脱位保险金": {
    "amount": 5000,
    "unit": "元",
    "note": "*比例"
  },
  "意外住院津贴": {
    "amount": 100,
    "unit": "元",
    "note": "/天, 累计限30天 (寒暑假200元/天; 寒假1-2月, 暑假7-9月, 累计赔付180天)"
  },
  "面部意外美容医疗": {
    "amount": 3000,
    "unit": "元"
  },
  "未成年人第三者责任": {
    "amount": 20000,
    "unit": "元"
  }
}$$, NULL, NULL, NULL, '30天-17周岁', 0, $${
   "必选": 174,
   "可选": 220,
   "单位":"元"
 }$$, NULL, '1年', NULL, '1年', '二级及二级以上公立医院普通部及私立医院普通部', '不限社保', $${"plan":"default","amount": 500000,"unit":"元"}$$, $${"plan":"default","amount": 1000000,"unit":"元"}$$, '100000元(含门诊住院)', 89.4);
INSERT INTO insurance_products_accident (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, hospital_scope, reimbursement_scope, accidental_death, accidental_disability, accidental_medical, total_score) VALUES ('少儿意外（高品质）【推荐】', '平安财险 – 平安小玩童6号高端版', '意外险', $${
  "其他责任": {
    "note": "其他责任:"
  },
  "预防接种意外身故及伤残保险金": {
    "amount": 200000,
    "unit": "元"
  },
  "意外烫伤导致的意外医疗": {
    "amount": 30000,
    "unit": "元",
    "note": "/0免赔/100%报销 (私立医院挂号费限500元/次, 床位费限200元/天)"
  },
  "误食异物导致的意外医疗": {
    "amount": 20000,
    "unit": "元"
  },
  "航空意外身故伤残": {
    "amount": 1000000,
    "unit": "元"
  },
  "救护车费用": {
    "amount": 5000,
    "unit": "元"
  },
  "急性病身故(含猝死)": {
    "amount": 500000,
    "unit": "元"
  },
  "意外骨折/关节脱位保险金": {
    "amount": 5000,
    "unit": "元",
    "note": "*比例"
  },
  "意外住院津贴": {
    "amount": 100,
    "unit": "元",
    "note": "/天, 累计限30天 (寒暑假200元/天; 寒假1-2月, 暑假7-9月, 累计赔付180天)"
  },
  "面部意外美容医疗": {
    "amount": 3000,
    "unit": "元"
  },
  "未成年人第三者责任": {
    "amount": 100000,
    "unit": "元"
  }
}$$::jsonb, NULL, NULL, NULL, '30天-17周岁', 0, $${"plan":"default","amount": 520,"unit":"元"}$$::jsonb, NULL, '1年', NULL, '1年', '二级及二级以上公立医院普通部/特需部/国际部/VIP部/干部病房等及私立医院', '不限社保', $${"plan":"default","amount": 500000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 1000000,"unit":"元"}$$::jsonb, '200000元(含门诊住院)', 85.5);
INSERT INTO insurance_products_accident (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, hospital_scope, reimbursement_scope, accidental_death, accidental_disability, accidental_medical, total_score) VALUES ('少儿意外（高品质）【推荐】', '美亚 – “宝贝无忧”2023版计划五', '意外险', $${
  "其他责任": {
    "note": "其他责任:"
  },
  "私立医院和诊所每次事故限额": {
    "amount": 10000,
    "unit": "元",
    "note": "其中挂号费限额1500元/次"
  },
  "救护车每次事故限额": {
    "amount": 2000,
    "unit": "元"
  },
  "未成年人责任": {
    "amount": 200000,
    "unit": "元"
  },
  "① 人身伤亡限额": {
    "amount": 200000,
    "unit": "元",
    "note": "免赔: 0元"
  },
  "② 财产损失限额": {
    "amount": 20000,
    "unit": "元",
    "note": "免赔: 1000元"
  },
  "其他认可医疗机构": {
    "note": "其他认可医疗机构:"
  }
}$$::jsonb, NULL, NULL, NULL, '30天-17周岁', 0, $${
   "必选": 699,
   "可选": 719}$$::jsonb, NULL, '1年', NULL, '1年', '二级及二级以上公立医院普通部/特需部/境内外认可医疗机构', '不限社保', $${"plan":"default","amount": 200000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 200000,"unit":"元"}$$::jsonb, '50000元(次限额)', 80.9);
INSERT INTO insurance_products_accident (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, hospital_scope, reimbursement_scope, accidental_death, accidental_disability, accidental_medical, total_score) VALUES ('成人意外（性价比）', '中国人民保险 – 人保财大金刚2号至尊版', '意外险', $${
  "其他责任": {
    "note": "其他责任:"
  },
  "航空意外身故伤残": {
    "amount": 5000000,
    "unit": "元"
  },
  "火车意外身故伤残": {
    "amount": 1000000,
    "unit": "元"
  },
  "轮船意外身故伤残": {
    "amount": 800000,
    "unit": "元"
  },
  "乘坐汽车身故伤残": {
    "amount": 300000,
    "unit": "元"
  },
  "驾驶非营运汽车身故伤残": {
    "amount": 300000,
    "unit": "元"
  },
  "急性病身故(含猝死)": {
    "amount": 400000,
    "unit": "元"
  },
  "一般意外住院津贴": {
    "amount": 150,
    "unit": "元",
    "note": "/天, 0免赔, 单次≤30天, 年度≤180天"
  },
  "意外ICU住院津贴": {
    "amount": 600,
    "unit": "元",
    "note": "/天, 与一般住院津贴不叠加"
  },
  "预防接种医疗意外身故伤残": {
    "amount": 250000,
    "unit": "元"
  },
  "骨折脱臼": {
    "amount": 5000,
    "unit": "元",
    "note": "*比例"
  },
  "救护车责任": {
    "amount": 3000,
    "unit": "元",
    "note": "0免赔"
  }
}$$::jsonb, NULL, NULL, NULL, '18周岁-55周岁', 0, $${"plan":"default","amount": 298,"unit":"元"}$$::jsonb, NULL, '1年', '1-3类', '1年', '二级及二级以上医院或保险人认可的医疗机构', '不限社保', $${"plan":"default","amount": 1000000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 1000000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 120000,"unit":"元"}$$::jsonb, 85.1);
INSERT INTO insurance_products_accident (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, hospital_scope, reimbursement_scope, accidental_death, accidental_disability, accidental_medical, total_score) VALUES ('成人意外', '平安 – 平安大守护至尊版', '意外险', $${
  "其他责任": {
    "note": "其他责任:"
  },
  "航空意外身故伤残": {
    "amount": 3000000,
    "unit": "元"
  },
  "意外住院津贴": {
    "amount": 150,
    "unit": "元",
    "note": "/天, 无免赔, 单次≤90天, 年度≤180天"
  },
  "猝死保险金": {
    "amount": 500000,
    "unit": "元"
  },
  "备注": {
    "note": "无健康告知"
  }
}$$::jsonb, NULL, NULL, NULL, '18周岁-60周岁', 0, $${"plan":"default","amount": 466,"unit":"元"}$$::jsonb, NULL, '1年', '1-3类', '1年', '二级及二级以上公立医院普通部', '意外医疗拓展社保外用药', $${"plan":"default","amount": 1000000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 1000000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 50000,"unit":"元"}$$::jsonb, 79.3);
INSERT INTO insurance_products_accident (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, hospital_scope, reimbursement_scope, accidental_death, accidental_disability, accidental_medical, total_score) VALUES ('成人意外（停售）', '黄河财险 – 黄河大保镖至尊版', '意外险', $${
  "其他责任": {
    "note": "其他责任:"
  },
  "飞机意外身故伤残": {
    "amount": 3000000,
    "unit": "元"
  },
  "轨道交通意外身故伤残": {
    "amount": 800000,
    "unit": "元"
  },
  "轮船意外身故伤残": {
    "amount": 800000,
    "unit": "元"
  },
  "营运汽车意外身故伤残": {
    "amount": 800000,
    "unit": "元"
  },
  "非营运汽车意外身故伤残": {
    "amount": 800000,
    "unit": "元"
  },
  "猝死保险金": {
    "amount": 300000,
    "unit": "元"
  },
  "骨折津贴": {
    "amount": 100,
    "unit": "元",
    "note": "/天, 单次≤60天, 年度≤180天"
  },
  "意外住院津贴": {
    "amount": 150,
    "unit": "元",
    "note": "/天, 单次≤90天, 年度≤180天"
  }
}$$::jsonb, NULL, NULL, NULL, '18周岁-60周岁', 0, $${"plan":"default","amount": 308,"unit":"元"}$$::jsonb, NULL, '1年', '1-3类', '1年', '二级及二级以上公立医院普通部', '拓展乙类/丙类药品费用', $${"plan":"default","amount": 1000000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 1000000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 50000,"unit":"元"}$$::jsonb, 79.3);
INSERT INTO insurance_products_accident (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, hospital_scope, reimbursement_scope, accidental_death, accidental_disability, accidental_medical, total_score) VALUES ('成人意外（性价比）', '中国人民保险 – 人保全能守护者综合意外', '意外险', $${
  "其他责任": {
    "note": "其他责任:"
  },
  "航空意外身故伤残": {
    "amount": 8000000,
    "unit": "元"
  },
  "火车意外身故伤残": {
    "amount": 800000,
    "unit": "元"
  },
  "轮船意外身故伤残": {
    "amount": 1000000,
    "unit": "元"
  },
  "营运汽车身故伤残": {
    "amount": 600000,
    "unit": "元"
  },
  "驾驶非营运汽车身故伤残": {
    "amount": 300000,
    "unit": "元"
  },
  "突发疾病身故": {
    "amount": 500000,
    "unit": "元",
    "note": "(等待期7天)"
  },
  "一般意外住院津贴": {
    "amount": 200,
    "unit": "元",
    "note": "/天, 0免赔, 单次≤90天, 年度≤180天"
  }
}$$::jsonb, NULL, NULL, NULL, '18周岁-60周岁', 0, $${"plan":"default","amount": 355,"unit":"元"}$$::jsonb, NULL, '1年', '1-3类', '1年', '二级及二级以上公立医院普通部', '不限社保', $${"plan":"default","amount": 1000000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 1000000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 100000,"unit":"元"}$$::jsonb, 86.4);
INSERT INTO insurance_products_accident (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, hospital_scope, reimbursement_scope, accidental_death, accidental_disability, accidental_medical, total_score) VALUES ('成人意外（停售）', '亚太财险 – 麒麟保1号至尊版', '意外险', $${
  "其他责任": {
    "note": "其他责任:"
  },
  "猝死保险金": {
    "amount": 500000,
    "unit": "元"
  },
  "ICU津贴": {
    "amount": 300,
    "unit": "元",
    "note": "/天"
  },
  "预防接种身故伤残": {
    "amount": 200000,
    "unit": "元"
  },
  "自行车驾驶人身故伤残": {
    "amount": 100000,
    "unit": "元"
  },
  "救护车费用": {
    "amount": 1000,
    "unit": "元"
  },
  "公共场所个人第三者责任": {
    "amount": 50000,
    "unit": "元"
  },
  "乘坐飞机身故伤残": {
    "amount": 10000000,
    "unit": "元"
  },
  "乘坐火车身故伤残": {
    "amount": 800000,
    "unit": "元"
  },
  "乘坐轮船身故伤残": {
    "amount": 800000,
    "unit": "元"
  },
  "客运汽车身故伤残": {
    "amount": 300000,
    "unit": "元"
  },
  "驾驶非营运汽车身故伤残": {
    "amount": 300000,
    "unit": "元"
  }
}$$::jsonb, NULL, NULL, NULL, '18周岁-60周岁', 0, $${"plan":"default","amount": 296,"unit":"元"}$$::jsonb, NULL, '1年', '1-3类', '1年', '二级及二级以上公立医院普通部', '不限社保', $${"plan":"default","amount": 1000000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 1000000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 100000,"unit":"元"}$$::jsonb, 87.3);
INSERT INTO insurance_products_accident (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, hospital_scope, reimbursement_scope, accidental_death, accidental_disability, accidental_medical, total_score) VALUES ('高额意外', '安盛天平 – 卓越优选计划四', '意外险', $${
  "其他责任": {
    "note": "/"
  }
}$$::jsonb, NULL, NULL, NULL, '18周岁-55周岁', 0, $${
   "一类": 840,
   "二类": 1050,
   "三类": 1680}$$::jsonb, NULL, '1年', '1-3类', '1年', NULL, NULL, $${"plan":"default","amount": 3000000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 3000000,"unit":"元"}$$::jsonb, NULL, 73.2);
INSERT INTO insurance_products_accident (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, hospital_scope, reimbursement_scope, accidental_death, accidental_disability, accidental_medical, total_score) VALUES ('高额意外', '美亚 – 美满无忧高端个人意外计划三', '意外险', $${
  "其他责任": {
    "note": "其他责任:"
  },
  "意外医疗(可选)": {
    "note": "境内二级及以上公立医院 100% 赔付 / 境外或公立医院特需部 80% 赔付"
  }
}$$::jsonb, NULL, NULL, NULL, '18周岁-65周岁', 0, $${"plan":"default","amount": 1933,"unit":"元"}$$::jsonb, NULL, '1年', '1-2类', '1年', NULL, NULL, $${"plan":"default","amount": 3000000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 3000000,"unit":"元"}$$::jsonb, NULL, 72.7);
INSERT INTO insurance_products_accident (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, hospital_scope, reimbursement_scope, accidental_death, accidental_disability, accidental_medical, total_score) VALUES ('老年意外', '美亚 – “尊长无忧”中老年人个人意外险计划三', '意外险', $${
  "其他责任": {
    "note": "其他责任:"
  },
  "意外医疗": {
    "note": "每次限额 20000元, 免赔 100元, 赔付 90% (未社保结算 60%)"
  },
  "骨折理疗": {
    "amount": 2000,
    "unit": "元",
    "note": "/次, 0免赔, 赔付 90%"
  },
  "骨折护具": {
    "amount": 2000,
    "unit": "元",
    "note": "/次, 0免赔, 赔付 90%"
  },
  "骨折意外保障": {
    "amount": 10000,
    "unit": "元",
    "note": "*比例 (66岁以上 5000元)"
  },
  "骨折住院定额给付": {
    "amount": 700,
    "unit": "元",
    "note": "/周, 单次≤2周, 年度≤12周"
  },
  "账户咨询损失": {
    "amount": 100000,
    "unit": "元",
    "note": "100%赔付"
  },
  "电信网络诈骗损失": {
    "amount": 50000,
    "unit": "元",
    "note": "70%赔付, 免赔 3000元, 等待期 7天"
  },
  "可选": {
    "note": "医疗运送与送返 300000元/150000元, 身故遗体送返 100000元/50000元,"
  }
}$$::jsonb, NULL, NULL, NULL, '50周岁-80周岁', 0, $${
   "必选": 422,
   "可选": 522}$$::jsonb, NULL, '1年', '1-3类', '1年', '二级及二级以上公立医院普通部', '意外医疗不限社保', $${"plan":"default","amount": 100000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 200000,"unit":"元"}$$::jsonb, '100000元(次限额20000元)', 83.4);
INSERT INTO insurance_products_accident (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, hospital_scope, reimbursement_scope, accidental_death, accidental_disability, accidental_medical, total_score) VALUES ('老年意外', '平安 – 平安大守护老年人意外险计划四', '意外险', $${
  "其他责任": {
    "note": "其他责任:"
  },
  "意外住院津贴": {
    "amount": 100,
    "unit": "元",
    "note": "/天, 免赔3天, 单次≤90天, 年度≤180天"
  },
  "救护车费用补偿": {
    "amount": 1000,
    "unit": "元"
  }
}$$::jsonb, NULL, NULL, NULL, '50周岁-80周岁', 0, $${"plan":"default","amount": 400,"unit":"元"}$$::jsonb, NULL, '1年', '1-3类', '1年', '二级及二级以上公立医院普通部', '拓展社保外用药', $${"plan":"default","amount": 50000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 50000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 20000,"unit":"元"}$$::jsonb, 73.4);
INSERT INTO insurance_products_accident (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, hospital_scope, reimbursement_scope, accidental_death, accidental_disability, accidental_medical, total_score) VALUES ('老年意外', '安盛天平 – 守护时光个人意外伤害保险计划二', '意外险', $${
  "其他责任": {
    "note": "其他责任:"
  },
  "救护车费用": {
    "amount": 1000,
    "unit": "元"
  },
  "住院津贴": {
    "amount": 80,
    "unit": "元",
    "note": "/天, 限30天"
  },
  "骨折住院津贴": {
    "amount": 100,
    "unit": "元",
    "note": "/天, 限30天"
  }
}$$::jsonb, NULL, NULL, NULL, '45周岁（含）-80周岁', 0, $${"plan":"default","amount": 298,"unit":"元"}$$::jsonb, NULL, '1年', '1-3类', '1年', '二级及二级以上公立医院普通部', '意外医疗拓展社保外用药', $${"plan":"default","amount": 100000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 100000,"unit":"元"}$$::jsonb, '30000元(骨折限额15000元)', 75.3);
INSERT INTO insurance_products_accident (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, hospital_scope, reimbursement_scope, accidental_death, accidental_disability, accidental_medical, total_score) VALUES ('特定职业意外', '中国人民保险 – 人保1-6类职业成人意外', '意外险', $${
  "其他责任": {
    "amount": 5,
    "unit": "元",
    "note": "-6类职业可选"
  }
}$$::jsonb, NULL, NULL, NULL, '18周岁-55周岁', 0, $${
   "plan1": 299,
   "plan2": 599,
   "plan3": 999}$$::jsonb, NULL, '1年', '1-6类', '1年', '二级及二级以上公立医院普通部', '限社保内', NULL, NULL,NULL, 76.8);
INSERT INTO insurance_products_accident (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, hospital_scope, reimbursement_scope, accidental_death, accidental_disability, accidental_medical, total_score) VALUES ('综合意外（高品质）（停售）', '利宝保险 – 利宝蓝精灵个人意外险升级款', '意外险', $${
  "其他责任": {
    "note": "其他责任:"
  },
  "飞机意外身故伤残": {
    "amount": 1000000,
    "unit": "元"
  },
  "轨道交通意外身故伤残": {
    "amount": 500000,
    "unit": "元"
  },
  "救护车费用": {
    "note": "≤2000元"
  },
  "私立医院/公立医院特需部单次赔付限额": {
    "amount": 10000,
    "unit": "元"
  },
  "动物咬伤治疗费用": {
    "note": "单次限额 3000元"
  }
}$$::jsonb, NULL, NULL, NULL, '30天-60周岁', 0, $${"plan":"default","amount": 630,"unit":"元"}$$::jsonb, NULL, '1年', '1-3类', '1年', '二级及二级以上公立医院普通部/特需部/国际部/私立医院', '不限社保', $${"plan":"default","amount": 200000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 200000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 50000,"unit":"元"}$$::jsonb, 80.3);
INSERT INTO insurance_products_accident (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, hospital_scope, reimbursement_scope, accidental_death, accidental_disability, accidental_medical, total_score) VALUES ('综合意外（性价比）', '中华联合财险 – 中华联合大护法意外险2024版计划四', '意外险', $${
  "其他责任": {
    "note": "其他责任:"
  },
  "住院津贴": {
    "amount": 100,
    "unit": "元",
    "note": "/天, 免赔3天, 手术10天/非手术3天, 年度≤180天"
  },
  "猝死保险金": {
    "amount": 0,
    "unit": "元",
    "note": "-50岁 150000元; 51-65岁 50000元"
  },
  "新冠疫苗接种身故伤残": {
    "amount": 200000,
    "unit": "元"
  },
  "甲乙类法定传染病身故": {
    "amount": 200000,
    "unit": "元",
    "note": "(30天等待期)"
  },
  "航空意外身故伤残": {
    "amount": 5000000,
    "unit": "元"
  },
  "火车意外身故伤残": {
    "amount": 1000000,
    "unit": "元"
  },
  "轮船意外身故伤残": {
    "amount": 1000000,
    "unit": "元"
  },
  "营运汽车意外身故伤残": {
    "amount": 300000,
    "unit": "元"
  },
  "骑行意外": {
    "amount": 50000,
    "unit": "元"
  },
  "救护车费用": {
    "amount": 2000,
    "unit": "元"
  }
}$$::jsonb, NULL, NULL, NULL, '28天-65周岁', 0, $${"plan":"default","amount": 309,"unit":"元"}$$::jsonb, NULL, '1年', '1-3类', '1年', '二级及二级以上公立医院普通部', '不限社保', $${"plan":"default","amount": 1000000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 1000000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 100000,"unit":"元"}$$::jsonb, 86.8);
INSERT INTO insurance_products_accident (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, hospital_scope, reimbursement_scope, accidental_death, accidental_disability, accidental_medical, total_score) VALUES ('综合意外', '美亚财险 – 乐享百万人生个人意外伤害保险计划三', '意外险', $${
  "其他责任": {
    "note": "其他责任:"
  },
  "民航意外身故伤残": {
    "amount": 800000,
    "unit": "元"
  },
  "轨道交通意外身故伤残": {
    "amount": 300000,
    "unit": "元"
  },
  "公交意外身故伤残": {
    "amount": 300000,
    "unit": "元"
  },
  "私家车驾乘意外身故伤残": {
    "amount": 300000,
    "unit": "元"
  },
  "猝死保障": {
    "amount": 50000,
    "unit": "元"
  },
  "住院津贴": {
    "amount": 300,
    "unit": "元",
    "note": "/天, 免赔3天, 年度≤100天 (非手术住院≤7天/次)"
  },
  "ICU津贴": {
    "amount": 600,
    "unit": "元",
    "note": "/天, 限30天, 等待期90天"
  }
}$$::jsonb, NULL, NULL, NULL, '3个月-65周岁', 0, $${"plan":"default","amount": 450,"unit":"元"}$$::jsonb, NULL, '1年', '1-4类', '1年', '大陆境内二级及以上公立医院普通部及境外认可机构', '不限社保', $${"plan":"default","amount": 300000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 300000,"unit":"元"}$$::jsonb, '20000元/次', 81.9);
INSERT INTO insurance_products_accident (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, hospital_scope, reimbursement_scope, accidental_death, accidental_disability, accidental_medical, total_score) VALUES ('综合意外', '安盛天平 – 守护百万综合意外伤害保险计划三', '意外险', $${
  "其他责任": {
    "note": "其他责任:"
  },
  "民航飞机意外身故伤残": {
    "amount": 1000000,
    "unit": "元"
  },
  "轨道/水上公共交通工具意外身故伤残": {
    "amount": 300000,
    "unit": "元"
  },
  "公共营运汽车/出租车/网约车意外身故伤残": {
    "amount": 300000,
    "unit": "元"
  },
  "自驾/乘坐非营运机动车意外身故伤残": {
    "amount": 300000,
    "unit": "元"
  },
  "猝死保障": {
    "amount": 50000,
    "unit": "元"
  },
  "住院津贴": {
    "amount": 100,
    "unit": "元",
    "note": "/天, 免赔3天, 年度≤100天"
  },
  "ICU津贴": {
    "amount": 600,
    "unit": "元",
    "note": "/天, 限30天"
  },
  "救护车费用": {
    "amount": 500,
    "unit": "元"
  },
  "个人责任": {
    "amount": 80000,
    "unit": "元",
    "note": "(单件物品限额 1000元)"
  }
}$$::jsonb, NULL, NULL, NULL, '180天-65周岁', 0, $${"plan":"default","amount": 450,"unit":"元"}$$::jsonb, NULL, '1年', '1-4类', '1年', '二级及二级以上公立医院普通部及保险人认可医疗机构', '拓展社保外医疗', $${"plan":"default","amount": 300000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 300000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 25000,"unit":"元"}$$::jsonb, 76.9);
INSERT INTO insurance_products_accident (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, hospital_scope, reimbursement_scope, accidental_death, accidental_disability, accidental_medical, total_score) VALUES ('综合意外', '平安财险 – 小蜜蜂（轻享版）综合意外险尊享版', '意外险', $${
  "其他责任": {
    "note": "其他责任:"
  },
  "飞机意外身故伤残": {
    "amount": 3000000,
    "unit": "元"
  },
  "火车意外身故伤残": {
    "amount": 500000,
    "unit": "元"
  },
  "轮船意外身故伤残": {
    "amount": 500000,
    "unit": "元"
  },
  "汽车意外身故伤残": {
    "amount": 300000,
    "unit": "元"
  },
  "救护车责任": {
    "amount": 1000,
    "unit": "元",
    "note": "报销"
  }
}$$::jsonb, NULL, NULL, NULL, '18-60周岁', 0, $${"plan":"default","amount": 168,"unit":"元"}$$::jsonb, NULL, '1年', '1-3类', '1年', '二级及二级以上公立医院普通部', '限社保内', $${"plan":"default","amount": 500000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 500000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 40000,"unit":"元"}$$::jsonb, 85.2);
INSERT INTO insurance_products_accident (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, hospital_scope, reimbursement_scope, accidental_death, accidental_disability, accidental_medical, total_score) VALUES ('非明亚—老年意外', '友邦保险 – 守护长青意外保险', '意外险', $${
  "其他责任": {
    "note": "其他责任:"
  },
  "意外骨折医疗保险金": {
    "amount": 10000,
    "unit": "元",
    "note": "*骨折比例"
  },
  "骨折住院日额保险金": {
    "note": "基本保额100元/日"
  },
  "ICU骨折住院日额": {
    "note": "基本保额300元/日"
  },
  "意外医药费用补偿金": {
    "amount": 10000,
    "unit": "元",
    "note": "(限社保内用药)"
  }
}$$::jsonb, NULL, NULL, NULL, '45-75周岁', 0, $${"plan":"default","amount": 567,"unit":"元"}$$::jsonb, NULL, '1年', '1-3类', '1年', '二级及二级以上公立医院普通部', '限社保内', NULL, $${"plan":"default","amount": 300000,"unit":"元"}$$::jsonb, $${"plan":"default","amount": 10000,"unit":"元"}$$::jsonb, 76.8);
