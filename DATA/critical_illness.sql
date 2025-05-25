
-- =========================================================
-- Table: insurance_products_critical_illness (Critical Illness Products)
-- =========================================================
DROP TABLE IF EXISTS insurance_products_critical_illness;

CREATE TABLE insurance_products_critical_illness (
    product_id              SERIAL           PRIMARY KEY,
    company_name            VARCHAR(100),
    product_name            VARCHAR(150),
    insurance_type          VARCHAR(50),
    
    coverage_content        JSONB,
    exclusion_clause        JSONB,
    renewable               BOOLEAN,
    underwriting_rules      JSONB,
    
    entry_age               JSONB,
    deductible              DECIMAL(12,2), 
    premium                 JSONB,
    coverage_amount         JSONB,
    
    coverage_period         VARCHAR(50),
    waiting_period          VARCHAR(10),
    payment_period          VARCHAR(50),
    payment_method          VARCHAR(20),
    
    second_insured          VARCHAR(10),
    clause_link             TEXT,
    optional_liabilities    TEXT,
    insurance_option        TEXT,
    
    addtl_owner_waiver      VARCHAR(20), 
    payment_term_options    JSONB,
    insurance_rules         TEXT,
    owner_waiver            VARCHAR(50),
    
    additional_riders       TEXT,
    highlights              TEXT,
    
    commission              DECIMAL(12,2), 
    commission_year1        DECIMAL(12,2),
    commission_year2        DECIMAL(12,2),
    commission_year3        DECIMAL(12,2),
    commission_year4        DECIMAL(12,2),
    commission_year5        DECIMAL(12,2),
    
    total_score             DECIMAL(6,2)
);


INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '瑞众人寿', '瑞泽一生（臻选版）重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 130,
        "groups": 6,
        "maxPayouts": 6,
        "interval": "180天",
        "cancerSeparateGroup": true,
        "firstPayoutBasis": "首次给付累计保费&现金价值&100%保额取大，后5次100%基本保额"
    },
    "中症": {
        "count": 20,
        "maxPayouts": 2,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "轻症": {
        "count": 50,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "min": "28天",
    "max": "65周岁"
}
$$, 0, $$
{
    "minAmount": 5322,
    "unit": "元"
}
$$, $$
{
    "minAmount": 1000000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/crg0qV5Ulb2n', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '附加', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        3,
        5,
        10,
        15,
        20,
        25,
        30
    ]
}
$$, '10000元保额起，1千整数倍递增', '可附加', '可附加住院医疗', '1、满足条件可选优选体费率，费率更优
2、可选单次、多次赔付
3、可选轻中症
4、轻/中/重疾均有三同',2048.97,0.385,0.17,0.03,0.025,0.01,90.75);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '瑞众人寿', '瑞泽一生（多倍版）重大疾病保险', '多次赔付重疾险', $$
{
    "产品责任": {
        "count": 130,
        "groups": 6,
        "maxPayouts": 6,
        "interval": "180天",
        "cancerSeparateGroup": true,
        "firstPayoutBasis": "首次给付累计保费&现金价值&100%保额取大，后5次100%基本保额"
    },
    "中症": {
        "count": 20,
        "maxPayouts": 2,
        "groups": 0,
        "stepPayout": "依次给付55%/65%基本保额"
    },
    "轻症": {
        "count": 50,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "min": "28天",
    "max": "65周岁"
}
$$, 0, $$
{
    "minAmount": 5523,
    "unit": "元"
}
$$, $$
{
    "minAmount": 1000000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/csrCyR1hgivz', $$
{
    "其他": {}
}
$$, NULL, '附加', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        3,
        5,
        10,
        15,
        20,
        25,
        30
    ]
}
$$, '10000元保额起，1千整数倍递增', '可附加', '可附加住院医疗', '1、满足条件可选优选体费率，费率更优
2、自带首次重疾额外给付
3、轻/中/重疾均有三同',4268.88,0.385,0.17,0.03,0.025,0.01,91.95);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '信泰人寿', '如意久久守护（2025）重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "groups": 6,
        "maxPayouts": 6,
        "cancerSeparateGroup": true,
        "stepPayout": "依次给付100%/130%/160%/200%/200%/200%基本保额"
    },
    "中症": {
        "count": 25,
        "maxPayouts": 2,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "重疾": {},
    "轻症": {
        "count": 50,
        "maxPayouts": 4,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "min": "28天",
    "max": "60周岁"
}
$$, 0, $$
{
    "minAmount": 7659,
    "unit": "元"
}
$$, $$
{
    "minAmount": 800000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/cuCNTaZxfU6B', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '不附加', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        5,
        10,
        15,
        20,
        30
    ]
}
$$, '10000元保额起，1千整数倍递增', '可附加', '可附加两全', '1、可选70周岁前首次重疾双倍赔付
2、首次确诊轻/中症后，再确诊首次重疾，首次重疾可额外给付
3、重疾给付结束前，轻中症可继续给付
4、可选身故赔付保额/保费
5、轻中重症均有三同',3331.66,0.435,0.21,0.1,0.06,0.035,85.05);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '富德生命人寿', '臻健康重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 121,
        "groups": 6,
        "maxPayouts": 6,
        "cancerSeparateGroup": true,
        "eachPayout": "每次给付100%基本保额"
    },
    "中症": {
        "count": 20,
        "maxPayouts": 2,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 6,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "min": "25天",
    "max": "60周岁"
}
$$, 0, $$
{
    "minAmount": 4800,
    "unit": "元"
}
$$, $$
{
    "minAmount": 410000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/ctAvahjsMBbs', NULL, NULL, '不附加', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        5,
        10,
        15,
        20,
        25,
        30
    ]
}
$$, '50000元保额起，1千整数倍递增', '身故/全残，轻中重症，可附加', '可附加医疗，意外', '1、重疾6组6次，65岁及以后重疾1.5倍赔付
2、身故默认赔付保费
3、重疾分组无三同，轻中症有三同',1992,0.415,0.1,0.05,0.03,0.015,83.55);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '百年人寿', '康多保（2024版）终身重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "groups": 6,
        "maxPayouts": 6,
        "eachPayout": "间隔期365天，每次给付100%基本保额"
    },
    "中症": {
        "count": 25,
        "maxPayouts": 2,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 5,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "min": "28天",
    "max": "55周岁"
}
$$, 0, $$
{
    "minAmount": 6201,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/cdnAtklqyEB6', $$
{
    "轻症": {}
}
$$, $$
{
    "其他": {
        "firstPayoutBasis": "首次重疾豁免保费&重疾多次给付保险金"
    }
}
$$, '暂无', $$
{
    "其他": {}
}
$$, '100000元保额起，1千整数倍递增', '暂无', '可附加', '1、百年首创，20种更早期的前症（肺结节手术/乳腺不典型小叶增生手术等）也可以赔付，可豁免后续保费',2418.39,0.39,0.2,0.09,0.025,0.005,72.45);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '瑞泰人寿', '乐享无忧（尊享版）重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 110,
        "maxPayouts": 2,
        "groups": 0,
        "stepPayout": "间隔期365天，依次给付100%/130%基本保额"
    },
    "中症": {
        "count": 20,
        "maxPayouts": 2,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额",
        "firstPayoutBasis": "首次重疾确诊之日起满3年发生中症，若中症给付未达2次，可继续给付一次"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额",
        "firstPayoutBasis": "首次重疾确诊之日起满3年发生轻症，若轻症给付未达3次，可继续给付一次"
    }
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "65周岁"
}
$$, 0, $$
{
    "minAmount": 4953,
    "unit": "元"
}
$$, $$
{
    "minAmount": 700000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/crnrUr7YSHCz', $$
{
    "其他": {
        "firstPayoutBasis": "首次重疾额外给付保险金"
    }
}
$$, NULL, '不附加', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        3,
        5,
        10,
        15,
        20,
        30
    ]
}
$$, '100000元保额起，1千整数倍递增', '身故/全残，轻中重症，可附加', '可附加', '1、四大可选，随意搭配，责任齐全
2、可选60周岁前首次重疾1.5倍赔付
3、特色良性肿瘤切除术，良性肿瘤也可以赔
4、自带少儿特定重疾，26周岁前首次重疾最高可赔付2.5倍保额
5、重疾不分组，有三同，轻中症有三同',2625.09,0.53,0.22,0.09,0.025,0.015,77.4);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '工银安盛人寿', '御享欣生重大疾病保险（龙腾版）', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 110,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "间隔期365天，每次给付100%基本保额"
    },
    "中症": {
        "count": 30,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付60基本保额"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "min": "28天",
    "max": "55周岁"
}
$$, 0, $$
{
    "minAmount": 6420,
    "unit": "元"
}
$$, $$
{
    "text": "免体检"
}
$$, '终身', '90天，身故无等待期', '20年', NULL, NULL, 'https://kdocs.cn/l/caknXCL3knqA', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '不附加', $$
{
    "其他": {}
}
$$, '100000元保额起', '暂无', '可附加', '1、重疾不分组，无三同
2、重疾绿通保单有效期内有效，疑似阶段即可启动
3、公司品牌好+绿通服务好
4、支持保单复议，保单原核保结果为加费或者除外，保单承保满2周年可申请保单复议（包含甲状腺结节、乳腺结节、胆囊息肉等20多种疾病）',2889,0.45,0.17,0.05,0.04,0.015,73.35);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '工银安盛人寿', '御享人生重大疾病保险（龙腾版）', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 110,
        "maxPayouts": 2,
        "groups": 0,
        "eachPayout": "间隔期365天，每次给付100%基本保额"
    },
    "中症": {
        "count": 30,
        "maxPayouts": 5,
        "groups": 0,
        "eachPayout": "每次给付50%基本保额"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 5,
        "groups": 0,
        "eachPayout": "每次给付20%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 5076,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '90天，身故无等待期', '20年', NULL, NULL, 'https://kdocs.cn/l/cjZFOAIqhh6v', $$
{
    "其他": {}
}
$$, NULL, '不附加', $$
{
    "其他": {}
}
$$, '待上线', NULL, NULL, NULL,2284.2,0.45,0.17,0.05,0.05,0.04,74.55);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '同方全球人寿', '【新康健2024】（无忧版）重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "间隔期365天，每次给付100%基本保额"
    },
    "中症": {
        "count": 25,
        "maxPayouts": 6,
        "groups": 0,
        "interval": "180天",
        "eachPayout": "每次给付60%基本保额（同因疾病间隔180天，不同因无间隔）"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 6,
        "groups": 0,
        "interval": "180天",
        "eachPayout": "每次给付30%基本保额（同因疾病间隔180天，不同因无间隔）"
    }
}
$$, NULL, NULL, NULL, $$
{
    "min": "7天",
    "max": "60周岁"
}
$$, 0, $$
{
    "text": "6004.3032"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/cqi79EaLZ2a4', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '附加', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        5,
        10,
        15,
        18,
        20,
        30
    ]
}
$$, '100000元保额起，10000元整数倍递增', '身故/全残/重症/轻症', '保费3000元以上且保额100000元以上可添加附加险', NULL,2636.55,0.45,0.17,0.05,0.04,0.015,72.45);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '中英人寿', '爱守护3.0重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "maxPayouts": 2,
        "groups": 0,
        "eachPayout": "间隔期365天，每次给付100%基本保额"
    },
    "中症": {
        "count": 20,
        "maxPayouts": 2,
        "groups": 0,
        "interval": "180天",
        "eachPayout": "每次给付50%基本保额（同因疾病间隔180天，不同因无间隔）"
    },
    "轻症": {
        "count": 50,
        "maxPayouts": 2,
        "groups": 0,
        "interval": "180天",
        "eachPayout": "每次给付30%基本保额（同因疾病间隔180天，不同因无间隔）"
    }
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "60周岁"
}
$$, 0, $$
{
    "minAmount": 4974,
    "unit": "元"
}
$$, $$
{
    "minAmount": 600000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/cf1WB2a6uit0', NULL, NULL, '暂无', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        5,
        10,
        18,
        20,
        30
    ]
}
$$, '50000元保额起，10000元整数倍递增', '暂无', '可附加住院医疗，意外医疗', '1、重疾不分组，无三同
2、核保宽松，品牌好，增值服务好
3、自带少儿特定疾病和成人特定疾病额外赔付
4、支持预核保',2288.04,0.46,0.13,0.07,0.05,0.02,80.55);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '长城人寿', '吉康人生重大疾病保险（2024）', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 128,
        "maxPayouts": 3,
        "groups": 0,
        "stepPayout": "间隔期1年，依次给付100%/120%/150%基本保额"
    },
    "中症": {
        "count": 20,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "轻症": {
        "count": 50,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "55周岁"
}
$$, 0, $$
{
    "minAmount": 7230,
    "unit": "元"
}
$$, $$
{
    "text": "暂无"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/ckFRDySeBEuK', $$
{
    "其他": {}
}
$$, $$
{
    "重疾": {}
}
$$, '暂无', $$
{
    "其他": {}
}
$$, '100000元保额起，10000元整数倍递增', '暂无', '可附加住院医疗，意外医疗', '1、轻中重症不分组，无三同
2、可选保单前30年轻中重额外赔付',3542.7,0.49,0.23,0.06,0.04,0.015,79.35);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '昆仑健康', '健康保青春多倍版重大疾病保险', '多次赔付重疾险', $$
{
    "产品责任": {
        "count": 100,
        "maxPayouts": 3,
        "groups": 0,
        "firstPayoutBasis": "间隔期1年，首次重疾60岁前给付160%保额/100%累计保费/现金价值取大"
    },
    "中症": {
        "count": 25,
        "maxPayouts": 2,
        "groups": 0,
        "firstPayoutBasis": "首次中症60岁前给付90%基本保额，60岁后给付60%基本保额"
    },
    "轻症": {
        "count": 50,
        "maxPayouts": 3,
        "groups": 0,
        "firstPayoutBasis": "首次轻症60岁前给付45%基本保额，60岁后给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "出生满28天-35周岁"
}
$$, 0, $$
{
    "minAmount": 6720,
    "unit": "元"
}
$$, $$
{
    "minAmount": 500000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/cgHj7fXMe0DK', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '暂无', $$
{
    "其他": {}
}
$$, '保终身300000元起，保至70周岁400000元起，最高1000000元', '暂无', '暂无', '1、轻中重均有三同
2、可选保定期，降低预算
3、可不含身故责任，拉高性价比
4、自带轻中重症60岁前额外赔付，可选癌症多次和心脑血管疾病多次，保障足够全面
5、只保0-35周岁，35岁以上不可购买',1747.2,0.26,0.21,0.09,0.04,0.015,70.05);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '昆仑健康', '健康保普惠多倍版重大疾病保险', '多次赔付重疾险', $$
{
    "产品责任": {
        "count": 100,
        "maxPayouts": 2,
        "groups": 0,
        "firstPayoutBasis": "间隔期1年，首次重疾15保单周年日前给付150%保额/100%累计保费/现金价值取大"
    },
    "中症": {
        "count": 25,
        "maxPayouts": 2,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "轻症": {
        "count": 50,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "出生满28天-45周岁"
}
$$, 0, $$
{
    "minAmount": 5817,
    "unit": "元"
}
$$, $$
{
    "minAmount": 500000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/cqG6jKsRTGx5', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '暂无', $$
{
    "其他": {}
}
$$, '保终身300000元起，保至70周岁400000元起，最高1000000元', '暂无', '暂无', '1、轻中重均有三同
2、责任比青春版略低',1512.42,0.26,0.21,0.09,0.04,0.015,70.05);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '长生人寿', '福安康重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 100,
        "maxPayouts": 2,
        "groups": 0,
        "eachPayout": "间隔期365天，每次给付100%基本保额"
    },
    "中症": {
        "count": 20,
        "maxPayouts": 2,
        "groups": 0,
        "stepPayout": "依次给付50%/60%基本保额"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "出生满30天-65周岁"
}
$$, 0, $$
{
    "minAmount": 5592,
    "unit": "元"
}
$$, $$
{
    "minAmount": 1000000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/cviZK21pYqYz', $$
{
    "其他": {}
}
$$, NULL, '暂无', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        5,
        10,
        15,
        20,
        30
    ]
}
$$, '10000元保额起，10000元整数倍递增', '暂无', '暂无', NULL,2740.08,0.49,0.23,0.06,0.04,0.02,72.45);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '利安人寿', '健利保（馨享版）重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "间隔期365天，每次给付100%基本保额"
    },
    "中症": {
        "count": 20,
        "maxPayouts": 6,
        "groups": 0,
        "interval": "180天",
        "eachPayout": "每次给付60%基本保额（同因疾病间隔180天，不同因无间隔）"
    },
    "轻症": {
        "count": 50,
        "maxPayouts": 6,
        "groups": 0,
        "interval": "180天",
        "eachPayout": "每次给付30%基本保额（同因疾病间隔180天，不同因无间隔）"
    }
}
$$, NULL, NULL, NULL, $$
{
    "min": "28天",
    "max": "55周岁"
}
$$, 0, $$
{
    "minAmount": 7260,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/ccMJ977BMtD1', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '暂无', $$
{
    "其他": {}
}
$$, '最低10000元起，10000元整数倍递增', '暂无', '住院医疗', '1、重疾不分组，有三同
2、可选少儿，成人和老年特定疾病双倍赔付，全年龄段特定疾病双倍保障',3375.9,0.465,0.23,0.09,0.07,0.025,76.05);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '中华联合人寿', '中华福（优享版）终身重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "maxPayouts": 2,
        "groups": 0,
        "eachPayout": "间隔期365天，每次给付100%基本保额"
    },
    "中症": {
        "count": 15,
        "maxPayouts": 2,
        "groups": 0,
        "eachPayout": "间隔期90天，每次给付50%基本保额"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 4,
        "groups": 0,
        "eachPayout": "间隔期90天，每次给付25%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "60周岁"
}
$$, 0, $$
{
    "minAmount": 5235,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/ccMJ977BMtD1', $$
{
    "其他": {}
}
$$, NULL, '暂无', $$
{
    "其他": {}
}
$$, '最低保额100000元起，10000元整数倍递增', '暂无', NULL, NULL,2826.9,0.54,0.19,0.045,0.025,0.025,78.15);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '阳光人寿', '倍享阳光关爱多重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "maxPayouts": 1,
        "groups": 0
    },
    "中症": {
        "count": 30,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付50%基本保额"
    },
    "轻症": {
        "count": 30,
        "maxPayouts": 6,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "出生满28天-60周岁"
}
$$, 0, $$
{
    "minAmount": 3720,
    "unit": "元"
}
$$, $$
{
    "minAmount": 900000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/cl9lzY599X5z', $$
{
    "其他": {}
}
$$, NULL, '不附加', $$
{
    "其他": {}
}
$$, '最低保额100000元，10000元整数倍递增', '可附加', '可附加住院医疗，住院津贴，意外伤害保险', '1、可选60岁前重疾和中症双倍赔付，中症也可以赔付100%保额
2、自带少儿特定疾病双倍赔付，18岁前，罹患少儿特定疾病可赔付3倍保额
3、可选身故赔付保额&保费，提高性价比
4、可选癌症二次赔付
5、卫健委合作的直通30绿通服务',1748.4,0.47,0.2,0.06,0.02,0.02,79.35);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '中韩人寿', '一生无忧终身重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 34,
        "maxPayouts": 1,
        "groups": 0
    }
}
$$, NULL, NULL, NULL, $$
{
    "min": "28天",
    "max": "60周岁"
}
$$, 0, $$
{
    "minAmount": 3990,
    "unit": "元"
}
$$, $$
{
    "text": "暂无"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/cj0zVFOIAxpG', NULL, NULL, '暂无', $$
{
    "其他": {}
}
$$, '100000元保额起，10000元整数倍递增', '暂无', '暂无', '1、重疾种类34种，但价格较低',1835.4,0.46,0.1,0.025,0.015,0.015,56.7);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '复星联合健康', '康乐爱相守2.0重大疾病保险（单人版）', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "maxPayouts": 1,
        "groups": 0
    },
    "中症": {},
    "轻症": {}
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "55周岁"
}
$$, 0, $$
{
    "minAmount": 3783,
    "unit": "元"
}
$$, $$
{
    "minAmount": 200000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/cg4Xo9zUvVa6', $$
{
    "其他": {}
}
$$, NULL, '暂无', $$
{
    "其他": {}
}
$$, '50000元保额起，10000元整数倍递增', '暂无', '暂无', '1、单次重疾，价格相对较低
2、未成年人身故全残也是赔付保额',2118.48,0.56,0.2,0.025,0.015,0.01,70.5);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '复星联合健康', '康乐爱相守2.0重大疾病保险（双人版）', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "maxPayouts": 1,
        "groups": 0
    },
    "中症": {},
    "轻症": {}
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "55周岁"
}
$$, 0, $$
{
    "minAmount": 4614,
    "unit": "元"
}
$$, $$
{
    "minAmount": 200000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/cg4Xo9zUvVa6', $$
{
    "其他": {}
}
$$, NULL, '暂无', $$
{
    "其他": {}
}
$$, '50000元保额起，10000元整数倍递增', '暂无', '暂无', '1、单次重疾，价格相对较低
2、未成年人身故全残也是赔付保额
3、创新双被保人，一定条件下，双被保人均可获得1次重疾赔付，但是如果一方理赔了身故或者重疾，另一方保障到61岁截止，需要仔细解释',2583.84,0.56,0.2,0.025,0.015,0.01,66.9);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '中意人寿', '悦享安康重大疾病保险（惠选版）', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 128,
        "maxPayouts": 1,
        "groups": 0
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 2,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "出生满7天-67周岁"
}
$$, 0, $$
{
    "text": "5044.89"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '保至80周岁', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/cueELCu7FsNB', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '暂无', $$
{
    "其他": {}
}
$$, '一类地区200000元起售，二类地区100000元起售，10000元整数倍递增', '暂无', '可附加意外和住院医疗', NULL,2144.08,0.425,0.3,0.05,0.045,0.025,70.65);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '复星联合健康', '康乐一生（易核版3.0）重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 28,
        "maxPayouts": 1,
        "groups": 0
    },
    "轻症": {
        "count": 3,
        "maxPayouts": 1,
        "groups": 0
    }
}
$$, NULL, NULL, NULL, $$
{
    "min": "30周岁",
    "max": "65周岁"
}
$$, 0, $$
{
    "minAmount": 0,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/cn664wqbZY5J', $$
{
    "其他": {}
}
$$, NULL, '暂无', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        5,
        10,
        15,
        20
    ]
}
$$, '最低保额50000元起，10000元整数倍递增', '暂无', '暂无', NULL,0,0.465,0.23,0.025,0.015,0.01,66.45);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '百年人寿', '康享保终身重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 28,
        "maxPayouts": 1,
        "groups": 0
    },
    "轻症": {
        "count": 3,
        "maxPayouts": 1,
        "groups": 0
    }
}
$$, NULL, NULL, NULL, $$
{
    "min": "18",
    "max": "60周岁"
}
$$, 0, $$
{
    "minAmount": 0,
    "unit": "元"
}
$$, $$
{
    "minAmount": 720000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/cn664wqbZY5J', NULL, NULL, '暂无', $$
{
    "其他": {}
}
$$, '最低保额50000元起，1千整数倍递增', '一年期投保人豁免', '可附加住院医疗，意外', '1、保28种行业规定重疾和3种轻症，免体检保额相对较高，适合组合搭配',0,0.365,0.15,0.02,0.015,0.015,67.5);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '中华联合人寿', '中华悦（随心版）重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 130,
        "maxPayouts": 1,
        "groups": 0
    }
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "60周岁"
}
$$, 0, $$
{
    "minAmount": 3429,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/ckBeev3Godge', NULL, NULL, '暂无', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        5,
        10,
        15,
        20,
        30
    ]
}
$$, '最低保额100000元起，10000元整数倍递增', NULL, NULL, NULL,1680.21,0.49,0.19,0.045,0.025,0.025,69.3);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '中汇人寿', '健康源（汇享）终身重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "maxPayouts": 1,
        "groups": 0
    },
    "中症": {
        "count": 20,
        "maxPayouts": 2,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 6,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "min": "28天",
    "max": "60周岁"
}
$$, 0, $$
{
    "minAmount": 4623,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/cg1KiUGgDMWY', NULL, NULL, '暂无', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        3,
        5,
        10,
        15,
        20,
        30
    ]
}
$$, '最低保额100000元起，10000元整数倍递增', '暂无', NULL, NULL,1802.97,0.39,0.16,0.015,0.005,0,76.5);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '招商信诺人寿', '信享宝倍少儿重大疾病保险', '少儿重疾险', $$
{
    "产品责任": {
        "count": 138,
        "groups": 3,
        "maxPayouts": 3,
        "eachPayout": "间隔期1年，每次给付100%基本保额"
    },
    "中症": {
        "count": 30,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "轻症": {
        "count": 52,
        "maxPayouts": 6,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "出生满28天-17周岁"
}
$$, 0, $$
{
    "minAmount": 6075,
    "unit": "元"
}
$$, $$
{
    "minAmount": 880000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/cuRlBlvLpwbj', $$
{
    "其他": {
        "firstPayoutBasis": "首次重大疾病保险金、多次给付重大疾病保险金、重大疾病豁免保险费、首次重大疾病关爱保险金、青少年特定疾病或罕见疾病保险金、中症疾病保险金、轻症疾病保险金、中症疾病或轻症疾病豁免保险费、身故保险金、身故慰问保险金"
    }
}
$$, NULL, '暂无', $$
{
    "其他": {}
}
$$, '100000元保额起，10000元整数倍递增', '暂无', '暂无', '1、可选单次/多次，可选轻中症
2、可选30岁前少儿特定疾病和罕见疾病额外双倍赔付
3、可选30岁前首次重疾额外双倍赔付，30岁前，首次重疾最高可赔付5倍基本保额',2490.75,0.41,0.34,0.09,0.01,0.01,86.55);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '复星联合健康', '妈咪保贝（星礼版2.0）少儿重大疾病保险（少儿计划）', '少儿重疾险', $$
{
    "产品责任": {
        "count": 120,
        "maxPayouts": 3,
        "groups": 0,
        "stepPayout": "间隔期180天，依次给付100%/120%/120%基本保额"
    },
    "中症": {
        "count": 30,
        "maxPayouts": 2,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "轻症": {
        "count": 45,
        "maxPayouts": 4,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "出生满28天-17周岁"
}
$$, 0, $$
{
    "text": "5627.22"
}
$$, $$
{
    "minAmount": 500000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/cn664wqbZY5J', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '不附加', $$
{
    "annualPaymentYears": [
        30,
        5,
        10,
        15,
        20
    ]
}
$$, '100000元-700000元，1千整数倍递增', '可附加，投保人轻中重，身故全残豁免', '无', '1、可选单次/多次赔付
2、可选身故赔付保费/赔付保额，赔付105%累计保费
3、自带20种少儿特定重疾和10种少儿罕见病额外赔付
4、可选30周岁前轻中重额外赔付',2644.79,0.47,0.04,0.015,0.005,0.005,79.05);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '君龙人寿', '小青龙5号少儿重大疾病保险（互联网）', '少儿重疾险', $$
{
    "产品责任": {
        "count": 128,
        "maxPayouts": 4,
        "groups": 0,
        "stepPayout": "间隔期365天，依次给付100%/120%/140%/160%基本保额"
    },
    "中症": {
        "count": 30,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额",
        "interval": "90天"
    },
    "轻症": {
        "count": 52,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额",
        "interval": "90天"
    }
}
$$, NULL, NULL, NULL, $$
{
    "min": "28天",
    "max": "17周岁"
}
$$, 0, $$
{
    "minAmount": 5622,
    "unit": "元"
}
$$, $$
{
    "minAmount": 800000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/clcoelP67zsC', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '不附加', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        5,
        10,
        15,
        20,
        30
    ]
}
$$, '100000元保额起，10000元整数倍递增', '身故/全残/轻中重症可豁免', NULL, '1、自带少儿特定疾病额外1.2倍给付和罕见病额外2倍给付，无年龄限制
2、保障全面，超高性价比',1461.72,0.26,0.16,0.09,0.045,0.03,83.55);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '君龙人寿', '小青龙6号少儿重大疾病保险（互联网）', '少儿重疾险', $$
{
    "产品责任": {
        "count": 128,
        "maxPayouts": 6,
        "groups": 0,
        "stepPayout": "间隔期365天，依次给付100%/120%/140%/160%/160%/160%基本保额"
    },
    "中症": {
        "count": 30,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额",
        "interval": "90天"
    },
    "轻症": {
        "count": 52,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额",
        "interval": "90天"
    }
}
$$, NULL, NULL, NULL, $$
{
    "min": "14天",
    "max": "17周岁"
}
$$, 0, $$
{
    "minAmount": 6084,
    "unit": "元"
}
$$, $$
{
    "minAmount": 1000000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/cpDQxbefUcf9', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '不附加', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        5,
        10,
        15,
        20,
        30
    ]
}
$$, '100000元保额起，10000元整数倍递增', '身故/全残/轻中重症可豁免', NULL, '1、自带少儿特定疾病额外1.2倍给付和罕见病额外2倍给付，无年龄限制
2、保障全面，超高性价比
3、3岁前，因符合约定的新发先天性疾病导致的重疾，可以赔付',1642.68,0.27,0.16,0.09,0.045,0.03,83.55);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '中国人民保险寿险', 'i无忧3.0重大疾病保险（互联网专属）', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 125,
        "maxPayouts": 1,
        "groups": 0
    },
    "中症": {
        "count": 20,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 5,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "min": "28天",
    "max": "55周岁"
}
$$, 0, $$
{
    "minAmount": 6960,
    "unit": "元"
}
$$, $$
{
    "minAmount": 900000,
    "unit": "元"
}
$$, '保终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/ccFqYB17MEPu', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '暂无', $$
{
    "其他": {}
}
$$, '50000元保额起', '无', '无', '1、可选身故责任和轻中症责任
2、智能核保，健告宽松，乙肝，甲状腺结节，乳房结节，胃或肠道息肉，子宫肌瘤，进追捕，1级高血压等疾病符合条件有机会投保',1496.4,0.215,0.19,0.1,0.05,0.025,79.95);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '瑞华健康', '达尔文10号（福瑞保3.0）', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 110,
        "maxPayouts": 1,
        "groups": 0,
        "interval": "365天"
    },
    "中症": {
        "count": 35,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "重疾": {},
    "轻症": {
        "count": 40,
        "maxPayouts": 4,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "出生满28天-55周岁"
}
$$, 0, $$
{
    "text": "5675.3551416"
}
$$, $$
{
    "minAmount": 1000000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/ch7mFVENLLD4', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '附加', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        5,
        10,
        15,
        20,
        30,
        35
    ]
}
$$, '50000元保额起，10000元整数倍递增', '可附加', '暂无', '1、8项可选，搭配更灵活
2、创新孕期发生重疾额外赔付
3、重疾赔付后，轻中症继续有效且没有分组限制
4、核保尺度放宽，健告不在问询既往限额承保史，智能核保放款甲状腺功能异常，肺部结节，肾结晶/肾结石，卵巢囊肿，HPV阳性，慢性宫颈炎CIN/TCT异常，尿酸升高',1521.3,0.275,0.15,0.05,0,0,81.75);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '复星联合健康', '达尔文11号重大疾病保险（互联网）', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "maxPayouts": 1,
        "groups": 0,
        "interval": "1095天"
    },
    "中症": {
        "count": 30,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "重疾": {},
    "轻症": {
        "count": 45,
        "maxPayouts": 4,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "出生满28天-55周岁"
}
$$, 0, $$
{
    "text": "5608.62357"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/chqgXdHK3u2c', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '附加', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        5,
        10,
        15,
        20,
        30
    ]
}
$$, '50000元保额起，50000元整数倍递增,最高500000元', '可附加', '暂无', '1、7项可选，搭配更灵活
2、创新孕期发生重疾额外赔付
3、重疾赔付后，轻中症继续有效且没有分组限制，无间隔期',1367.25,0.25,0.17,0.06,0,0,82.65);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '君龙人寿', '超级玛丽12号（健康美满B款）重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 110,
        "maxPayouts": 2,
        "groups": 0,
        "firstPayoutBasis": "首次给付100%基本保额"
    },
    "中症": {
        "count": 35,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "重疾": {},
    "轻症": {
        "count": 40,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "出生满28天-50周岁"
}
$$, 0, $$
{
    "minAmount": 6033,
    "unit": "元"
}
$$, $$
{
    "minAmount": 500000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/cl9E7fkxFJeT', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '不附加', $$
{
    "其他": {}
}
$$, '最低50000元保额，以10000元整数倍递增', '可附加', '暂无', '1、基础责任单次赔付，提高性价比，附加责任全面
2、癌症多次给付可选不限次赔付和三次赔付，赔付间隔短
3、创新肺结节责任+一站式健康管理
4、同种重疾也可以二次赔付',1447.92,0.24,0.24,0.07,0,0,81.75);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '君龙人寿', '超级玛丽13号（健康美满C款）重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 110,
        "maxPayouts": 1,
        "groups": 0
    },
    "中症": {
        "count": 35,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "重疾": {},
    "轻症": {
        "count": 40,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "出生满28天-50周岁"
}
$$, 0, $$
{
    "minAmount": 5631,
    "unit": "元"
}
$$, $$
{
    "minAmount": 500000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/cfYq3KhSU8mg', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '不附加', $$
{
    "其他": {}
}
$$, '最低50000元保额，以10000元整数倍递增', '可附加', '暂无', '1、基础责任单次赔付，提高性价比，附加责任全面
2、癌症多次给付可选不限次赔付和三次赔付，赔付间隔短
3、创新肺结节责任+一站式健康管理
4、同种重疾也可以二次赔付',1351.44,0.24,0.24,0.07,0,0,81.75);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '北京人寿', '大黄蜂12号（京康宝贝S/T/U款）少儿重疾险（焕新版）', '少儿重疾险', $$
{
    "产品责任": {
        "count": 125,
        "maxPayouts": 4,
        "groups": 0,
        "stepPayout": "间隔期365天，依次给付100%/120%/130%/150%基本保额"
    },
    "中症": {
        "count": 30,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "重疾": {},
    "轻症": {
        "count": 43,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "出生满28天-17周岁"
}
$$, 0, $$
{
    "minAmount": 4689,
    "unit": "元"
}
$$, $$
{
    "minAmount": 700000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/clH1iseSKYiI', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '不附加', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        5,
        10,
        15,
        20,
        30,
        35
    ]
}
$$, '100000元保额起，100000元递增，最高700000元（保30年最低300000元）', '轻/中/重症/身故/全残可附加', NULL, '1、自带少儿特定重疾和罕见病额外给付，可选特定疾病额外多次给付
2、可选60周岁前轻中重疾额外给付，60周岁前，第2个保单周年日后，罹患少儿特定疾病最高给付360%基本保额（100%+80%+130%+50%），罹患少儿罕见病最高给付440%基本保额（100%+80%+210%+50%）
3、首次重疾给付后，轻中症未给付满6次，非同组轻中症可给继续给付
4、癌症多次给付不限次数，直至保单终止
5、重疾不分组，无三同',1242.59,0.265,0.26,0.06,0,0,85.5);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '招商仁和', '青云卫5号A款少儿重大疾病保险（互联网）', '少儿重疾险', $$
{
    "产品责任": {
        "count": 137,
        "maxPayouts": 4,
        "groups": 0,
        "stepPayout": "间隔期365天，依次给付100%/120%/130%/150%基本保额"
    },
    "中症": {
        "count": 30,
        "maxPayouts": 2,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "重疾": {},
    "轻症": {
        "count": 51,
        "maxPayouts": 5,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "出生满28天-17周岁"
}
$$, 0, $$
{
    "minAmount": 4938,
    "unit": "元"
}
$$, $$
{
    "minAmount": 800000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/csL1cmOpRLbw', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '不附加', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        10,
        15,
        20,
        30,
        35
    ]
}
$$, '（保至70周岁/终身）100000元保额起，以10000元的整数倍递增', '轻/中/重症/身故/全残可附加', NULL, '1、重疾不分组，无三同
2、重疾赔付之后，非同组轻中症继续有效
3、自带20种少儿特定疾病和20种少儿罕见病额外赔付，无年龄限制
4、可选重疾多次给付，轻中重额外给付，癌症多次给付，组合更灵活
5、自带白血病和严重肥胖手术关爱金和健康管理服务
6、可选60周岁前轻中重疾病额外给付',1333.26,0.27,0.175,0.04,0.02,0.02,90.3);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '北京人寿', '小福娃少儿重疾保险计划（互联网专属）', '少儿重疾险', $$
{
    "产品责任": {
        "count": 125,
        "maxPayouts": 4,
        "groups": 0,
        "stepPayout": "间隔期365天，依次给付100%/120%/140%/160%基本保额"
    },
    "中症": {
        "count": 30,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "重疾": {},
    "轻症": {
        "count": 43,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "出生满28天-17周岁"
}
$$, 0, $$
{
    "text": "4684.5"
}
$$, $$
{
    "minAmount": 700000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/cefmeeRdE98l', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '不附加', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        5,
        10,
        15,
        20,
        30,
        35
    ]
}
$$, '100000元保额起', '轻/中/重症/身故/全残可附加', NULL, '1、自带少儿特定重疾和罕见病额外给付
2、可选60周岁前轻中重疾额外给付
3、首次重疾给付后，轻中症未给付满6次，非同组轻中症可给继续给付
4、重疾不分组，无三同',1217.97,0.26,0.19,0.06,0,0,85.5);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '国富人寿', '小红花2025重大疾病保险（互联网专属）', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 110,
        "maxPayouts": 2,
        "groups": 0,
        "firstPayoutBasis": "首次重疾给付100%基本保额/100%累计保费/现金价值取大"
    },
    "中症": {
        "count": 35,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "重疾": {},
    "轻症": {
        "count": 40,
        "maxPayouts": 4,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "min": "28天",
    "max": "55周岁"
}
$$, 0, $$
{
    "minAmount": 5343,
    "unit": "元"
}
$$, $$
{
    "minAmount": 700000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/cnAPk7oLjP2h', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '不附加', $$
{
    "其他": {}
}
$$, '0-45岁，未选身故全残最低200000元，选择身故全残最低100000元/46-55岁，最低50000元，10000元递增', '可附加（基础责任重疾豁免保费，可选轻中症，身故全残豁免保费，费率默认附加）', '暂无', '1、重疾不分组，无三同
2、可选60周岁前轻中重症额外赔付
3、可不捆绑身故责任，单次赔付性价比很高',1389.18,0.26,0.22,0.06,0,0,82.95);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '复星保德信人寿', '大黄蜂13号少儿重大疾病保险（旗舰版）', '少儿重疾险', $$
{
    "产品责任": {
        "count": 125,
        "maxPayouts": 4,
        "groups": 0,
        "stepPayout": "间隔期365天，依次给付100%/120%/140%/160%基本保额"
    },
    "中症": {
        "count": 30,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "重疾": {},
    "轻症": {
        "count": 43,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "出生满28天-17周岁"
}
$$, 0, $$
{
    "minAmount": 4770,
    "unit": "元"
}
$$, $$
{
    "minAmount": 800000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/cvnSc2tM06Ga', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '不附加', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        5,
        10,
        15,
        20,
        30,
        35
    ]
}
$$, '100000元保额起，100000元递增，最高800000元', '轻/中/重症/身故/全残可附加', NULL, NULL,1216.35,0.255,0.19,0.06,0,0,86.7);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '北京人寿', '大黄蜂13号少儿重疾险（全能版）', '少儿重疾险', $$
{
    "产品责任": {
        "count": 125,
        "maxPayouts": 4,
        "groups": 0,
        "stepPayout": "间隔期365天，依次给付100%/120%/140%/160%基本保额"
    },
    "中症": {
        "count": 30,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "重疾": {},
    "轻症": {
        "count": 43,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "出生满28天-17周岁"
}
$$, 0, $$
{
    "minAmount": 4671,
    "unit": "元"
}
$$, $$
{
    "minAmount": 700000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/cvnSc2tM06Ga', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '不附加', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        5,
        10,
        15,
        20,
        29,
        30,
        35
    ]
}
$$, '100000元保额起，100000元递增，最高700000元（保30年最低300000元）', '轻/中/重症/身故/全残可附加', NULL, '1、自带少儿特定重疾和罕见病额外给付，可选特定疾病额外多次给付
2、可选60周岁前轻中重疾额外给付
3、首次重疾给付后，轻中症未给付满6次，非同组轻中症可给继续给付
4、癌症多次给付不限次数，直至保单终止
5、重疾不分组，无三同',1191.11,0.255,0.25,0.06,0,0,85.5);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '君龙人寿', '守卫者7号', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "maxPayouts": 6,
        "groups": 0,
        "eachPayout": "间隔期365天，每次给付100%基本保额/累计保费/现价取大"
    },
    "中症": {
        "count": 35,
        "maxPayouts": 6,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 6,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "min": "28天",
    "max": "50周岁"
}
$$, 0, $$
{
    "minAmount": 5718,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/ciaOZwrYsJP7', $$
{
    "轻症": {}
}
$$, $$
{
    "其他": {}
}
$$, '暂无', $$
{
    "其他": {}
}
$$, '100000元保额起，50000元递增', NULL, NULL, NULL,1343.73,0.235,0.18,0.07,0,0,80.55);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '复星联合健康', '妈咪保贝爱常在少儿重大疾病保险（A款）（互联网）', '少儿重疾险', $$
{
    "产品责任": {
        "count": 135,
        "maxPayouts": 4,
        "groups": 0,
        "stepPayout": "依次给付100%/120%/140%/160%基本保额"
    },
    "中症": {
        "count": 30,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "重疾": {},
    "轻症": {
        "count": 50,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "出生满28天-17周岁"
}
$$, 0, $$
{
    "text": "5822.8"
}
$$, $$
{
    "minAmount": 800000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/cjRWZwe6pGwL', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '不附加', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        10,
        15,
        20,
        30,
        35
    ]
}
$$, '（保至70周岁/终身）100000元保额起，以10000元的整数倍递增', '轻/中/重症/身故/全残可附加', NULL, '1、重疾不分组，无三同
2、重疾赔付之后，非同组轻中症继续有效，且无间隔
3、自带20种少儿特定疾病和20种少儿罕见病额外赔付，无年龄限制
4、可选重疾多次给付，轻中重额外给付，癌症多次给付，组合更灵活
5、可附加关爱金，投保人出险，被保人保障翻倍
6、可选60周岁前轻中重疾病额外给付',1659.5,0.285,0.27,0.04,0.005,0,85.35);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '中华联合财险', '中华联合百万重疾（互联网版）', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "groups": 6,
        "maxPayouts": 6,
        "eachPayout": "间隔期180天，每次给付150000元"
    },
    "中症": {
        "count": 28,
        "maxPayouts": 1,
        "groups": 0
    },
    "轻症": {
        "count": 45,
        "maxPayouts": 1,
        "groups": 0
    }
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "60周岁"
}
$$, 0, $$
{
    "minAmount": 170,
    "unit": "元"
}
$$, $$
{
    "minAmount": 150000,
    "unit": "元"
}
$$, '1年', '90天', '1年', NULL, NULL, NULL, NULL, NULL, '无', $$
{
    "其他": {}
}
$$, '150000元', '无', '无', NULL,35.7,0.21,NULL,NULL,NULL,NULL,88.05);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '华贵保险', '华贵麦兜兜2024少儿重大疾病保险（互联网）', '少儿重疾险', $$
{
    "产品责任": {
        "count": 120
    }
}
$$, NULL, NULL, NULL, $$
{
    "min": "28天",
    "max": "17周岁"
}
$$, 0, $$
{
    "minAmount": 536,
    "unit": "元"
}
$$, $$
{
    "minAmount": 1000000,
    "unit": "元"
}
$$, '30年', '180天', '30年', NULL, NULL, NULL, $$
{
    "其他": {}
}
$$, NULL, '无', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        5,
        10,
        20,
        30
    ]
}
$$, '1000000元', '无', '无', NULL,147.4,0.275,0.2,0.08,0,0,75.45);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '众安保险', '众民保·重疾险（简易健告）', '单次赔付重疾险', $$
{
    "产品责任": {
        "groups": 5,
        "maxPayouts": 5
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "出生满28天-70周岁"
}
$$, 0, $$
{
    "minAmount": 294,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '1年', '90天', '1年', NULL, NULL, '支持家庭单，2人95折，3人9折', NULL, NULL, '无', $$
{
    "其他": {}
}
$$, '50000元保额起', '无', '无', NULL,73.5,0.25,NULL,NULL,NULL,NULL,79.05);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '众安保险', '众民保·百万重疾险（免健告）', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 160,
        "maxPayouts": 2,
        "groups": 0,
        "eachPayout": "间隔期180天，每次给付100%基本保额"
    },
    "轻症": {
        "count": 60,
        "maxPayouts": 1,
        "groups": 0
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "出生满28天-70周岁"
}
$$, 0, $$
{
    "text": "523.8"
}
$$, $$
{
    "minAmount": 500000,
    "unit": "元"
}
$$, '1年', '90天', '1年', NULL, NULL, NULL, $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '无', $$
{
    "其他": {}
}
$$, '50000元保额起', '无', '无', NULL,130.95,0.25,NULL,NULL,NULL,NULL,98.25);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '招商仁和', '小淘气5号少儿重大疾病保险', '少儿重疾险', $$
{
    "产品责任": {
        "count": 137,
        "maxPayouts": 4,
        "groups": 0,
        "stepPayout": "间隔期365天，依次给付100%/120%/150%/150%基本保额"
    },
    "中症": {
        "count": 30,
        "maxPayouts": 2,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "重疾": {},
    "轻症": {
        "count": 51,
        "maxPayouts": 5,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 5865,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/ckJpP7ZIXWc5', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '出生满28天-17周岁', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        5,
        10,
        20,
        30,
        35
    ]
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,66.7);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '中国人寿', '康宁惠享终身重大疾病保险（2024版）', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 28,
        "maxPayouts": 1,
        "groups": 0
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 3990,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/coJJTr4m11a5', NULL, NULL, '出生满28天-60周岁', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        3,
        5,
        10,
        20
    ]
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,62.85);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '中国人寿', '康宁尊享重大疾病保险（2024版）', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "groups": 6,
        "maxPayouts": 6,
        "cancerSeparateGroup": true,
        "eachPayout": "第2-6次，88岁前每次给付100%基本保额"
    },
    "中症": {
        "count": 20,
        "maxPayouts": 2,
        "groups": 0,
        "eachPayout": "每次给付50%基本保额"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 6,
        "groups": 0,
        "eachPayout": "每次给付20%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 6270,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '180天', '19年', NULL, NULL, 'https://kdocs.cn/l/cp5lIyn2eBHw', $$
{
    "其他": {}
}
$$, $$
{
    "重疾": {}
}
$$, '出生满28天-62周岁', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        3,
        5,
        10,
        15,
        19
    ]
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,86.25);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '中国平安寿险', '少儿鑫福星（2025）重大疾病保险', '少儿重疾险', $$
{
    "产品责任": {
        "count": 120,
        "maxPayouts": 1,
        "groups": 0
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 4200,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/cnbgaX8IhKjZ', NULL, NULL, '出生满28天-17周岁', $$
{
    "其他": {}
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,75.45);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '中国平安寿险', '鑫福星（2025）重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "maxPayouts": 1,
        "groups": 0
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 0,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/chcnF3Jsua4g', NULL, NULL, '18-55周岁', $$
{
    "其他": {}
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,75.45);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '中国平安寿险', '少儿平安如意全能（2025）保障计划', '少儿重疾险', $$
{
    "产品责任": {
        "count": 120,
        "maxPayouts": 1,
        "groups": 0
    },
    "中症": {
        "count": 20,
        "maxPayouts": 5,
        "groups": 0,
        "eachPayout": "每次给付50%基本保额"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 5,
        "groups": 0,
        "eachPayout": "每次给付20%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 6150,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '保至80周岁', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/cuW3msoZVe5J', NULL, NULL, '出生满28天-17周岁', $$
{
    "其他": {}
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,91.05);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '中国平安寿险', '平安如意全能（2025）保障计划', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "maxPayouts": 1,
        "groups": 0
    },
    "中症": {
        "count": 20,
        "maxPayouts": 5,
        "groups": 0,
        "eachPayout": "每次给付50%基本保额"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 5,
        "groups": 0,
        "eachPayout": "每次给付20%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 0,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '保至60周岁', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/csPIjmi5WXVE', NULL, NULL, '18-54周岁', $$
{
    "其他": {}
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,86.25);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '中国平安寿险', '少儿守护百分百（2025）', '少儿重疾险', $$
{
    "产品责任": {
        "count": 120,
        "maxPayouts": 1,
        "groups": 0
    },
    "中症": {
        "count": 20,
        "maxPayouts": 5,
        "groups": 0,
        "eachPayout": "每次给付50%基本保额"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 5,
        "groups": 0,
        "eachPayout": "每次给付20%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 5070,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '保至80周岁', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/cm4Mv52PGbfX', NULL, NULL, '出生满28天-17周岁', $$
{
    "其他": {}
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,91.05);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '中国平安寿险', '守护百分百（2025）', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "maxPayouts": 1,
        "groups": 0
    },
    "中症": {
        "count": 20,
        "maxPayouts": 5,
        "groups": 0,
        "eachPayout": "每次给付50%基本保额"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 5,
        "groups": 0,
        "eachPayout": "每次给付20%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 0,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '保至80周岁', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/cp9b3aEmJryw', NULL, NULL, '18-54周岁', $$
{
    "其他": {}
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,86.25);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '友邦保险', '友如意顺心版（2024）重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "groups": 5,
        "maxPayouts": 5,
        "cancerSeparateGroup": true,
        "eachPayout": "每次给付100%基本保额"
    },
    "中症": {
        "count": 25,
        "maxPayouts": 5,
        "groups": 0,
        "eachPayout": "每次给付40%基本保额"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 5,
        "groups": 0,
        "eachPayout": "每次给付20%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 0,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/coS4E46CzcfP', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '18-55周岁', $$
{
    "其他": {}
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,87.15);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '友邦保险', '友如意星享版（2024）重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "groups": 5,
        "maxPayouts": 5,
        "cancerSeparateGroup": true,
        "eachPayout": "每次给付100%基本保额"
    },
    "中症": {
        "count": 25,
        "maxPayouts": 5,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 5,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "text": "6496.2"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/cunNhVkTKmS3', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '出生满7天-17周岁', $$
{
    "其他": {}
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,87.15);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '友邦保险', '如意双全安心版保险产品计划', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "groups": 5,
        "maxPayouts": 1
    },
    "中症": {
        "count": 25,
        "maxPayouts": 5,
        "groups": 0,
        "eachPayout": "每次给付40%基本保额"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 5,
        "groups": 0,
        "eachPayout": "每次给付20%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 0,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '至65周岁', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/cpkjz3MsJZ2u', $$
{
    "其他": {}
}
$$, NULL, '18-45周岁', $$
{
    "其他": {}
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,87.15);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '友邦保险', '如意双全星悦版保险产品计划', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "groups": 5,
        "maxPayouts": 1
    },
    "中症": {
        "count": 25,
        "maxPayouts": 5,
        "groups": 0,
        "eachPayout": "每次给付40%基本保额"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 5,
        "groups": 0,
        "eachPayout": "每次给付20%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "text": "3891.71"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '至65周岁', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/crFp8C3igRs3', $$
{
    "其他": {}
}
$$, NULL, '出生满7天-17周岁', $$
{
    "其他": {}
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,87.15);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '太平洋保险', '金生无忧2024（成人版）重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "maxPayouts": 1,
        "groups": 0
    },
    "中症": {
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 0,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/cqB4hrQx9UVJ', NULL, NULL, '18岁-65周岁', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        5,
        10,
        15,
        20,
        30
    ]
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,76.35);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '太平洋保险', '太安心终身重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 125,
        "groups": 6,
        "maxPayouts": 6,
        "cancerSeparateGroup": true,
        "eachPayout": "每次给付100%基本保额"
    },
    "中症": {
        "count": 25,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付50%基本保额"
    },
    "轻症": {
        "count": 50,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 5151,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/ca4vpMTqYEdd', $$
{
    "其他": {}
}
$$, $$
{
    "重疾": {}
}
$$, '出生满30天-70周岁', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        15,
        20
    ]
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,90.75);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '太平洋保险', '金生无忧2024（少儿版）重大疾病保险', '少儿重疾险', $$
{
    "产品责任": {
        "count": 120,
        "maxPayouts": 1,
        "groups": 0
    },
    "中症": {
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 5040,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '180天', '20年', NULL, NULL, 'https://kdocs.cn/l/ce6Y0BRak4sR', NULL, NULL, '出生满28天-17周岁', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        5,
        10,
        15,
        20,
        30
    ]
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,75.45);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '中国太平', '福禄倍禧2024终身重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "groups": 6,
        "maxPayouts": 6,
        "eachPayout": "间隔期1年，每次给付100%基本保额"
    },
    "中症": {
        "count": 20,
        "maxPayouts": 2,
        "groups": 0,
        "eachPayout": "每次给付50%基本保额"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 6,
        "groups": 0,
        "eachPayout": "每次给付20%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 5460,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/cfuz3Mswkyqk', NULL, NULL, '出生满28天-65周岁', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        5,
        10,
        15,
        20,
        30
    ]
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,87.15);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '中国太平', '福禄添禧B款重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "maxPayouts": 1,
        "groups": 0
    },
    "中症": {
        "count": 20,
        "maxPayouts": 2,
        "groups": 0,
        "eachPayout": "每次给付50%基本保额"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 6,
        "groups": 0,
        "eachPayout": "每次给付25%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 5040,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '保至99周岁', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/ccI2gtTo2dps', NULL, NULL, '出生满28天-55周岁', $$
{
    "其他": {}
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,86.25);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '中国太平', '福禄娃2024终身重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "maxPayouts": 1,
        "groups": 0
    },
    "轻症": {
        "count": 50,
        "maxPayouts": 5,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 4350,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/cflS0yfn3FVZ', NULL, NULL, '出生满28天-17周岁', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        5,
        10,
        15,
        20
    ]
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,82.65);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '中国太平', '福禄康宁终身重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "groups": 6,
        "maxPayouts": 6,
        "eachPayout": "间隔期1年，每次给付100%基本保额"
    },
    "中症": {
        "count": 20,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付50%基本保额"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 6,
        "groups": 0,
        "eachPayout": "每次给付20%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 5310,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/cbE8lH6ZRoBU', NULL, NULL, '出生满28天-65周岁', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        5,
        10,
        15,
        20
    ]
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,87.15);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '新华保险', '多倍保障重大疾病保险（智赢版）', '多次赔付重疾险', $$
{
    "产品责任": {
        "count": 130,
        "groups": 6,
        "maxPayouts": 6,
        "cancerSeparateGroup": true,
        "firstPayoutBasis": "（85岁前给付首次重疾的，合同于被保人满85周岁的保单周年日终止，85岁后给付首次重疾的，给付首次重疾后合同终止）"
    },
    "中症": {
        "count": 20,
        "maxPayouts": 2,
        "groups": 0,
        "eachPayout": "每次给付50%基本保额"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 6,
        "groups": 0,
        "eachPayout": "每次给付20%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 0,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/ce4WatUJB2Jr', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '18-60周岁', $$
{
    "其他": {}
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,86.25);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '新华保险', '多倍保障少儿重大疾病保险（智赢版）', '多次赔付重疾险', $$
{
    "产品责任": {
        "count": 130,
        "groups": 6,
        "maxPayouts": 6,
        "cancerSeparateGroup": true,
        "firstPayoutBasis": "（85岁前给付首次重疾的，合同于被保人满85周岁的保单周年日终止，85岁后给付首次重疾的，给付首次重疾后合同终止）"
    },
    "中症": {
        "count": 20,
        "maxPayouts": 2,
        "groups": 0,
        "eachPayout": "每次给付50%基本保额"
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 6,
        "groups": 0,
        "eachPayout": "每次给付20%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 6150,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/cefO5ceVAOyO', $$
{
    "其他": {}
}
$$, NULL, '出生满30天-17周岁', $$
{
    "其他": {}
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,86.25);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '泰康人寿', '乐享健康（成人A/B款）重大疾病保险', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 120,
        "maxPayouts": 1,
        "groups": 0
    },
    "中症": {
        "count": 20,
        "maxPayouts": 1,
        "groups": 0
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 6,
        "groups": 0,
        "eachPayout": "每次给付25%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 0,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '90天', '18年', NULL, NULL, 'https://kdocs.cn/l/cmRe5M547jyv', $$
{
    "其他": {}
}
$$, NULL, '18-70周岁', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        3,
        5,
        10,
        15,
        18,
        23,
        28
    ]
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,87.15);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '泰康人寿', '乐享健康（少儿A/B款）重大疾病保险', '少儿重疾险', $$
{
    "产品责任": {
        "count": 120,
        "groups": 5,
        "maxPayouts": 5,
        "eachPayout": "每次给付100%基本保额/100%累计保费取大"
    },
    "中症": {
        "count": 20,
        "maxPayouts": 1,
        "groups": 0
    },
    "轻症": {
        "count": 40,
        "maxPayouts": 6,
        "groups": 0,
        "eachPayout": "每次给付25%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 6420,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '90天', '18年', NULL, NULL, 'https://kdocs.cn/l/caRX7BvsHKqJ', $$
{
    "其他": {}
}
$$, NULL, '0-17周岁', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        3,
        5,
        10,
        15,
        18,
        23,
        28
    ]
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,91.05);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '泰康养老', '健康有约终身重大疾病保险K款', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 135,
        "maxPayouts": 1,
        "groups": 0
    },
    "中症": {
        "count": 30,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付50%基本保额"
    },
    "轻症": {
        "count": 55,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付20%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 4143,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/ccrdHBHqMw25', NULL, NULL, '0-70周岁', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        15,
        20
    ]
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,98.25);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '泰康养老', '健康有约终身重大疾病保险K1款', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 135,
        "maxPayouts": 1,
        "groups": 0
    },
    "中症": {
        "count": 30,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付50%基本保额"
    },
    "轻症": {
        "count": 55,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付20%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 4452,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/crwxOCqtvAzG', NULL, NULL, '0-70周岁', $$
{
    "lumpSumModes": [
        "趸交"
    ],
    "annualPaymentYears": [
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        15,
        20
    ]
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,98.25);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '蚂蚁保·人保健康', '健康福·终身重疾险（升级版）', '单次赔付重疾险', $$
{
    "产品责任": {
        "count": 125,
        "maxPayouts": 1,
        "groups": 0
    },
    "中症": {
        "count": 25,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "重疾": {},
    "轻症": {
        "count": 50,
        "maxPayouts": 5,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 5601,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/cmkBNBdFk09r', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '出生满28天-55周岁', $$
{
    "其他": {}
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,96.15);

INSERT INTO insurance_products_critical_illness (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, payment_method, second_insured, clause_link, optional_liabilities, insurance_option, addtl_owner_waiver, payment_term_options, insurance_rules, owner_waiver, additional_riders, highlights, commission, commission_year1, commission_year2, commission_year3, commission_year4, commission_year5, total_score
) VALUES (
    '蚂蚁保·太平洋保险', '健康福·少儿终身重疾险', '少儿重疾险', $$
{
    "产品责任": {
        "count": 125,
        "maxPayouts": 4,
        "groups": 0,
        "stepPayout": "间隔期365天，依次给付100%/120%/150%/200%基本保额"
    },
    "中症": {
        "count": 25,
        "maxPayouts": 3,
        "groups": 0,
        "eachPayout": "每次给付60%基本保额"
    },
    "重疾": {},
    "轻症": {
        "count": 50,
        "maxPayouts": 4,
        "groups": 0,
        "eachPayout": "每次给付30%基本保额"
    }
}
$$, NULL, NULL, NULL, $$
{
    "text": "NULL"
}
$$, 0, $$
{
    "minAmount": 3378,
    "unit": "元"
}
$$, $$
{
    "minAmount": 300000,
    "unit": "元"
}
$$, '终身', '90天', '20年', NULL, NULL, 'https://kdocs.cn/l/cpBwJkY1UEv7', $$
{
    "其他": {}
}
$$, $$
{
    "其他": {}
}
$$, '出生满28天-17周岁', $$
{
    "其他": {}
}
$$, NULL, NULL, NULL, NULL,NULL,NULL,NULL,NULL,NULL,NULL,88.95);

