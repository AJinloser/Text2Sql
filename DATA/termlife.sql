CREATE TABLE insurance_products_termlife (
    product_id         SERIAL PRIMARY KEY,                 -- 产品唯一标识符
    company_name       VARCHAR(100),                       -- 保险公司名称
    product_name       VARCHAR(150),                       -- 产品名称
    insurance_type     VARCHAR(50),                        -- 保险类型（如消费型定期寿险）
    coverage_content   JSONB,                              -- 保险责任内容
    exclusion_clause   JSONB,                              -- 责任免除条款
    renewable          BOOLEAN,                            -- 是否可续保
    underwriting_rules JSONB,                              -- 核保规则
    entry_age          VARCHAR(50),                        -- 投保年龄范围
    deductible         DECIMAL(12,2),                      -- 免赔额
    premium            JSONB,                              -- 保费金额
    coverage_amount    JSONB,                              -- 保额
    coverage_period    VARCHAR(50),                        -- 保障期限
    waiting_period     VARCHAR(20),                        -- 等待期
    payment_period     VARCHAR(20),                        -- 缴费期间
    total_score        DECIMAL(6,2)                        -- 综合评分
);

-- ---------------------------------------------------------
-- 华贵人寿
-- ---------------------------------------------------------
INSERT INTO insurance_products_termlife (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, total_score
) VALUES (
    '华贵人寿',
    '华贵大麦2024定期寿险（互联网专属）',
    '消费型定期寿险',
    $$
{
    "身故保险金": {
        "percentage": 100.0,
        "basis": "基本保额"
    },
    "全残保险金": {
        "percentage": 100.0,
        "basis": "基本保额"
    },
    "其他责任": {
        "航空意外身故或全残保险金": {
            "note": "额外给付",
            "extraAmount": 10000000,
            "unit": "元"
        },
        "水陆公共交通意外身故或身体全残保险金": {
            "note": "额外给付",
            "multiplier": 2.0
        }
    }
}
$$,
    NULL,
    NULL,
    NULL,
    NULL,
    0,
    $$
{
    "amount": 1576.2,
    "unit": "元"
}
$$,
    $$
{
    "amount": 1000000,
    "unit": "元"
}
$$,
    '至60周岁',
    '90天',
    '20年',
    98
);

-- ---------------------------------------------------------
-- 同方全球人寿
-- ---------------------------------------------------------
INSERT INTO insurance_products_termlife (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, total_score
) VALUES (
    '同方全球人寿',
    '同方全球【臻爱2024】互联网定期寿险',
    '消费型定期寿险',
    $$
{
    "身故保险金": {
        "percentage": 100.0,
        "basis": "基本保额"
    },
    "全残保险金": {
        "percentage": 100.0,
        "basis": "基本保额"
    },
    "其他责任": {
        "猝死关爱金": {
            "percentage": 30.0,
            "ageLimit": "65周岁前"
        },
        "水陆空公共交通意外身故或全残保险金": {
            "percentage": 100.0
        },
        "癌症身故保险金": {
            "percentage": 50.0,
            "ageLimit": "65周岁前"
        }
    }
}
$$,
    NULL,
    NULL,
    NULL,
    NULL,
    0,
    $$
{
    "amount": 1581,
    "unit": "元"
}
$$,
    $$
{
    "amount": 1000000,
    "unit": "元"
}
$$,
    '至60周岁',
    '90天',
    '20年',
    97.78
);

-- ---------------------------------------------------------
-- 北京人寿
-- ---------------------------------------------------------
INSERT INTO insurance_products_termlife (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, total_score
) VALUES (
    '北京人寿',
    '擎天柱9号定期寿险（互联网专属）',
    '消费型定期寿险',
    $$
{
    "身故保险金": {
        "percentage": 100.0,
        "basis": "基本保额"
    },
    "全残保险金": {
        "percentage": 100.0,
        "basis": "基本保额"
    },
    "其他责任": {
        "猝死关爱保险金": {
            "extraAmount": 500000,
            "unit": "元",
            "percentage": 30.0,
            "ageLimit": "65周岁前",
            "maxCap": 500000
        },
        "航空意外身故或高度残疾额外保险金": {
            "percentage": 400.0
        },
        "家庭关爱身故或高度残疾保险金": {
            "percentage": 30.0
        }
    },
    "水陆公共交通工具意外身故或高度残疾额外保险金": {
        "percentage": 100.0,
        "basis": "基本保额"
    },
    "法定节假日自驾车意外身故或高度残疾额外保险金": {
        "percentage": 50.0,
        "basis": "基本保额"
    }
}
$$,
    NULL,
    NULL,
    NULL,
    NULL,
    0,
    $$
{
    "amount": 1546,
    "unit": "元"
}
$$,
    $$
{
    "amount": 1000000,
    "unit": "元"
}
$$,
    '至60周岁',
    '90天',
    '20年',
    100
);

-- ---------------------------------------------------------
-- 国富人寿
-- ---------------------------------------------------------
INSERT INTO insurance_products_termlife (
    company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, waiting_period, payment_period, total_score
) VALUES (
    '国富人寿',
    '定海柱6号（齐家2025定期寿险（互联网专属））定期寿险（互联网专属）',
    '消费型定期寿险',
    $$
{
    "身故保险金": {
        "percentage": 100.0,
        "basis": "基本保额"
    },
    "全残保险金": {
        "percentage": 100.0,
        "basis": "基本保额"
    },
    "其他责任": {
        "猝死关爱保险金": {
            "percentage": 30.0,
            "ageLimit": "65周岁前"
        },
        "身故或全残特别关爱金": {
            "note": "额外给付",
            "percentage": 50.0
        },
        "航空意外身故或高度残疾额外保险金": {
            "percentage": 400.0
        }
    },
    "水陆公共交通工具意外身故或高度残疾额外保险金": {
        "percentage": 100.0,
        "basis": "基本保额"
    },
    "法定节假日自驾车意外身故或高度残疾额外保险金": {
        "percentage": 50.0,
        "basis": "基本保额"
    }
}
$$,
    NULL,
    NULL,
    NULL,
    NULL,
    0,
    $$
{
    "amount": 1876,
    "unit": "元"
}
$$,
    $$
{
    "amount": 1000000,
    "unit": "元"
}
$$,
    '至60周岁',
    '90天',
    '20年',
    82
);
