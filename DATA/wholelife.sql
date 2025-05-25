/* =============================================================
   1. 建表语句（字段长度 & 类型已按前文说明全部调整）
   ============================================================= */
CREATE TABLE insurance_products_wholelife (
    product_id                   SERIAL PRIMARY KEY,
    company_name                 VARCHAR(100),
    product_name                 VARCHAR(150),
    insurance_type               VARCHAR(50),

    coverage_content             JSONB,
    exclusion_clause             JSONB,
    renewable                    BOOLEAN,
    underwriting_rules           JSONB,

    entry_age                    JSONB,
    deductible                   DECIMAL(12,2),

    premium                      JSONB,
    coverage_amount              JSONB,

    coverage_period              VARCHAR(50),
    waiting_period               VARCHAR(20),
    payment_period               VARCHAR(50),
    payment_method               VARCHAR(50),

    second_insured               VARCHAR(30),
    intergenerational_insurance  VARCHAR(100),

    trust                        JSONB,
    trust_threshold              JSONB,

    retirement_community         VARCHAR(200),
    reduction_supported          VARCHAR(200),
    reduced_paid_up              VARCHAR(200),

    policy_loan_rate             DECIMAL(8,5),
    value_added_services         JSONB,
    sales_regions                JSONB,
    total_score                  DECIMAL(6,2)
);




/* =============================================================
   2. 全量数据（14 条）
   ============================================================= */

/* ---------- 01 同方全球 【新传世荣耀】（薪火版） ---------- */
INSERT INTO insurance_products_wholelife (
    company_name, product_name, insurance_type,
    coverage_content, exclusion_clause, renewable, underwriting_rules,
    entry_age, deductible, premium, coverage_amount,
    coverage_period, waiting_period, payment_period, payment_method,
    second_insured, intergenerational_insurance,
    trust, trust_threshold,
    retirement_community, reduction_supported, reduced_paid_up,
    policy_loan_rate, value_added_services, sales_regions, total_score
) VALUES (
    '同方全球',
    '【新传世荣耀】（薪火版）',
    '定额终身寿险',
    $${
        "基础": { "note": "身故/全残" },
        "附加责任": [
            "1.意外身故"
        ],
        "_": [
            "①一般意外（18-70周岁，额外10%保额）",
            "②航空意外（18-70周岁，额外200%保额，最高20000000元）",
            "③动车组列车意外（18-70周岁，额外50%保额，最高5000000元）"
        ],
        "2.猝死关爱金": { "percentage": 10, "amount": 2000000, "unit": "CNY", "note": "18-70周岁，额外保额，最高" },
        "身体状况": { "note": "标准体和优选体，其中优选体投保必须进行体检。" }
    }$$,
    $${
        "count": 5,
        "details": ["吸毒", "酒驾"]
    }$$,
    NULL,
    $${
        "health": "18-45岁，最高寿险风险保额15000000元",
        "other": ["优选体投保的客户都需参加核保体检。"]
    }$$,
    $${
        "min": "7天", "max": "70周"
    }$$,
    0,
    NULL,
    $${
        "amount": 500000, "unit": "CNY", "note": "优选体1000000元"
    }$$,
    '终身',
    '无',
    '1/3/5/10/15/20/30年',
    '月交/季交/半年交/年交',
    '支持',
    '支持，不要求被保人年龄',
    $$[
        "中信信托", "平安信托", "北京国际信托", "粤财信托", "国投泰康信托"
    ]$$,
    $${
        "amountType": "coverage", "amount": 3000000, "unit": "CNY"
    }$$,
    '不支持',
    '支持（写进条款，无比例和年限要求）',
    '不支持',
    0.046,
    $$[
        "专家门诊", "住院/手术安排", "就诊绿通", "每年一次体检（标保1000000元二星以上客户）"
    ]$$,
    $$[
        "上海","北京","四川","天津","宁波","山东","广东","江苏",
        "河北","浙江","深圳","湖北","福建","青岛"
    ]$$,
    86.57351154313487
);



/* ---------- 02 中荷人寿 荣耀世家（臻世版） ---------- */
INSERT INTO insurance_products_wholelife (
    company_name, product_name, insurance_type,
    coverage_content, exclusion_clause, renewable, underwriting_rules,
    entry_age, deductible, premium, coverage_amount,
    coverage_period, waiting_period, payment_period, payment_method,
    second_insured, intergenerational_insurance,
    trust, trust_threshold,
    retirement_community, reduction_supported, reduced_paid_up,
    policy_loan_rate, value_added_services, sales_regions, total_score
) VALUES (
    '中荷人寿',
    '荣耀世家（臻世版）',
    '定额终身寿险',
    $${
        "基础": { "note": "身故/全残" },
        "附加责任": [],
        "①航空意外身故": { "multiplier": 1,  "amount": 20000000, "unit": "CNY", "note": "18-70周岁，额外保额，最高" },
        "②轨道交通意外身故": { "percentage": 50, "amount": 20000000, "unit": "CNY", "note": "18-70周岁，额外保额，最高" }
    }$$,
    $${
        "count": 7,
        "details": ["吸毒","酒驾","战争","核辐射"]
    }$$,
    NULL,
    $${
        "health": "18-45岁最高15000000元（最高层级）",
        "financial": "18-45岁最高15000000元（最高层级）"
    }$$,
    $${
        "min": "30天", "max": "75周"
    }$$,
    0,
    NULL,
    $${
        "amount": 500000, "unit": "CNY"
    }$$,
    '终身',
    '无',
    '1/3/5/10/15/20/30年',
    '月交/季交/半年交/年交',
    '不支持',
    '支持（8岁以上）',
    $$[
        "北京","五矿","国投泰康","陕国投","中诚","外贸"
    ]$$,
    $${
        "amount": 1000000, "unit": "CNY", "note": "起投"
    }$$,
    NULL,
    '条款支持（合同有效期内，累计减保不超过生效时基本保额的20%）',
    '支持',
    0.0525,
    $$[
        "就医绿通","海外二诊","国内重疾绿通"
    ]$$,
    $$[
        "上海","北京","天津","安徽","山东","江苏","河北","河南",
        "辽宁","青岛"
    ]$$,
    75.03037667071689
);



/* ---------- 03 百年人寿 珍爱永恒（典藏版） ---------- */
INSERT INTO insurance_products_wholelife (
    company_name, product_name, insurance_type,
    coverage_content, exclusion_clause, renewable, underwriting_rules,
    entry_age, deductible, premium, coverage_amount,
    coverage_period, waiting_period, payment_period, payment_method,
    second_insured, intergenerational_insurance,
    trust, trust_threshold,
    retirement_community, reduction_supported, reduced_paid_up,
    policy_loan_rate, value_added_services, sales_regions, total_score
) VALUES (
    '百年人寿',
    '珍爱永恒（典藏版）',
    '定额终身寿险',
    $${
        "基础": { "note": "身故/全残" },
        "附加责任": [],
        "_": [
            "①意外身故（一般意外，航空意外）",
            "②投保人意外身故或全残保费豁免"
        ]
    }$$,
    $${
        "count": 7,
        "details": ["吸毒","酒驾","战争","核辐射"]
    }$$,
    NULL,
    $${
        "health": "18-45岁，8000000元累计寿险风险保额",
        "financial": "18-45岁，同业寿险额度超过5000000元"
    }$$,
    $${
        "min":"28天", "max":"70岁"
    }$$,
    0,
    NULL,
    $${
        "amount": 1000000, "unit": "CNY"
    }$$,
    '终身',
    '90天',
    '1/3/5/10/15/20/30年',
    '年交',
    '不支持',
    '支持（8岁以上）',
    $$["北京信托"]$$,
    $${
        "amount": 5000000,"amountType":"coverage","unit":"CNY"
    }$$,
    NULL,
    '条款支持（合同有效期内，累计减保不超过生效时基本保额的20%）',
    '不支持',
    0.0575,
    NULL,
    $$[
        "内蒙古","北京","吉林","四川","大连","安徽","山东","山西","广东",
        "江苏","江西","河北","河南","浙江","湖北","福建","辽宁",
        "重庆","陕西","黑龙江"
    ]$$,
    76.76184690157959
);



/* ---------- 04 长城人寿 山海关锦绣版 ---------- */
INSERT INTO insurance_products_wholelife (
    company_name, product_name, insurance_type,
    coverage_content, exclusion_clause, renewable, underwriting_rules,
    entry_age, deductible, premium, coverage_amount,
    coverage_period, waiting_period, payment_period, payment_method,
    second_insured, intergenerational_insurance,
    trust, trust_threshold,
    retirement_community, reduction_supported, reduced_paid_up,
    policy_loan_rate, value_added_services, sales_regions, total_score
) VALUES (
    '长城人寿',
    '山海关锦绣版',
    '定额终身寿险',
    $${
        "基础": { "note": "身故" },
        "附加责任": [],
        "_": ["全残、18-70岁意外身故、18-70岁航空意外身故"]
    }$$,
    $${
        "count": 5,
        "details": ["吸毒","酒驾"]
    }$$,
    NULL,
    $${
        "health": "18-45岁，高净值客户12000000元累计寿险风险保额，普通客户4000000元",
        "financial": "4000000元"
    }$$,
    $${
        "min":"30天","max":"70周"
    }$$,
    0,
    $${
        "travelResidenceMinTotalPremium": 200000,
        "longResidenceMinTotalPremium": 800000,
        "unit":"CNY"
    }$$,
    $${
        "amount": 500000,"unit":"CNY"
    }$$,
    '终身',
    '无',
    '1/3/5/10/15/20/30年',
    '年交',
    '支持',
    '支持（8岁以上）',
    $$["北京信托","五矿信托","万向信托"]$$,
    $${
        "amount": 3000000,"amountType":"totalPremium","unit":"CNY"
    }$$,
    '支持',
    '支持（写进条款，合同有效期内，每年累计减保以投保时保单载明的20%基本保额为限）',
    '条款支持',
    0.05,
    NULL,
    $$[
        "北京","四川","天津","安徽","山东","广东","江苏","河北","河南",
        "湖北","湖南","重庆","陕西","青岛"
    ]$$,
    74.1494532199271
);



/* ---------- 05 中韩人寿 东方鑫享一生 ---------- */
INSERT INTO insurance_products_wholelife (
    company_name, product_name, insurance_type,
    coverage_content, exclusion_clause, renewable, underwriting_rules,
    entry_age, deductible, premium, coverage_amount,
    coverage_period, waiting_period, payment_period, payment_method,
    second_insured, intergenerational_insurance,
    trust, trust_threshold,
    retirement_community, reduction_supported, reduced_paid_up,
    policy_loan_rate, value_added_services, sales_regions, total_score
) VALUES (
    '中韩人寿',
    '东方鑫享一生',
    '定额终身寿险',
    $${
        "_": ["身故"]
    }$$,
    $${
        "count": 7,
        "details": ["吸毒","酒驾","战争","核辐射"]
    }$$,
    NULL,
    $${
        "health": "18-45周岁，私行客户8000000元，普通客户4000000元",
        "financial": "18-45周岁，私行客户8000000元，普通客户4000000元"
    }$$,
    $${
        "min":"28天","max":"65周"
    }$$,
    0,
    NULL,
    $${
        "amount": 200000,"unit":"CNY"
    }$$,
    '终身',
    '无',
    '1/3/5/10/20年',
    '年交',
    '支持',
    NULL,
    NULL,
    NULL,
    NULL,
    '支持（写进条款，5年后，每年累计减保金额不超过累计保费的20%）',
    '条款支持',
    0.049,
    NULL,
    $$["宁波","安徽","江苏","浙江"]$$,
    79.28311057108141
);



/* ---------- 06 中意人寿 臻享一生（传玺版）（分红型） ---------- */
INSERT INTO insurance_products_wholelife (
    company_name, product_name, insurance_type,
    coverage_content, exclusion_clause, renewable, underwriting_rules,
    entry_age, deductible, premium, coverage_amount,
    coverage_period, waiting_period, payment_period, payment_method,
    second_insured, intergenerational_insurance,
    trust, trust_threshold,
    retirement_community, reduction_supported, reduced_paid_up,
    policy_loan_rate, value_added_services, sales_regions, total_score
) VALUES (
    '中意人寿',
    '臻享一生（传玺版）（分红型）',
    '定额终身寿险',
    $${
        "_": ["身故/全残（可附加）"]
    }$$,
    $${
        "count": 5,
        "details": ["吸毒","酒驾"]
    }$$,
    NULL,
    $${
        "health": "18-45周岁，超高净值客户15000000元，高净值12000000元，优质客户5000000元，普通客户最高4000000元",
        "financial": "18-45周岁，超高净值客户15000000元，高净值12000000元，优质客户5000000元，普通客户最高4000000元"
    }$$,
    $${
        "min":"7天","max":"70周"
    }$$,
    0,
    NULL,
    $${
        "amount": 200000,"unit":"CNY"
    }$$,
    '终身',
    '无',
    '1/3/5/10/15/20/30年',
    '月交/季交/半年交/年交',
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    $$[
        "上海","北京","四川","宁波","山东","广东","江苏","河北","河南",
        "浙江","深圳","湖北","福建","辽宁","重庆","陕西","青岛"
    ]$$,
    94.65370595382745
);



/* ---------- 07 中英人寿 心爱永恒 ---------- */
INSERT INTO insurance_products_wholelife (
    company_name, product_name, insurance_type,
    coverage_content, exclusion_clause, renewable, underwriting_rules,
    entry_age, deductible, premium, coverage_amount,
    coverage_period, waiting_period, payment_period, payment_method,
    second_insured, intergenerational_insurance,
    trust, trust_threshold,
    retirement_community, reduction_supported, reduced_paid_up,
    policy_loan_rate, value_added_services, sales_regions, total_score
) VALUES (
    '中英人寿',
    '心爱永恒',
    '定额终身寿险',
    $${
        "_": ["身故/航空意外身故/公共交通意外身故"]
    }$$,
    $${
        "count": 5,
        "details": ["吸毒","酒驾"]
    }$$,
    NULL,
    $${
        "health":"12000000元",
        "financial":"12000000元"
    }$$,
    $${
        "min":"30天","max":"70周"
    }$$,
    0,
    NULL,
    $${
        "amount": 100000,"unit":"CNY","note":"未成年人，成人1000000元"
    }$$,
    '终身',
    '无',
    '1/3/5/10/15/20/30年',
    '月交/季交/半年交/年交',
    '不支持',
    '支持（8岁以上）',
    $$["支持"]$$,
    NULL,
    NULL,
    '条款支持',
    '支持',
    NULL,
    NULL,
    $$[
        "上海","北京","厦门","四川","安徽","山东","广东","江苏","江西",
        "河北","河南","深圳","湖北","湖南","福建","辽宁","陕西","青岛",
        "黑龙江"
    ]$$,
    76.09356014580801
);



/* ---------- 08 工银安盛人寿 鑫享世承 ---------- */
INSERT INTO insurance_products_wholelife (
    company_name, product_name, insurance_type,
    coverage_content, exclusion_clause, renewable, underwriting_rules,
    entry_age, deductible, premium, coverage_amount,
    coverage_period, waiting_period, payment_period, payment_method,
    second_insured, intergenerational_insurance,
    trust, trust_threshold,
    retirement_community, reduction_supported, reduced_paid_up,
    policy_loan_rate, value_added_services, sales_regions, total_score
) VALUES (
    '工银安盛人寿',
    '鑫享世承',
    '定额终身寿险',
    $${
        "基础": { "note": "一般身故保险金" },
        "_": ["航空意外身故，轨道交通意外身故","全残保险金"],
        "附加责任":[]
    }$$,
    $${
        "text":"3条/航空意外身故5条"
    }$$,
    NULL,
    $${
        "health":"18-45岁，4000000元累计寿险风险保额，高净值12000000元",
        "financial":"18-45岁，4000000元累计寿险风险保额，高净值12000000元"
    }$$,
    $${
        "min":"28天","max":"75岁"
    }$$,
    0,
    NULL,
    $${
        "amount": 1000000,"unit":"CNY"
    }$$,
    '终身',
    '无',
    '1/5/10/15/20/30年',
    '年交',
    '条款支持',
    NULL,
    $$["五矿信托"]$$,
    NULL,
    NULL,
    NULL,
    '支持',
    NULL,
    NULL,
    $$[
        "上海","云南","北京","四川","天津","安徽","山东","山西","广东","广西",
        "江苏","江西","河北","河南","浙江","深圳","湖北","福建","辽宁","重庆",
        "陕西"
    ]$$,
    72.60024301336574
);



/* ---------- 09 陆家嘴国泰人寿 顺意人生2.0 ---------- */
INSERT INTO insurance_products_wholelife (
    company_name, product_name, insurance_type,
    coverage_content, exclusion_clause, renewable, underwriting_rules,
    entry_age, deductible, premium, coverage_amount,
    coverage_period, waiting_period, payment_period, payment_method,
    second_insured, intergenerational_insurance,
    trust, trust_threshold,
    retirement_community, reduction_supported, reduced_paid_up,
    policy_loan_rate, value_added_services, sales_regions, total_score
) VALUES (
    '陆家嘴国泰人寿',
    '顺意人生2.0',
    '定额终身寿险',
    $${
        "_": ["身故保险金","航空意外身故保险金","新能源私家车意外身故保险金"]
    }$$,
    $${
        "text":"身故和航空意外身故3条/新能源6条"
    }$$,
    NULL,
    $${
        "health":"18-45岁，最高15000000元保额",
        "financial":"18-45岁，最高15000000元保额"
    }$$,
    $${
        "min":"28天","max":"75岁"
    }$$,
    0,
    $${
        "unit":"CNY",
        "travelResidenceMinTotalPremium": 200000,
        "longResidenceMinTotalPremium": 800000
    }$$,
    $${
        "amount": 1000000,"unit":"CNY"
    }$$,
    '终身',
    '无',
    '1/3/5/10/15/20/30年',
    '年交',
    NULL,
    NULL,
    $$["支持","17家信托"]$$,
    $${
        "amountType":"totalPremium","amount": 3000000,"unit":"CNY"
    }$$,
    '支持',
    '支持（写进条款，合同有效期内，每年累计减保以投保时保单载明的20%基本保额为限）',
    NULL,
    0.0475,
    NULL,
    $$[
        "上海","北京","厦门","四川","天津","宁波","山东","广东","江苏",
        "河南","浙江","深圳","福建","辽宁"
    ]$$,
    75.94167679222357
);



/* ---------- 10 中信保诚 【祯祥世家】（典藏版） ---------- */
INSERT INTO insurance_products_wholelife (
    company_name, product_name, insurance_type,
    coverage_content, exclusion_clause, renewable, underwriting_rules,
    entry_age, deductible, premium, coverage_amount,
    coverage_period, waiting_period, payment_period, payment_method,
    second_insured, intergenerational_insurance,
    trust, trust_threshold,
    retirement_community, reduction_supported, reduced_paid_up,
    policy_loan_rate, value_added_services, sales_regions, total_score
) VALUES (
    '中信保诚',
    '【祯祥世家】（典藏版）',
    '定额终身寿险',
    $${
        "_": ["身故"]
    }$$,
    $${
        "text":"3条"
    }$$,
    NULL,
    NULL,
    $${
        "min":"30天","max":"75岁"
    }$$,
    0,
    NULL,
    $${
        "amount": 1000000,"unit":"CNY"
    }$$,
    '终身',
    '无',
    '1/3/5/10/15/20/25/30年',
    '年交',
    '支持',
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    '条款支持',
    NULL,
    NULL,
    $$[
        "上海","北京","四川","天津","宁波","安徽","山东","山西","广东",
        "广西","江苏","河北","河南","浙江","深圳","湖北","湖南","福建",
        "辽宁","陕西","青岛"
    ]$$,
    73.23815309842041
);



/* ---------- 11 安联人寿 盛世臻传C（Ⅲ）（分红型） ---------- */
INSERT INTO insurance_products_wholelife (
    company_name, product_name, insurance_type,
    coverage_content, exclusion_clause, renewable, underwriting_rules,
    entry_age, deductible, premium, coverage_amount,
    coverage_period, waiting_period, payment_period, payment_method,
    second_insured, intergenerational_insurance,
    trust, trust_threshold,
    retirement_community, reduction_supported, reduced_paid_up,
    policy_loan_rate, value_added_services, sales_regions, total_score
) VALUES (
    '安联人寿',
    '盛世臻传C（Ⅲ）（分红型）',
    '定额终身寿险',
    $${
        "_": ["身故/增额红利"]
    }$$,
    $${
        "text":"3条"
    }$$,
    NULL,
    $${
        "health":"18-45周岁，一类客户12000000元，二类8000000元，三类5000000元保额",
        "financial":"18-45周岁，一类客户12000000元，二类8000000元，三类5000000元保额"
    }$$,
    $${
        "min":"7天","max":"65周"
    }$$,
    0,
    NULL,
    $${
        "amount": 500000,"unit":"CNY","note":"未成年人，成人标准体1000000元，优选体3000000元"
    }$$,
    '终身',
    '无',
    '1/5/10/15/20/25/30年',
    '年交',
    '条款支持',
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    '条款支持',
    NULL,
    NULL,
    $$[
        "上海","北京","四川","宁波","山东","广东","江苏","浙江","深圳",
        "湖北","青岛"
    ]$$,
    91.67679222357229
);



/* ---------- 12 瑞泰人寿 鸿利传世终身寿险 ---------- */
INSERT INTO insurance_products_wholelife (
    company_name, product_name, insurance_type,
    coverage_content, exclusion_clause, renewable, underwriting_rules,
    entry_age, deductible, premium, coverage_amount,
    coverage_period, waiting_period, payment_period, payment_method,
    second_insured, intergenerational_insurance,
    trust, trust_threshold,
    retirement_community, reduction_supported, reduced_paid_up,
    policy_loan_rate, value_added_services, sales_regions, total_score
) VALUES (
    '瑞泰人寿',
    '瑞泰人寿鸿利传世终身寿险',
    '定额终身寿险',
    $${
        "基础责任": { "note":"身故或全残保险金" },
        "附加责任": ["航空意外身故","公共交通意外身故","猝死关爱金","身故或全残额外关爱保险金"]
    }$$,
    $${
        "text":"3条"
    }$$,
    NULL,
    $${
        "health":"12000000元",
        "financial":"12000000元"
    }$$,
    $${
        "min":"30天","max":"70周"
    }$$,
    0,
    NULL,
    $${
        "amount": 500000,"unit":"CNY"
    }$$,
    '终身',
    '90天，意外无等待期',
    '1/3/5/10/20/30年',
    '年交',
    '条款支持',
    NULL,
    $$["支持（国投泰康）"]$$,
    $${
        "amountType":"totalPremium","amount": 3000000,"unit":"CNY"
    }$$,
    NULL,
    '支持（写进条款，合同有效期内，每年累计减保以投保时保单载明的20%基本保额为限）',
    '支持',
    NULL,
    NULL,
    $$[
        "上海","北京","宁波","广东","江苏","浙江","深圳","湖北",
        "重庆","陕西"
    ]$$,
    76.39732685297692
);



/* ---------- 13 复星保德信人寿  星佑家终身寿险 ---------- */
INSERT INTO insurance_products_wholelife (
    company_name, product_name, insurance_type,
    coverage_content, exclusion_clause, renewable, underwriting_rules,
    entry_age, deductible, premium, coverage_amount,
    coverage_period, waiting_period, payment_period, payment_method,
    second_insured, intergenerational_insurance,
    trust, trust_threshold,
    retirement_community, reduction_supported, reduced_paid_up,
    policy_loan_rate, value_added_services, sales_regions, total_score
) VALUES (
    '复星保德信人寿',
    '复星保德信星佑家终身寿险',
    '定额终身寿险',
    $${
        "基础": { "note":"一般身故/全残保险金" },
        "_": ["航空意外身故，轨道交通意外身故","身故或全残额外关爱保险金可附加"],
        "附加责任":[]
    }$$,
    $${
        "count": 7,
        "details":["吸毒","酒驾","战争","核辐射"]
    }$$,
    NULL,
    $${
        "health":"18-45岁，最高15000000元保额",
        "financial":"18-45岁，最高15000000元保额"
    }$$,
    $${
        "min":"28天","max":"70岁"
    }$$,
    0,
    $${
        "unit":"CNY",
        "minTotalPremium": 1200000
    }$$,
    $${
        "amount": 1000000,"unit":"CNY"
    }$$,
    '终身',
    '无',
    '1/3/5/10/15/20/30年',
    '年交',
    NULL,
    NULL,
    $$["支持","17家信托公司"]$$,
    $${
        "amountType":"totalPremium","amount": 3000000,"unit":"CNY"
    }$$,
    '支持',
    NULL,
    '支持',
    NULL,
    NULL,
    $$[
        "上海","北京","四川","山东","江苏","河南","青岛"
    ]$$,
    71.17253948967193
);



/* ---------- 14 招商信诺人寿 信享传耀终身寿险（分红型） ---------- */
INSERT INTO insurance_products_wholelife (
    company_name, product_name, insurance_type,
    coverage_content, exclusion_clause, renewable, underwriting_rules,
    entry_age, deductible, premium, coverage_amount,
    coverage_period, waiting_period, payment_period, payment_method,
    second_insured, intergenerational_insurance,
    trust, trust_threshold,
    retirement_community, reduction_supported, reduced_paid_up,
    policy_loan_rate, value_added_services, sales_regions, total_score
) VALUES (
    '招商信诺人寿',
    '招商信诺信享传耀终身寿险（分红型）',
    '定额终身寿险',
    $${
        "_": ["身故保险金","保额分红"]
    }$$,
    $${
        "count": 7,
        "details":["吸毒","酒驾","战争","核辐射"]
    }$$,
    NULL,
    $${
        "health":"18-45岁，最高18000000元保额",
        "financial":"18-45岁，最高18000000元保额"
    }$$,
    $${
        "min":"28天","max":"68岁"
    }$$,
    0,
    NULL,
    $${
        "amount": 500000,"unit":"CNY"
    }$$,
    '终身',
    '无',
    '1/3/5/10/20年',
    '年交',
    NULL,
    NULL,
    $$["支持"]$$,
    $${
        "amountType":"coverage","amount": 3000000,"unit":"CNY"
    }$$,
    NULL,
    NULL,
    '支持',
    NULL,
    NULL,
    $$[
        "上海","云南","北京","四川","天津","宁波","安徽","山东",
        "广东","江苏","江西","河南","浙江","深圳","湖北","湖南",
        "福建","辽宁","重庆","陕西","青岛"
    ]$$,
    100.0
);

