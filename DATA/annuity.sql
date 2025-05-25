
-- =========================================================
-- Table: insurance_products_annuity  (Annuity Insurance Products)
-- =========================================================
DROP TABLE IF EXISTS insurance_products_annuity;

CREATE TABLE insurance_products_annuity (
    product_id SERIAL PRIMARY KEY,
    company_name VARCHAR(100),
    product_name VARCHAR(150),
    insurance_type VARCHAR(50),
    coverage_content JSONB,
    exclusion_clause JSONB,
    renewable VARCHAR(50),
    underwriting_rules JSONB,
    entry_age VARCHAR(50),
    deductible DECIMAL(12,2),
    premium JSONB,
    coverage_amount JSONB,
    coverage_period VARCHAR(50),
    sales_regions VARCHAR(100),
    payment_period VARCHAR(30),
    payment_method VARCHAR(20),
    payout_method VARCHAR(20),
    universal_account VARCHAR(120),
    irr_15y DECIMAL(8,5),
    irr_20y DECIMAL(8,5),
    irr_30y DECIMAL(8,5),
    irr_40y DECIMAL(8,5),
    trust TEXT,
    intergenerational_insurance VARCHAR(50),
    second_insured VARCHAR(20),
    beneficiaries VARCHAR(100),
    bonus_distribution VARCHAR(50),
    retirement_community VARCHAR(20),
    total_score DECIMAL(6,2)
);

INSERT INTO insurance_products_annuity (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, sales_regions, payment_period, payment_method, payout_method, universal_account, irr_15y, irr_20y, irr_30y, irr_40y, trust, intergenerational_insurance, second_insured, beneficiaries, bonus_distribution, retirement_community, total_score) VALUES ('''瑞众人寿''', '''东方红年金保险''', '''终身年金''', $$
{
    "若被保险人在本合同有效期间内身故/全残，我们将按以下两项的较大者给付身故/全残保险金，同时本合同终止": {
        "text": ""
    },
    "_": [
        {
            "text": "自本合同第5个保单周年日起（含第5个保单周年日），若被保险人在任一保单周年日零时仍生存，"
        },
        {
            "text": "我们将于该保单周年日按本合同基本保险金额给付一次年金"
        },
        {
            "text": "身故/全残保险金"
        },
        {
            "text": "（一）被保险人身故/全残时本合同的已交保险费减去累计已领取的年金"
        },
        {
            "text": "（二）被保险人身故/全残时本合同的现金价值"
        }
    ]
}
$$, $$
{
    "身故责任": {
        "text": "(已交保费-已领取年金)vs现金价值"
    }
}
$$, NULL, NULL, '''5天-80周岁''', 0, '{
    "note": "3000元"
}', NULL, '''终身''', '上海、云南、内蒙古、北京、四川、天津、宁波、安徽、山东、广东、江苏、江西、河北、河南、浙江、海南、深圳、湖南、陕西、青岛', '1/3/5/6/10/20年', '年', '年', '金管家终身寿险（万能型，祥瑞版）', 0.01682483474459695, 0.01871137466931216, 0.0204295811703068, 0.02120931617937138, NULL, NULL, NULL, NULL, NULL, NULL, '63');
INSERT INTO insurance_products_annuity (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, sales_regions, payment_period, payment_method, payout_method, universal_account, irr_15y, irr_20y, irr_30y, irr_40y, trust, intergenerational_insurance, second_insured, beneficiaries, bonus_distribution, retirement_community, total_score) VALUES ('''长城人寿''', '''八达岭赤兔版甄选年金保险''', '''终身年金''', $$
{
    "可选年金领取日": {
        "text": "第5/7/10/15/20/30年开始领取"
    },
    "普惠计划": {
        "text": "年金固定领取1倍保额"
    },
    "幸福计划": {
        "text": "60岁前1倍保额，60岁后2倍保额"
    },
    "疾速计划": {
        "percentage": 12.0,
        "note": "累计保费，领取第6年后1倍保额"
    },
    "尊享计划": {
        "text": "年金固定领取1倍保额，终身现价不低于总保费"
    }
}
$$, $$
{
    "3. 普惠计划": {
        "text": "首期年金领取日开始至身故，按照基本保额 X1领取（年领）"
    },
    "4. 幸福计划": {
        "text": "首期年金领取日开始至60周岁，按照基本保额 X1领取"
    },
    "5. 疾速计划": {
        "percentage": 12.0
    },
    "6. 尊享计划（仅年领）": {
        "text": "首期领取日时仍生存，按照基本保额给付"
    },
    "之后每个领取日至身故或至105周岁，第5个保单日": {
        "percentage": 2.6,
        "note": "/2.5%X1"
    },
    "第7个保单日": {
        "percentage": 2.75,
        "note": "/2.65%X1"
    },
    "第10个保单日": {
        "percentage": 3.0,
        "note": "/2.85%/2.7%X1"
    },
    "_": [
        {
            "text": "1.至首期年金领取日前的第6个月零时止，普惠计划、幸福计划、尊享计划不可变更为疾速计划。首期年金领取后，不得变更保障计划"
        },
        {
            "text": "2. 普惠/幸福/疾速计划保险期为终身"
        },
        {
            "text": "尊享计划为保至105周岁"
        },
        {
            "text": "60周岁后至身故X2领取（年领）"
        },
        {
            "text": "第6个年金领取日后至身故，按照基本保额给付"
        },
        {
            "percentage": 2.6,
            "note": "）（第5保单日）"
        }
    ]
}
$$, NULL, NULL, '''30天-70周岁''', 0, '{
    "singlePayMin": 10000,
    "unit": "元"
}', NULL, '''终身（普惠/幸福/疾速）/至105岁（尊享）''', '北京、四川、天津、安徽、山东、广东、江苏、河北、河南、湖北、湖南、重庆、陕西、青岛', '1/3/5/6/7/10/15/20年', '年', '月/年（疾速计划仅可年领）', NULL, 0.01431339104219154, 0.01685538304933765, 0.01919046828350957, 0.02026939874895017, '北京信托、五矿信托、万向信托（300万总保费起）', '要求被保人大于8岁', '支持', '直系', NULL, NULL, '75');
INSERT INTO insurance_products_annuity (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, sales_regions, payment_period, payment_method, payout_method, universal_account, irr_15y, irr_20y, irr_30y, irr_40y, trust, intergenerational_insurance, second_insured, beneficiaries, bonus_distribution, retirement_community, total_score) VALUES ('''富德生命人寿''', '''鑫福多年金保险''', '''终身年金''', $$
{
    "生存金": {
        "text": "可以选择第6年、第10年，年满60周岁开始领取"
    },
    "_": [
        {
            "percentage": 50.0,
            "note": "保额"
        },
        {
            "percentage": 85.0
        },
        {
            "text": "给付1次"
        },
        {
            "text": "经同意可以在领取日之前变更领取"
        }
    ]
}
$$, $$
{
    "身故责任:60岁前": {
        "text": "累计保费/现金价值取大"
    },
    "60岁后": {
        "text": "无身故保险金"
    }
}
$$, NULL, NULL, '''25天-55岁''', 0, '{
    "singlePayMin": 100000,
    "unit": "元",
    "note": "3年30000元；5年20000元；6年15000元；10年10000元"
}', NULL, '''终身''', '上海、云南、内蒙古、北京、厦门、吉林、四川、大连、天津、宁夏、宁波、安徽、山东、山西、广东、广西、新疆、江苏、江西、河北、河南、浙江、海南、深圳、湖北、湖南、福建、贵州、辽宁、重庆、陕西、青岛、黑龙江', '1/3/5/6/10年', '年', '年', NULL, 0.01444627586824754, 0.01702334913164916, 0.01941206724565903, 0.0205394085050139, NULL, '要求被保人大于8岁', '不支持', '可非直系，需要柜面提交', NULL, NULL, '63');
INSERT INTO insurance_products_annuity (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, sales_regions, payment_period, payment_method, payout_method, universal_account, irr_15y, irr_20y, irr_30y, irr_40y, trust, intergenerational_insurance, second_insured, beneficiaries, bonus_distribution, retirement_community, total_score) VALUES ('''安联''', '''逸升尊享(Ⅲ)年金保险（保额分红）''', '''保额分红年金''', $$
[
    "领取有三部分：基本年金给付",
    "额外年金给付",
    "满期生存给付；保单红利等"
]
$$, NULL, NULL, NULL, '''7天-65岁''', 0, '{
    "singlePayMin": 10000,
    "unit": "元",
    "annualPayMin": 1000,
    "semiAnnualPayMin": 500,
    "quarterlyPayMin": 250,
    "monthlyPayMin": 100,
    "note": "年领金额必须是120整数倍"
}', NULL, '''至85岁''', '上海、北京、四川、宁波、山东、广东、江苏、浙江、深圳、湖北、青岛', '1/5/10年', '月/季/半年/年', '年/月', '安盈添利寿险万能（保底2%）', 0.009489806765988318, 0.01218861535206095, 0.01506933459855353, 0.01653958375368503, '长安信托、中信信托（300万起）', '要求被保险人8岁以上', '支持', '投保时不支持，保全时有合理原因可修改', NULL, '不支持', '84');
INSERT INTO insurance_products_annuity (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, sales_regions, payment_period, payment_method, payout_method, universal_account, irr_15y, irr_20y, irr_30y, irr_40y, trust, intergenerational_insurance, second_insured, beneficiaries, bonus_distribution, retirement_community, total_score) VALUES ('''安联''', '''逸升汇赢（2018）终身年金保险（现金分红）''', '''现金分红年金''', $$
{
    "保险类型": {
        "text": "A型、B型"
    },
    "_": [
        {
            "percentage": 5.0,
            "note": "递增、B型年金复利5%递增等"
        }
    ]
}
$$, NULL, NULL, NULL, '''7天-60岁''', 0, '{
    "singlePayMin": 10000,
    "unit": "元",
    "annualPayMin": 5000,
    "semiAnnualPayMin": 2500,
    "quarterlyPayMin": 1250,
    "monthlyPayMin": 450,
    "note": "年领金额必须是120整数倍"
}', NULL, '''终身''', '上海、北京、四川、宁波、山东、广东、江苏、浙江、深圳、湖北、青岛', '1/5/10年', '月/季/半年/年', '年/月', '安盈添利寿险万能（保底2%）', 0.01070044387508262, 0.01467926045829926, 0.0182330197546432, 0.01984404217090541, '长安信托、中信信托（300万起）', '支持，8岁以上', '支持', '可保全修改', NULL, '不支持', '84');
INSERT INTO insurance_products_annuity (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, sales_regions, payment_period, payment_method, payout_method, universal_account, irr_15y, irr_20y, irr_30y, irr_40y, trust, intergenerational_insurance, second_insured, beneficiaries, bonus_distribution, retirement_community, total_score) VALUES ('''恒安标准人寿''', '''恒爱尊享2.0终身年金保险（分红型）''', '''终身年金''', $$
{
    "生存金": {
        "percentage": 10.0,
        "note": "合同保险金额"
    },
    "_": [
        {
            "text": "年度分红+终了红利累计入本"
        }
    ]
}
$$, NULL, NULL, '[
    {
        "text": "无健康告知"
    }
]', '''28天-60周岁''', 0, '{
    "singlePayMin": 50000,
    "unit": "元",
    "note": "3年交30000元；5年交20000元"
}', NULL, '''终身''', '北京、四川、山东、广东、江苏、河北、河南、深圳、辽宁、青岛', '1/3/5年', '年', '年', NULL, 0.01680269887882635, 0.01872119337706657, 0.02047432090598589, 0.02127850196164371, NULL, '支持，无年龄限制', '支持', NULL, '保额分红（年金可累计生息3.3%）', NULL, '93');
INSERT INTO insurance_products_annuity (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, sales_regions, payment_period, payment_method, payout_method, universal_account, irr_15y, irr_20y, irr_30y, irr_40y, trust, intergenerational_insurance, second_insured, beneficiaries, bonus_distribution, retirement_community, total_score) VALUES ('''泰康人寿''', '''乐鑫年年D款年金保险（分红型）''', '''终身年金''', $$
{
    "第6年给付": {
        "percentage": 100.0,
        "note": "年交保费"
    },
    "_": [
        {
            "percentage": 20.0,
            "note": "基本保额"
        },
        {
            "percentage": 30.0,
            "note": "基本保额"
        },
        {
            "text": "祝寿金等"
        }
    ]
}
$$, NULL, NULL, NULL, '''30天-70周岁''', 0, '{
    "note": "3000元"
}', NULL, '''终身''', '上海、云南、内蒙古、北京、厦门、吉林、四川、大连、天津、宁夏、宁波、安徽、山东、山西、广东、广西、新疆、江苏、江西、河北、河南、浙江、海南、深圳、湖北、湖南、福建、贵州、辽宁、重庆、陕西、青岛、黑龙江', '1/3/5/10/15/20年', '年', '年', NULL, 0.009511253926818286, 0.01207486133502544, 0.01444006435693335, 0.01554289294667344, '多家（中信、长安等；200万起）', '要求8岁以上', '可设置', '可申请修改', '现金分红（累计生息2%）', '多选方案', '63');
INSERT INTO insurance_products_annuity (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, sales_regions, payment_period, payment_method, payout_method, universal_account, irr_15y, irr_20y, irr_30y, irr_40y, trust, intergenerational_insurance, second_insured, beneficiaries, bonus_distribution, retirement_community, total_score) VALUES ('''长城人寿''', '''八达岭赤兔版年金保险（分红型）''', '''终身年金''', $$
[
    {
        "text": "同“甄选年金保险”条款，分红模式相同"
    }
]
$$, NULL, NULL, NULL, '''30天-70周岁''', 0, '{
    "singlePayMin": 10000,
    "unit": "元"
}', NULL, '''终身（普惠/幸福/疾速）/至105岁（尊享）''', '北京、四川、天津、安徽、山东、广东、江苏、河北、河南、湖北、湖南、重庆、陕西、青岛', '1/3/5/6/7/10/15/20年', '年', '月/年（疾速计划仅可年领）', NULL, 0.02358722462522467, 0.0263411497267434, 0.02904819576415263, 0.03048247281847316, '北京信托、五矿信托、万向信托（300万起）', '要求8岁以上', '支持', '直系', '现金分红（累计生息2%）', NULL, '75');
INSERT INTO insurance_products_annuity (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, sales_regions, payment_period, payment_method, payout_method, universal_account, irr_15y, irr_20y, irr_30y, irr_40y, trust, intergenerational_insurance, second_insured, beneficiaries, bonus_distribution, retirement_community, total_score) VALUES ('''招商仁和人寿''', '''鑫福来年金保险''', '''年金保险''', $$
{
    "生存保险金": {
        "percentage": 3.0,
        "note": "累计保费"
    },
    "_": [
        {
            "percentage": 100.0,
            "note": "基本保额"
        }
    ]
}
$$, NULL, NULL, NULL, '''28天-75周岁''', 0, '{
    "singlePayMin": 50000,
    "unit": "元",
    "note": "期交10000元"
}', NULL, '''15年''', '北京、广东、江苏、河南、深圳', '1/3/5年', '年', '一次性领取', '招管家3.0', 0.009511253926818286, 0.01207486133502544, 0.01150567209645192, 0.01752634179163337, NULL, NULL, NULL, NULL, NULL, NULL, '57');
INSERT INTO insurance_products_annuity (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, sales_regions, payment_period, payment_method, payout_method, universal_account, irr_15y, irr_20y, irr_30y, irr_40y, trust, intergenerational_insurance, second_insured, beneficiaries, bonus_distribution, retirement_community, total_score) VALUES ('''中华联合人寿''', '''中华红（钻石版）''', '''终身年金''', $$
{
    "生存保险金": {
        "percentage": 100.0,
        "note": "基本保额"
    }
}
$$, NULL, '[
    {
        "text": "简易告知"
    }
]', NULL, '''30天-70周岁''', 0, '{
    "note": "最低保费10000元"
}', NULL, '''终身''', '北京、四川、天津、新疆、河北', '1/3/5/10年', '年', '年', NULL, 0.02286605883406301, 0.02541578816716039, 0.02773564996392941, 0.0287874721872694, '中粮信托、英大信托', NULL, NULL, NULL, NULL, NULL, '84');
INSERT INTO insurance_products_annuity (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, sales_regions, payment_period, payment_method, payout_method, universal_account, irr_15y, irr_20y, irr_30y, irr_40y, trust, intergenerational_insurance, second_insured, beneficiaries, bonus_distribution, retirement_community, total_score) VALUES ('''同方全球人寿''', '''[同耀鑫享3.0]年金保险（分红型）''', '''现金分红年金''', $$
{
    "计划一": {
        "percentage": 100.0,
        "note": "保额"
    },
    "计划二": {
        "percentage": 100.0,
        "note": "、后200%保额"
    }
}
$$, NULL, NULL, NULL, '''7天-75周岁''', 0, '{
    "note": "最低保费3000元"
}', NULL, '''至105周岁''', '上海、北京、四川、天津、宁波、山东、广东、江苏、河北、浙江、深圳、湖北、福建、青岛', '1/3/5/6/10年', '年/半年/季/月', '年/月', '智惠鑫选养老年金（万能型）', 0.02310084572899629, 0.02568854823382316, 0.0280648546614588, 0.02916203426387032, NULL, NULL, NULL, NULL, '现金领取/累计生息', NULL, '84');
INSERT INTO insurance_products_annuity (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, sales_regions, payment_period, payment_method, payout_method, universal_account, irr_15y, irr_20y, irr_30y, irr_40y, trust, intergenerational_insurance, second_insured, beneficiaries, bonus_distribution, retirement_community, total_score) VALUES ('''信泰人寿''', '''如意悦享年金保险''', '''年金保险''', $$
{
    "特别生存金": {
        "percentage": 7.5,
        "note": "/3年交6%/5年交4.5%累计保费"
    },
    "生存年金": {
        "percentage": 100.0,
        "note": "保额"
    },
    "满期金": {
        "percentage": 100.0,
        "note": "累计保费"
    }
}
$$, NULL, NULL, NULL, '''28天-80周岁''', 0, '{
    "singlePayMin": 10000,
    "unit": "元",
    "note": "期交5000元"
}', NULL, '''20年/30年''', '上海、北京、宁波、山东、广东、江苏、江西、河北、河南、浙江、深圳、湖北、福建、辽宁、青岛、黑龙江', '1/3/5年', '年', '年', NULL, 0.01766983493327801, 0.02025883034481213, 0.02300160524255346, 0.02437125848831023, NULL, NULL, NULL, NULL, NULL, NULL, '63');
INSERT INTO insurance_products_annuity (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, sales_regions, payment_period, payment_method, payout_method, universal_account, irr_15y, irr_20y, irr_30y, irr_40y, trust, intergenerational_insurance, second_insured, beneficiaries, bonus_distribution, retirement_community, total_score) VALUES ('''信泰人寿''', '''如意畅享年金保险''', '''终身年金''', $$
{
    "特别生存金": {
        "percentage": 12.5,
        "note": "/3年交10%/5年交7.5%累计保费"
    },
    "生存年金": {
        "percentage": 100.0,
        "note": "保额，60后150%保额"
    }
}
$$, NULL, NULL, '[
    {
        "text": "有"
    }
]', '''28天-70周岁''', 0, '{
    "singlePayMin": 10000,
    "unit": "元",
    "note": "期交5000元"
}', NULL, '''终身''', '上海、北京、宁波、山东、广东、江苏、江西、河北、河南、浙江、深圳、湖北、福建、辽宁、青岛、黑龙江', '1/3/5年', '年', '年', NULL, 0.02231333687880022, 0.02522848454838211, 0.02782772948160805, 0.0289862958181859, NULL, NULL, NULL, NULL, NULL, NULL, '75');
INSERT INTO insurance_products_annuity (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, sales_regions, payment_period, payment_method, payout_method, universal_account, irr_15y, irr_20y, irr_30y, irr_40y, trust, intergenerational_insurance, second_insured, beneficiaries, bonus_distribution, retirement_community, total_score) VALUES ('''中邮人寿''', '''邮爱一生2.0版年金保险''', '''终身年金''', $$
{
    "年金领取日": {
        "text": "第5周年/60岁"
    },
    "_": [
        {
            "text": "关爱金及生存年金、满期金给付比例详见条款"
        }
    ]
}
$$, NULL, NULL, NULL, '''30天-70周岁''', 0, '{
    "singlePayMin": 10000,
    "unit": "元",
    "note": "期交5000元"
}', NULL, '''终身''', '上海、北京、吉林、四川、天津、宁夏、安徽、山东、广东、广西、江苏、江西、河北、河南、浙江、湖北、湖南、福建、辽宁、重庆、陕西、黑龙江', '1/3/5/6/10年', '年', '年', NULL, 0.01782427644987061, 0.01872119337706657, 0.02047432090598589, 0.02127850196164371, NULL, '支持，被保人满8周岁', '支持', NULL, NULL, NULL, '63');
INSERT INTO insurance_products_annuity (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, sales_regions, payment_period, payment_method, payout_method, universal_account, irr_15y, irr_20y, irr_30y, irr_40y, trust, intergenerational_insurance, second_insured, beneficiaries, bonus_distribution, retirement_community, total_score) VALUES ('''瑞泰人寿''', '''鸿利鑫享年金保险''', '''年金保险''', $$
{
    "满期生存金": {
        "percentage": 100.0,
        "note": "基本保额"
    },
    "_": [
        {
            "text": "生存年金按投保期间及缴费方式给付比例"
        }
    ]
}
$$, NULL, NULL, NULL, '''30天-70周岁''', 0, '{
    "singlePayMin": 50000,
    "unit": "元",
    "note": "3年20000元；5年10000元"
}', NULL, '''8年/10年/15年/20年''', '上海、北京、宁波、广东、江苏、浙江、深圳、湖北、重庆、陕西', '1/3/5年', '年', '年', NULL, 0.00148947525587606, 0.008699183567316693, 0.01527490384782171, 0.01825685049197423, NULL, NULL, NULL, NULL, NULL, NULL, '63');
INSERT INTO insurance_products_annuity (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, sales_regions, payment_period, payment_method, payout_method, universal_account, irr_15y, irr_20y, irr_30y, irr_40y, trust, intergenerational_insurance, second_insured, beneficiaries, bonus_distribution, retirement_community, total_score) VALUES ('''国华人寿''', '''步步高升年金保险''', '''年金保险''', $$
{
    "生存保险金": {
        "percentage": 2.5,
        "note": "/月领0.21%累计保费"
    },
    "_": [
        {
            "percentage": 100.0,
            "note": "基本保额"
        }
    ]
}
$$, NULL, NULL, NULL, '''28天-70周岁''', 0, '{
    "note": "最低保费10000元"
}', NULL, '''至105周岁''', '上海、北京、四川、天津、安徽、山东、山西、广东、江苏、河北、河南、浙江、深圳、湖北、湖南、辽宁、重庆、青岛', '1/3/5/6/10年', '年', '年', NULL, 0.00148947525587606, 0.00874529085084541, 0.01833708144084278, 0.02209256284716687, NULL, NULL, NULL, NULL, NULL, NULL, '63');
INSERT INTO insurance_products_annuity (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, sales_regions, payment_period, payment_method, payout_method, universal_account, irr_15y, irr_20y, irr_30y, irr_40y, trust, intergenerational_insurance, second_insured, beneficiaries, bonus_distribution, retirement_community, total_score) VALUES ('''招商信诺''', '''信享盈家一号年金保险（分红型）''', '''终身年金''', $$
{
    "年金": {
        "percentage": 500.0,
        "note": "保额，第6年起100%保额"
    },
    "_": [
        {
            "percentage": 2.0,
            "note": "累计生息"
        }
    ]
}
$$, NULL, NULL, NULL, '''28天-70周岁''', 0, '{
    "singlePayMin": 150000,
    "unit": "元",
    "note": "3年50000元；5年30000元；10年15000元"
}', NULL, '''终身''', '上海、云南、北京、四川、天津、宁波、安徽、山东、广东、江苏、江西、河南、浙江、深圳、湖北、湖南、福建、辽宁、重庆、陕西、青岛', '1/3/5/10年', '年', '年/月', NULL, 0.00148947525587606, 0.005480425546649625, 0.01673868268459433, 0.02133979239979289, NULL, NULL, NULL, NULL, '累计生息', NULL, '75');
INSERT INTO insurance_products_annuity (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, sales_regions, payment_period, payment_method, payout_method, universal_account, irr_15y, irr_20y, irr_30y, irr_40y, trust, intergenerational_insurance, second_insured, beneficiaries, bonus_distribution, retirement_community, total_score) VALUES ('''长生人寿''', '''福运连连年金保险''', '''终身年金''', $$
{
    "特别生存保险金": {
        "percentage": 10.0,
        "note": "/3年30%/5年50%/10年100%累计保费"
    },
    "生存年金": {
        "percentage": 100.0,
        "note": "保额"
    }
}
$$, NULL, NULL, '[
    {
        "text": "1条"
    }
]', '''30天-80周岁''', 0, '{
    "note": "最低保费10000元"
}', NULL, '''终身''', '上海、北京、宁波、广东、江苏、浙江、深圳、湖北、重庆、陕西', '1/3/5/10年', '年', '年', NULL, 0.01639902549776395, 0.01846342010016011, 0.01833708144084278, 0.02209256284716687, NULL, NULL, NULL, NULL, NULL, NULL, '75');
INSERT INTO insurance_products_annuity (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, sales_regions, payment_period, payment_method, payout_method, universal_account, irr_15y, irr_20y, irr_30y, irr_40y, trust, intergenerational_insurance, second_insured, beneficiaries, bonus_distribution, retirement_community, total_score) VALUES ('''中韩人寿''', '''明悦金生（江南版）年金保险''', '''年金保险''', $$
{
    "年金领取日": {
        "text": "第6/10周年"
    },
    "生存年金": {
        "percentage": 100.0,
        "note": "保额"
    },
    "特别关爱金": {
        "percentage": 150.0,
        "note": "保额"
    },
    "长寿关爱金": {
        "percentage": 85.0,
        "note": "累计保费"
    }
}
$$, NULL, NULL, '[
    {
        "text": "有，7条"
    }
]', '''28天-55周岁''', 0, '{
    "singlePayMin": 10000,
    "unit": "元",
    "note": "期交5000元"
}', NULL, '''至105周岁''', '宁波、安徽、江苏、浙江', '1/3/5/6/10年', '年', '年', NULL, 0.01392417233862298, 0.01655816879847305, 0.01900669891085816, 0.02015865342225664, NULL, NULL, NULL, NULL, NULL, NULL, '75');
INSERT INTO insurance_products_annuity (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, sales_regions, payment_period, payment_method, payout_method, universal_account, irr_15y, irr_20y, irr_30y, irr_40y, trust, intergenerational_insurance, second_insured, beneficiaries, bonus_distribution, retirement_community, total_score) VALUES ('''中英人寿''', '''悦活人生年金保险''', '''终身年金''', $$
[
    {
        "text": "趸交/3/5年，第5年开始领"
    },
    {
        "text": "6年交，第6年开始领"
    },
    {
        "text": "10年交，第10年开始领"
    },
    {
        "percentage": 100.0,
        "note": "保额"
    }
]
$$, NULL, NULL, NULL, '''28天-75周岁''', 0, '{
    "singlePayMin": 100000,
    "unit": "元",
    "note": "3年30000元；5/6年20000元；10年10000元"
}', NULL, '''终身''', '上海、北京、厦门、四川、安徽、山东、广东、江苏、江西、河北、河南、深圳、湖北、湖南、福建、辽宁、陕西、青岛、黑龙江', '1/3/5/6/10年', '年', '年', NULL, 0.01983049595667041, 0.02103413998543169, 0.02213936832766117, 0.02265417098694544, NULL, NULL, NULL, NULL, NULL, NULL, '63');
INSERT INTO insurance_products_annuity (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, sales_regions, payment_period, payment_method, payout_method, universal_account, irr_15y, irr_20y, irr_30y, irr_40y, trust, intergenerational_insurance, second_insured, beneficiaries, bonus_distribution, retirement_community, total_score) VALUES ('''复星保德信''', '''星享未来年金保险''', '''年金保险''', $$
[
    {
        "text": "计划一/计划二年金及满期金给付比例详见条款"
    }
]
$$, NULL, NULL, NULL, '''30天-70周岁''', 0, '{
    "singlePayMin": 30000,
    "unit": "元",
    "note": "3年15000元；5/10年10000元"
}', NULL, '''至106周岁''', '上海、北京、四川、山东、江苏、河南、青岛', '1/3/5/10年', '年', '年', NULL, 0.01576598732636292, 0.02065822947749063, 0.02213936832766117, 0.02265417098694544, NULL, NULL, '支持', NULL, NULL, NULL, '63');
