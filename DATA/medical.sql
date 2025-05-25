
-- =========================================================
-- Table: insurance_products_medical (Medical Insurance Products)
-- =========================================================
DROP TABLE IF EXISTS insurance_products_medical;

CREATE TABLE insurance_products_medical (
    product_id SERIAL PRIMARY KEY,
    company_name VARCHAR(100),
    product_name VARCHAR(150),
    insurance_type VARCHAR(50),
    coverage_content JSONB,
    exclusion_clause JSONB,
    renewable BOOLEAN,
    underwriting_rules JSONB,
    entry_age JSONB,
    deductible DECIMAL(12,2),
    premium JSONB,
    coverage_amount JSONB,
    coverage_period VARCHAR(50),
    occupation VARCHAR(50),
    payment_period VARCHAR(20),
    coverage_region JSONB,
    hospital_scope JSONB,
    reimbursement_scope JSONB,
    reimbursement_ratio JSONB,
    waiting_period VARCHAR(100),
    remarks JSONB,
    age INTEGER,
    plan_choice JSONB,
    discount JSONB,
    company_intro TEXT,
    drug_list TEXT,
    service_manual TEXT,
    clause_link TEXT,
    total_score DECIMAL(6,2)
);

INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('众安保险', '安惠保（升级版）', '百万医疗险', $$
{
    "一般住院医疗保险金": "社保内：200万/社保外：100万",
    "重大疾病住院医疗保险金": "100万",
    "特殊门诊": "含",
    "门诊手术": "/",
    "住院前后门急诊": "/",
    "其他责任": "1、质子重离子医疗保险金：100万/0免赔，100%赔付，床位费：1500元/天",
    "_": [
        "2、特定药品医疗保险金：80万，0免赔，50种特药+2cart，80%赔付"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "105周岁"
}
$$, 20000, $$
{
    "amount": 99,
    "unit": "元"
}
$$, $$
{
    "amount": 2000000,
    "unit": "元"
}
$$, '1年', '无限制', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上公立医院普通部/上海质子重离子医院"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 0.8,
    "note": "必须以社保身份参保，结算，否则不赔付"
}
$$, '30天', $$
{
    "text": "无健康告知，承担一般既往症，不包含以下五类重大既往症：\n （1）肿瘤类：恶性肿瘤、白血病、淋巴瘤、颅内肿瘤或占位； （2）肝肾疾病类：肾功能不全，肝硬化、肝功能不全； （3）心脑血管及糖脂代谢疾病类：先天性心脏病、缺血性心脏病(冠心病、心肌梗死)、慢性心功能不全(心功能Ⅲ级及以上)、主 动脉夹层、心肌病，肺动脉高压，脑血管疾病(脑梗死、脑栓塞、脑出血)，帕金森病，高血压病(3级)或伴有并发症，糖尿病且伴 有并发症； （4）肺部疾病类：支气管扩张，慢性阻塞性肺病、慢性呼吸衰竭； （5）其他：动脉瘤，系统性红斑狼疮，再生障碍性贫血，胰腺炎，溃疡性结肠炎，艾滋病或HIV阳性，克罗恩病，骨坏死，脊椎/ 脊柱疾病。"
}
$$, NULL, $$
{
    "plan": "99"
}
$$, $$
{
    "familyDiscounts": [
        {
            "members": 2,
            "discount": 9.5
        },
        {
            "members": 3,
            "discount": 0.9
        },
        {
            "members": 4,
            "discount": 8.5
        },
        {
            "members": 7,
            "discount": 0.8
        }
    ]
}
$$, $$
{
    "text": "全国首家互联网保险公司\n由蚂蚁金服、中国平安和腾讯于2013年联合发起设立"
}
$$, NULL, NULL, NULL, 83.5);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('众安保险', '众民保普惠百万医疗', '百万医疗险', $$
{
    "一般住院医疗保险金": "200万",
    "重大疾病住院医疗保险金": "200万",
    "特殊门诊": "含",
    "门诊手术": "/",
    "住院前后门急诊": "/",
    "其他责任": "1、质子重离子医疗保险金：200万/0免赔，80%赔付，发生理赔后第二年不续保",
    "_": [
        "2、特定药品医疗保险金：80万，0免赔，50种特药+2cart/社保内药物未以社保身份结算50%赔付，社保外药物，均80%赔付，发生理赔第二年不续保",
        "3、重大疾病跨省异地转诊公共交通费：1万，0免赔，100%赔付",
        "4、救护车费用保险金：1000元，0免赔",
        "5、互联网药品保险金：2000元，0免赔，次限额200元，50%赔付"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "80周岁"
}
$$, 10000, $$
{
    "amount": 418,
    "unit": "元"
}
$$, $$
{
    "amount": 2000000,
    "unit": "元"
}
$$, '1年', '无限制', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上公立医院普通部/上海质子重离子医院"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 0.8,
    "note": "未以社保身份结算50%"
}
$$, '30天', $$
{
    "text": "无健康告知，承担一般既往症，不包含以下五类重大既往症：\n （1）肿瘤类：恶性肿瘤、白血病、淋巴瘤、颅内肿瘤或占位； （2）肝肾疾病类：肾功能不全，肝硬化、肝功能不全； （3）心脑血管及糖脂代谢疾病类：先天性心脏病、缺血性心脏病(冠心病、心肌梗死)、慢性心功能不全(心功能Ⅲ级及以上)、主 动脉夹层、心肌病，肺动脉高压，脑血管疾病(脑梗死、脑栓塞、脑出血)，帕金森病，高血压病(3级)或伴有并发症，糖尿病且伴 有并发症； （4）肺部疾病类：支气管扩张，慢性阻塞性肺病、慢性呼吸衰竭； （5）其他：动脉瘤，系统性红斑狼疮，再生障碍性贫血，胰腺炎，溃疡性结肠炎，艾滋病或HIV阳性，克罗恩病，骨坏死，脊椎/ 脊柱疾病。"
}
$$, NULL, $$
{
    "plan": "418"
}
$$, $$
{
    "familyDiscounts": [
        {
            "members": 2,
            "discount": 9.5
        },
        {
            "members": 3,
            "discount": 0.9
        },
        {
            "members": 4,
            "discount": 8.5
        },
        {
            "members": 7,
            "discount": 0.8
        }
    ]
}
$$, $$
{
    "text": "全国首家互联网保险公司\n由蚂蚁金服、中国平安和腾讯于2013年联合发起设立"
}
$$, NULL, NULL, NULL, 82.0);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('众安保险', '众民保·百万医疗险', '百万医疗险', $$
{
    "一般住院医疗保险金": "300万",
    "重大疾病住院医疗保险金": "300万",
    "特殊门诊": "含",
    "门诊手术": "/",
    "住院前后门急诊": "/",
    "其他责任": "1、质子重离子医疗保险金：300万/0免赔，80%/臻选版100赔付，床位费1500元/天，发生理赔后第二年不续保",
    "_": [
        "2、院外特定药品医疗保险金：300万，0免赔，80%/臻选版100%赔付，经典版50种/臻选版122种特药+2款car-t；",
        "社保内药物未以社保身份结算50%/臻选版60%赔付，社保外药物，均80%/臻选版100%赔付，发生理赔后可正常续保",
        "3、重大疾病跨省异地转诊公共交通费：1万/2万，0免赔，100%赔付",
        "4、救护车费用保险金：1000元，0免赔，100%赔付",
        "默认需要以公费医疗身份参保"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "105周岁"
}
$$, 10000, $$
{
    "amount": 368,
    "unit": "元"
}
$$, $$
{
    "text": "总额600万（社保内外各300万）"
}
$$, '1年', '无限制', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上公立医院普通部/上海质子重离子医院"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 0.8,
    "note": "臻选版100%，未以社保结算50%"
}
$$, '30天（意外无等待期）', $$
{
    "text": "（1）不承担初次投保前、等待期内或非连续重新投保前已罹患的以下1-5类疾病，及因该疾病或并发症导致的医疗费用；不承担初次投保前或非连续重新投保前已发生意外事故导致的相关医疗费用：\na. 肿瘤类：恶性肿瘤*、颅内肿瘤或占位、脊髓肿瘤或占位、肝占位； \nb. 肝肾疾病类：慢性肾病（CKD4期及以上）、肝硬化、肝衰竭；\nc. 心脑血管及糖脂代谢疾病类：冠心病、心肌梗死、心功能不全(心功能Ⅲ级及以上)、主动脉夹层、心肌病、房颤/房扑、肺动脉高压、脑梗死、脑出血、心瓣膜病、高血压伴并发症、糖尿病伴并发症； \nd. 肺部疾病类：慢性阻塞性肺病、呼吸衰竭、间质性肺病；\ne. 其他：帕金森病，动脉瘤，系统性红斑狼疮，再生障碍性贫血、骨髓增生异常综合征，嗜（噬）血细胞综合征，胰腺炎，溃疡性结肠炎，克罗恩病，骨坏死，脊椎/脊柱/胸廓疾病*，癫痫，瘫痪；\nf. 意外：初次投保前或非连续重新投保前已发生的意外事故。\n释义：1）恶性肿瘤：包括癌、肉瘤，含白血病、淋巴瘤。指首次投保前已罹患恶性肿瘤的持续、复发、转移。明确为投保后新发的恶性肿瘤不在此范围内，可正常赔付；2）脊椎/脊柱/胸廓疾病：包括脊柱侧弯、胸廓畸形、椎间盘疾患、椎骨滑脱、椎管狭窄、脊髓型颈椎病。"
}
$$, NULL, $$
{
    "plan": "臻选版"
}
$$, $$
{
    "familyDiscounts": [
        {
            "members": 2,
            "discount": 9.5
        },
        {
            "members": 3,
            "discount": 0.9
        },
        {
            "members": 4,
            "discount": 8.5
        },
        {
            "members": 7,
            "discount": 0.8
        }
    ]
}
$$, $$
{
    "text": "全国首家互联网保险公司\n由蚂蚁金服、中国平安和腾讯于2013年联合发起设立"
}
$$, NULL, NULL, NULL, 83.9);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('安盛天平', '卓越守护百万医疗保险（高血压版）标准版', '特定疾病医疗险', $$
{
    "一般住院医疗保险金": "100万",
    "重大疾病住院医疗保险金": "100万",
    "特殊门诊": "含",
    "门诊手术": "/",
    "住院前后门急诊": "前7天后7天"
}
$$, NULL, NULL, NULL, $$
{
    "min": "18",
    "max": "55周岁"
}
$$, 10000, $$
{
    "amount": 372,
    "unit": "元"
}
$$, $$
{
    "amount": 1000000,
    "unit": "元"
}
$$, '1年', '除特殊职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上公立医院普通部/上海质子重离子医院"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 0.9
}
$$, '90天', $$
{
    "text": "初次投保时确诊患有原发性高血压，能正常工作或生活，且正在进行高血压疾病管理的慢性病患者"
}
$$, NULL, $$
{
    "plan": "有社保"
}
$$, NULL, NULL, NULL, NULL, NULL, 78.8);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('安盛天平', '卓越守护百万医疗保险（糖尿病版）标准版', '特定疾病医疗险', $$
{
    "一般住院医疗保险金": "100万",
    "重大疾病住院医疗保险金": "100万",
    "特殊门诊": "含",
    "门诊手术": "/",
    "住院前后门急诊": "前7天后7天"
}
$$, NULL, NULL, NULL, $$
{
    "min": "18",
    "max": "55周岁"
}
$$, 10000, $$
{
    "amount": 372,
    "unit": "元"
}
$$, $$
{
    "amount": 1000000,
    "unit": "元"
}
$$, '1年', '除特殊职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上公立医院普通部/上海质子重离子医院"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 0.9
}
$$, '90天', $$
{
    "text": "初次投保时确诊患有2型糖尿病，能正常工作或生活，且正在进行糖尿病疾病管理的慢性病患者"
}
$$, NULL, $$
{
    "plan": "有社保"
}
$$, NULL, NULL, NULL, NULL, NULL, 78.8);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('安盛天平', '卓越孝欣老年综合医疗保险（个人版）标准版', '特定疾病医疗险', $$
{
    "一般住院医疗保险金": "200万（须每年通过体检）",
    "重大疾病住院医疗保险金": "200万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前7天后30天"
}
$$, NULL, NULL, NULL, $$
{
    "min": "46",
    "max": "80周岁"
}
$$, 10000, $$
{
    "amount": 0,
    "unit": "元"
}
$$, $$
{
    "amount": 2000000,
    "unit": "元"
}
$$, '1年', '除特殊职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上公立医院普通部/上海质子重离子医院"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '90天', $$
{
    "text": "主险为防癌医疗险，首保按照要求通过体检的，可除责部分或者完全承保拓展一般医疗险责任，体检由安盛指定医院，客户不需要承担费用。续保也需要进行体检，若体检不通过标准，责仅保留防癌医疗险或者全额退费"
}
$$, NULL, $$
{
    "plan": "有社保"
}
$$, NULL, NULL, NULL, NULL, NULL, 80.8);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('平安健康', '平安e生保2023（慢病版）互联网医疗保险产品组合', '特定疾病医疗险', $$
{
    "一般住院医疗保险金": "200万",
    "重大疾病住院医疗保险金": "200万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、质子重离子医疗保险经济：400万，0免赔，床位费1500元/天",
    "_": [
        "2、院外恶性肿瘤特定药品：188种，400万，0免赔",
        "3、院外恶性肿瘤特定用药基因检测费用：400万，0免赔，指定机构100%，非指定机构60%",
        "可选1：特定疾病特需医疗：6种疾病，400万，0免赔，100%赔付，等待期90天，床位费1500元/天（住院，指定门诊，住院前后30天，门诊手术，涵盖恶性肿瘤—重度、重大器官衰竭、造血干细胞功能损害或造血系统恶性肿瘤、严重非恶性颅内肿瘤、严重 III 度烧伤、 重型再生障碍性贫血）",
        "可选2：亚洲特定治疗医疗：恶性肿瘤400万，五项合计200万，0免赔（70%，直付）",
        "恶性肿瘤—重度治疗：400万",
        "特定治疗疾病（因严重冠心病需要进行冠状动脉旁路移植手术、因心脏疾病需要进行心脏瓣膜置换或修复、神经外科手术、活体器官移植、骨髓移植）：200万"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "18",
    "max": "65周岁"
}
$$, 10000, $$
{
    "amount": 680,
    "unit": "元"
}
$$, $$
{
    "amount": 2000000,
    "unit": "元"
}
$$, '1年', '除特殊职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级以上（含二级）公立医院普通部、上海质子重离子医院、河北一洲肿瘤医院、淄博万杰肿瘤医院、甘肃省武威肿瘤医院、上海交通大学医学院附属瑞金医院肿瘤质子中心、指定医院"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 0.7,
    "note": "未经社保42%）/计划二和三100%（未经社保60%"
}
$$, '30天', NULL, NULL, $$
{
    "plan": "有无社保"
}
$$, NULL, NULL, NULL, NULL, NULL, 79.7);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('平安健康', '平安互联网e惠保医疗保险', '特定疾病医疗险', $$
{
    "一般住院医疗保险金": "200万",
    "重大疾病住院医疗保险金": "200万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、质子重离子医疗：400万，0免赔，床位费1500元/天",
    "_": [
        "2、院外恶性肿瘤特的药品：400万，188种，0免赔，100%赔付（以社保身份参保，未以社保身份结算60%）",
        "3、特定药品基因检测：2万，0免赔，不限定机构"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "18",
    "max": "70周岁"
}
$$, 10000, $$
{
    "amount": 756,
    "unit": "元"
}
$$, $$
{
    "amount": 4000000,
    "unit": "元"
}
$$, '1年', '除特殊职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "中国大陆二级以上（含二级）公立医院普通部、扩展医院（由医疗网络持续提供扩展医院清单）、质重医院（上海质子重离子医院、河北一洲肿瘤医院、淄博万杰肿瘤医院、甘肃省武威肿瘤医院、上海交通大学医学院附属瑞金医院肿瘤质子中心）"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 0.8,
    "note": "未经社保48%）/计划二100%（未经社保60%"
}
$$, '30天', $$
{
    "text": "恶性肿瘤5年后可投保，不承担重大既往症，承担一般既往症\n\n1、 过去6个月内：您是否进行过核磁共振（MRI）、钼靶、PET-CT、病理检查或活检、穿刺、内窥镜、血液涂片细胞学的检查？或被医生建议进行肿瘤标志物、CT、核磁共振（MRI）、钼靶、PET-CT、病理检查或活检、穿刺、内窥镜、血液涂片细胞学的检查？\n2、 过去2年内：您是否曾被医生建议住院或手术？是否接受过住院手术或单次住院超过5天？\n3、 过去5年内您是否患有或被告知患有下列疾病：\n1）肿瘤：恶性肿瘤、原位癌、颅内肿瘤或颅内占位；\n2）呼吸循环系统疾病：慢性阻塞性肺疾病（COPD）、冠心病、心肌梗死、心脏瓣膜疾病、心肌病、心功能不全II级及以上、脑中风、脑血管畸形、高血压且存在清单中疾病（请点击链接查看疾病清单）、糖尿病且存在清单中疾病（请点击链接查看疾病清单）；\n\n3）其他疾病：自身免疫性肝炎、慢性病毒性肝炎且伴有肝功能异常、肝纤维化或肝硬化、慢性肾功能不全、再生障碍性贫血、瘫痪、系统性红斑狼疮、股骨头/颈坏死、重度骨性关节炎、接受过重大器官（心、肝、肾、肺）移植或造血干细胞移植。"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, NULL, NULL, NULL, NULL, NULL, 81.2);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('平安健康', '平安互联网终身防癌（费率可调）保险产品组合', '特定疾病医疗险', $$
{
    "一般住院医疗保险金": "/",
    "重大疾病住院医疗保险金": "400万（恶性肿瘤（包括恶性肿瘤重度和轻度）及原位癌）",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、恶性肿瘤及原位癌医疗保险金：400万，指定医院100%赔付，非指定医院90%赔付（购买有社保版本但就医时未使用社保，赔付60%）",
    "_": [
        "2、院外恶性肿瘤特定药品费用保险金：400万，指定医院100%赔付，非指定医院90%赔付（购买有社保版本但就医时未使用社保，赔付60%）",
        "3、质子重离子医疗保险（首年赠送）：100万；床位费：1500原/天，1年期，国内质子重离子医院",
        "；赔付比例100%"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "28天",
    "max": "70周岁"
}
$$, 0, $$
{
    "amount": 143,
    "unit": "元"
}
$$, $$
{
    "text": "年400万/保证续保期间总限额800万"
}
$$, '1年（终身保证续保）', '除特殊职业', '每年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级以上（含二级）公立医院普通部"
}
$$, NULL, $$
{
    "ratio": 1.0,
    "note": "赔付，非指定医院90%赔付（购买有社保版本但就医时未使用社保，赔付60%"
}
$$, '90天（续保无等待期）', NULL, NULL, $$
{
    "plan": "有无社保"
}
$$, $$
{
    "text": "支持家庭单，2人及以上95折"
}
$$, NULL, NULL, NULL, NULL, 82.0);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('太平洋健康', '蓝医保·终身防癌医疗险', '特定疾病医疗险', $$
{
    "一般住院医疗保险金": "/",
    "重大疾病住院医疗保险金": "400万（恶性肿瘤（包括恶性肿瘤重度和轻度）及原位癌）",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、恶性肿瘤及原位癌医疗保险金：400万",
    "_": [
        "2、重度质子重离子医疗保险金：400万",
        "3、特定药品费用医疗保险金：400万",
        "4、恶性肿瘤异地转诊客运公共交通费用保险金：1万",
        "健康服务：",
        "1.防癌早筛体检套餐",
        "2.防癌早定制服务",
        "3.在线健康咨询",
        "4.专家门诊预约",
        "5.全程就医陪同",
        "6.专家病房预约及专家手术预约",
        "7.海外二次诊疗8.多学科会诊",
        "9.住院垫付",
        "10.特药垫付",
        "11.院后照护指导"
    ]
}
$$, NULL, NULL, $$
{
    "text": "https://kdocs.cn/l/cvD94eHrCCFm"
}
$$, $$
{
    "min": "30天",
    "max": "70周岁"
}
$$, 0, $$
{
    "amount": 159,
    "unit": "元"
}
$$, $$
{
    "text": "年400万/保证续保期间总限额800万"
}
$$, '1年（终身保证续保）', '无限制', '每年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "中国境内(不含香港、澳门和台湾地区)经国家卫生行政管理部门正式评定的二级以上(含二级)属事业单位编制的公立医院，接受质子、重离子放射治疗的指定医疗机构为上海市质子重离子医院"
}
$$, NULL, $$
{
    "ratio": 1.0,
    "note": "其他90%（未经社保再×60%"
}
$$, '90天', NULL, NULL, $$
{
    "plan": "有无社保"
}
$$, $$
{
    "text": "最少2人，最多6人，95折"
}
$$, NULL, NULL, NULL, NULL, 82.4);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('众安保险', '尊享e生2025版', '特定疾病医疗险', $$
{
    "一般住院医疗保险金": "一般医疗及外购药械费用300万",
    "重大疾病住院医疗保险金": "重大疾病及外购药械费用300万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、一般住院医疗床位费：1000元/天（不包括单人病房，套房，家庭病房）",
    "_": [
        "2、一般门急诊医疗及外购药械费用医疗保险金：300万（与一般医疗及外购药费用保险金共享），100%赔付社保罚则60%",
        "3、恶性肿瘤先进疗法医疗责任：600万（恶性肿瘤质子重离子医疗费用/恶性肿瘤硼中子俘获疗法医疗费用/恶性肿瘤光免疫疗法医疗费用），床位费限1500元/天，0免赔，100%赔付",
        "4、特定药品费用医疗保险金：600万；恶性肿瘤特定药品基因检测费用限额1万；罕见病特定药品费用/特定进口药品费用及特定医疗机构住院医疗费用/指定疾病急需药品费用无单项限额；",
        "0免赔，100%赔付，社保罚则60%；社保目录外药品100%赔付",
        "5、重大疾病异地转诊公共交通费用保险金：1万，0免赔，100%赔付",
        "6、重大疾病住院护工费用保险金：500元/天，单次住院限30天，同一保险年度累计限30天",
        "可选1、家庭共享免赔额",
        "可选2、重大疾病关爱金：5万/给付/二级及二级以上公立医院;",
        "可选3、重大疾病特需拓展：300万",
        "可选4、住院医疗费用补充：1万/100%赔付/免赔额0",
        "可选5、门急诊医疗：1万/50%赔付/次免赔额0元/次限额300元",
        "可选6、住院津贴加油包：一般住院津贴100元/天，次免赔3天，单次累计限30天，年度累计限60天/ICU重症监护病房住院津贴500元/天，次免赔0天，单次住院限30天，年度限60天",
        "可选7、药省保加油包：1000元，50%赔付，月度赔付限100元，限指定互联网平台药店",
        "可选8、附加家财险"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "70周岁"
}
$$, 10000, $$
{
    "amount": 308,
    "unit": "元"
}
$$, $$
{
    "text": "总限额600万"
}
$$, '1年', '无', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上的公立医院及本保单约定的指定民营医疗机构普通部"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '30天', $$
{
    "text": "自带一般门急诊，责任内感冒发烧等门急诊无需住院也可以赔付，免赔额10000元和住院一般医疗共享；\n\n慢病人群（三高），结节人群等可投保，需要人工核保，核保之后条款责任，赔付比例会变更，具体以人工核保为准；"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, $$
{
    "familyDiscounts": [
        {
            "members": 2,
            "discount": 9.5
        },
        {
            "members": 3,
            "discount": 0.9
        },
        {
            "members": 4,
            "discount": 8.5
        },
        {
            "members": 7,
            "discount": 0.8
        }
    ]
}
$$, $$
{
    "text": "全国首家互联网保险公司\n由蚂蚁金服、中国平安和腾讯于2013年联合发起设立"
}
$$, NULL, NULL, NULL, 82.2);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('太平财险', '太平医保无忧百万医疗险2021版计划二', '特定疾病医疗险', $$
{
    "一般住院医疗保险金": "300万",
    "重大疾病住院医疗保险金": "300万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前7天后30天",
    "其他责任": "1、癌症特效药院外费用：同年度保额，社保目录内经社保结算100%赔付，未经社保结算60%赔付/社保目录外100%赔付",
    "_": [
        "2、癌症住院津贴：100元/天，单次最高90天，全年累计限180天，无免赔天数，社保目录内经社保结算100%赔付，未经社保结算60%赔付/社保目录外100%赔付",
        "3、意外身故：2万，给付",
        "可选1、重大疾病一次性给付金：（拓展疾病人群不可投保）0-45周岁：5万/10万；46-60周岁：5万；61周岁以上：不可承保",
        "可选2、癌症赴日责任：100万，70%赔付，限10次",
        "可选3、特需医疗责任：100万，100%赔付，诊疗费与床位费日累计限额1500元",
        "可选拓展责任：①0免赔/②甲状腺1-3级/③糖尿病/④高血压/⑤乳腺结节1-2级⑥慢性肾病/⑦慢性肝病",
        "以上拓展责任只可选择一项"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "65周岁（0免赔和慢病版至60周岁）"
}
$$, 10000, $$
{
    "amount": 371,
    "unit": "元"
}
$$, $$
{
    "amount": 3000000,
    "unit": "元"
}
$$, '1年', '除特殊职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上公立医院普通部/上海质子重离子医院"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "text": "以拓展版本为准"
}
$$, '30天', $$
{
    "text": "1、拓展版本只可选择一项\n2、支持慢病和结节加费承保"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, NULL, NULL, NULL, NULL, NULL, 82.0);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('安盛天平', '卓越守护百万医疗保险2022升级版尊贵款', '百万医疗险', $$
{
    "一般住院医疗保险金": "300万",
    "重大疾病住院医疗保险金": "300万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、重大疾病住院津贴：150元/天，限180天",
    "_": [
        "2、恶性肿瘤院外特药医疗费用保险金：150万（102种国内特药+15种海外特药+2种car-t）",
        "可选1：特需住院医疗保险金：100万，床位费1500元/天",
        "可选2：恶性肿瘤特需医疗：100万，床位费1500元/天",
        "可选3：恶性肿瘤赴日医疗：100万，床位费1500元/天",
        "可选4：重大疾病保险金：5万",
        "可选5：特定医疗器械费用医疗保险金：100万"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "28天",
    "max": "65周岁"
}
$$, 10000, $$
{
    "amount": 340,
    "unit": "元"
}
$$, $$
{
    "amount": 3000000,
    "unit": "元"
}
$$, '1年', '除特殊职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上公立医院普通部/上海质子重离子医院"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '30天', $$
{
    "text": "1、一般特需和癌症特需仅可选择一项"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, $$
{
    "text": "支持家庭单投保，2人单95折、3-5人单9折"
}
$$, $$
{
    "text": "安盛全资控股子公司\n全球顶尖健康险供应商\n2019年位列全球五百强第46位\n安盛（全球最大保险集团）\n全球顶尖健康险供应商\n全球统一续保政策\n风险评级：BB（2023年1季度）"
}
$$, NULL, NULL, NULL, 81.7);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('平安健康', '平安e生保互联网2025医疗保险', '百万医疗险', $$
{
    "一般住院医疗保险金": "200万",
    "重大疾病住院医疗保险金": "200万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、质子重离子医疗保险：400万/0免赔/100%赔付/床位费1500元/天",
    "_": [
        "2、院外特定恶性肿瘤费用医疗保险金（212种）：400万，0免赔，100%赔付",
        "3、院外恶性肿瘤特定用药基因检测费用保险金：2万，0免赔，100%赔付",
        "4、院外特定疾病急需药品费用保险金：5万，120种特定疾病，4大类药，同主险免赔额"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "28天",
    "max": "65周岁"
}
$$, 10000, $$
{
    "amount": 823,
    "unit": "元"
}
$$, $$
{
    "text": "总赔付限额400万"
}
$$, '1年', '除特殊职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级以上（含二级）公立医院普通部，质子重离子医院（上海质子重离子医院、河北一洲肿瘤医院、淄博万杰肿瘤医院、甘肃省武威肿瘤医院、上海交通大学医学院附属瑞金医院肿瘤质子中心、中国科学技术大学附属第一医院离子医学中心（合肥离子医学中心）、山东省肿瘤医院质子中心、武汉协和医院质子医学中心）"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '30天（意外无等待期）', NULL, NULL, $$
{
    "plan": "有无社保"
}
$$, $$
{
    "familyDiscounts": [
        {
            "members": 2,
            "discount": 9.5
        },
        {
            "members": 3,
            "discount": 0.9
        }
    ]
}
$$, $$
{
    "text": "国内保险老三家\n世界五百强前五十"
}
$$, NULL, NULL, NULL, 81.1);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('泰康在线', '泰康在线百万医疗险', '百万医疗险', $$
{
    "一般住院医疗保险金": "300万",
    "重大疾病住院医疗保险金": "600万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、质子重离子医疗保险金：600万"
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天（不含）",
    "max": "65周岁"
}
$$, 10000, $$
{
    "amount": 283,
    "unit": "元"
}
$$, $$
{
    "amount": 3000000,
    "unit": "元"
}
$$, '1年', '除高危职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "医疗相关保障限定为中华人民共和国境内合法经营的二级以上（含二级）公立医院（不含国际医疗及特需部）；质子重离子医疗保险责任的医疗机构限定为“上海质子重离子医院”"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '30天', NULL, NULL, $$
{
    "plan": "有无社保"
}
$$, NULL, NULL, NULL, NULL, NULL, 81.7);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('中国人寿', '中国人寿特安心医疗保险（2020版）尊贵款', '百万医疗险', $$
{
    "一般住院医疗保险金": "300万",
    "重大疾病住院医疗保险金": "300万（80种重大疾病）",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、轻症疾病一次性给付（20种）：1万，等待期90天，无免赔",
    "_": [
        "可选1、80种重大疾病一次性给付：5万/10万，等待期90天，无免赔",
        "医疗服务：",
        "1.国内专家门诊绿色通道，门诊/住院，年度内各1次",
        "2.多学科会诊；",
        "3.术后护理，10次；",
        "4.国内住院直付，保险金额内垫付；",
        "5.肿瘤基因检测+靶向药报告解读；",
        "6.靶向药院外药房直付"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "28天",
    "max": "65周岁"
}
$$, 10000, NULL, $$
{
    "text": "300万（总赔付限额600万）"
}
$$, '1年', '1-4类', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级以上（含二级）公立医院或者保险人指定或认可的医疗机构"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '一般医疗保险、80种重大疾病医疗保险、特需责任等待期30天；轻症疾病、重疾一次性给付等待期90天', NULL, NULL, $$
{
    "plan": "有无社保"
}
$$, NULL, NULL, NULL, NULL, NULL, 76.3);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('太平洋保险', '蓝医保长期医疗险（20年保证续保）', '长期百万医疗险', $$
{
    "一般住院医疗保险金": "200万",
    "重大疾病住院医疗保险金": "400万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、特定疾病医疗保险金：200万",
    "_": [
        "2、重大疾病医疗保险金：400万",
        "3、恶性肿瘤-质子重离子医疗保险金：400万",
        "4、重大疾病医疗保险金：1万，保证续保期间给付1次",
        "可选2、重大疾病保险金：1万/2万/3万/4万/5万",
        "可选3、特定疾病特需医疗保险：400万，5种疾病",
        "可选4、住院津贴医疗保险金：100元/天，每次住院免赔3天，最高给付30天，年度最高给付60天",
        "重症监护病房住院津贴医疗保险金：400元/天，每次住院免赔0天，最高给付30天，年度最高给付60天",
        "可选5、失能收入损失保险：",
        "保额可选5千元或1万元/月",
        "给付期限可选60个月或120个月",
        "保险期间1年，保证续保20年"
    ],
    "可选1、特定药品费用医疗保险金": "200万，100%赔付，社保罚则60%"
}
$$, NULL, NULL, $$
{
    "text": "https://kdocs.cn/l/cgSm5qXBgHeh"
}
$$, $$
{
    "min": "30天",
    "max": "65周岁"
}
$$, 10000, $$
{
    "amount": 247,
    "unit": "元"
}
$$, $$
{
    "text": "年200万/保证续保期间总限额800万"
}
$$, '1年', '1-4类', '20年（费率可调）', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上公立医院普通部/上海质子重离子医院"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '90天', $$
{
    "text": "保证续保20年"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, $$
{
    "text": "支持家庭单，2人起，最多6人，主险95折，附加可选无折扣"
}
$$, $$
{
    "text": "世界五百强\n中国保险老三家"
}
$$, $$
{
    "scope": "https://kdocs.cn/l/ccFm2ArO6W2l"
}
$$, $$
{
    "scope": "https://kdocs.cn/l/ckGe21tiKufQ"
}
$$, 'https://kdocs.cn/l/ccBFZslhctW0', 81.3);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('太平洋保险', '蓝医保长期医疗险（好医好药版）产品组合', '长期百万医疗险', $$
{
    "一般住院医疗保险金": "200万",
    "重大疾病住院医疗保险金": "400万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、特定疾病医疗保险金：200万",
    "_": [
        "2、重大疾病医疗保险金：400万",
        "3、恶性肿瘤-质子重离子医疗保险金：400万",
        "4、特定药品费用医疗保险金：200万，100%赔付，社保罚则60%",
        "5、外购药及外购医疗器械费用医疗保险金：100万，100%赔付",
        "6、重大疾病医疗保险金：1万，保证续保期间给付1次",
        "7、特需医疗保险金：100万，30%赔付，拓展特需部",
        "可选1：附加重大疾病特需医疗保险：400万，100%赔付，二级及二级以上公立医院特需部，国际部，VIP部；等待期90天",
        "可选2：个人门急诊C款医疗保险：",
        "门急诊意外医疗：2万，70%赔付，社保罚则42%",
        "门急诊疾病医疗：1万，50%赔付，社保罚则30%",
        "免赔额：门急诊意外医疗100元/门急诊疾病医疗100元",
        "互联网在线门诊费用：2000元，100%赔付",
        "互联网药品费用医疗保险：2万，80%赔付",
        "等待期30天，二级及二级以上公立医院普通部和认可的互联网医院",
        "可选3：个人重大疾病保险：10万/20万，给付，等待期90天"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "70周岁"
}
$$, 10000, $$
{
    "amount": 234,
    "unit": "元"
}
$$, $$
{
    "text": "年200万/保证续保期间总限额800万"
}
$$, '1年', '1-4类', '20年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级以上（含二级）公立医院普通部及指定质子重离子医疗机构/特需医疗保险金-指定医疗机构的特需医疗部、国际部、VIP部"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '90天', $$
{
    "text": "保证续保20年"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, $$
{
    "text": "支持家庭单，2人起，最多6人，主险95折，附加可选无折扣"
}
$$, $$
{
    "text": "世界五百强\n中国保险老三家"
}
$$, NULL, NULL, NULL, 81.8);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('平安健康', '平安e生保互联网长期医疗险（20年保证续保）', '长期百万医疗险', $$
{
    "一般住院医疗保险金": "200万",
    "重大疾病住院医疗保险金": "200万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、质子重离子医疗保险：100万/100%赔付/床位费1500元",
    "_": [
        "可选1、特定药品费用医疗保险：200万",
        "可选2、特定疾病特需医疗：400万"
    ]
}
$$, NULL, NULL, $$
{
    "text": "https://kdocs.cn/l/csgQYkmBHUj0"
}
$$, $$
{
    "min": "28天",
    "max": "55周岁"
}
$$, 10000, $$
{
    "amount": 468,
    "unit": "元"
}
$$, $$
{
    "text": "年200万/保证续保期间总限额800万"
}
$$, '1年', '除特殊职业', '20年（费率可调）', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上公立医院普通部/上海质子重离子医院"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '90天', $$
{
    "text": "保证续保20年"
}
$$, NULL, $$
{
    "plan": "性别"
}
$$, $$
{
    "text": "支持家庭单，3人起，95折，附加可选无折扣"
}
$$, $$
{
    "text": "国内保险老三家\n世界五百强前五十"
}
$$, NULL, NULL, NULL, 79.8);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('平安健康（明亚专属）', '平安健康安欣保长期医疗（20年保证续保）', '长期百万医疗险', $$
{
    "一般住院医疗保险金": "200万",
    "重大疾病住院医疗保险金": "400万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、55种特定疾病医疗保险金：200万（免赔额/报销比例/范围同主险）",
    "_": [
        "2、120种重大疾病医疗保险金：400万（含质子重离子，免赔额/报销比例/范围同主险，质子重离子无社保罚则，赔付比例100%）",
        "3、120种重大疾病津贴：1万，给付，保证续保期间给付1次",
        "可选1、157种特定药品费用医疗保险金：200万，0免赔",
        "可选2、重症监护住院津贴：800元/天，0免赔，100%赔付，不限重疾，每次限30天，每年限90天，非保证续保，二级及二级以上公立医院"
    ],
    "保障责任": "产品责任图解"
}
$$, NULL, NULL, NULL, $$
{
    "min": "28天",
    "max": "70周岁"
}
$$, 10000, $$
{
    "amount": 226,
    "unit": "元"
}
$$, $$
{
    "text": "年400万/保证续保期间总限额800万"
}
$$, '1年', '除特殊职业', '20年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上公立医院普通部/上海质子重离子医院"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '90天', $$
{
    "text": "1、保证续保20年\n2、前一年无理赔，下一年免赔额递减1千，最低减至免赔额5千\n3、家庭共享10000元免赔额"
}
$$, NULL, $$
{
    "plan": "性别"
}
$$, $$
{
    "familyDiscounts": [
        {
            "members": 2,
            "discount": 9.5
        },
        {
            "members": 3,
            "discount": 0.9
        }
    ]
}
$$, $$
{
    "text": "国内保险老三家\n世界五百强前五十"
}
$$, NULL, NULL, NULL, 82.2);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('平安健康（明亚专属）', '平安健康安欣保2.0长期医疗（20年保证续保）', '长期百万医疗险', $$
{
    "一般住院医疗保险金": "200万",
    "重大疾病住院医疗保险金": "400万（120种重大疾病）",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、55种特定疾病医疗保险金：200万（免赔额/报销比例/范围同主险）",
    "_": [
        "2、120种重大疾病医疗保险金：400万（含质子重离子，免赔额/报销比例/范围同主险，质子重离子无社保罚则，赔付比例100%）",
        "3、重大疾病关爱保险金：1万，给付，保证续保期间给付1次",
        "4、9种指定疾病康复医疗保险金：2万，次限额300元，日限额500元",
        "5、（计划二适用）188种院外恶性肿瘤特定药品费用医疗保险金：200万",
        "6、（计划二适用）院外恶性肿瘤特定药品基因检测费用保险金：2万，不限检测机构",
        "可选1、120种院外重大疾病药品费用医疗保险金：药品不限清单；2万，0免赔，100%赔付；二级及二级以上公立医院和指定拓展医疗；等待期90天；保障期1年，不保证续保",
        "可选2：重症监护病房住院津贴：800元/天，每次住院限30天，每年限90天；二级及二级以上公立医院；等待期30天；保障期1年，不保证续保",
        "可选3：少儿门急诊医疗保险金：5000元；",
        "意外门急诊年限额5000元/指定门急诊年限额5000元，日限额500元，20种指定疾病",
        "意外门诊0免赔/指定疾病门急诊次免赔100元",
        "赔付比例：医保或公费医疗报销＞0,80%赔付，=0则50%赔付；医保外报销比例30%；二级及二级以上公立医院普通部"
    ],
    "可选4": "6种特定疾病特需医疗：400万，100%赔付",
    "可选5": "120种重大疾病保险金：10万，1年期，给付1次，不保证续保",
    "保障责任": "产品责任图解"
}
$$, NULL, NULL, NULL, $$
{
    "min": "28天",
    "max": "70周岁"
}
$$, 10000, $$
{
    "amount": 221,
    "unit": "元"
}
$$, $$
{
    "text": "年400万/保证续保期间总限额800万"
}
$$, '1年', '除特殊职业', '20年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级以上（含二级）公立医院普通部，上海质子重离子医院、河北一洲肿瘤医院、淄博万杰肿瘤医院、甘肃省武威肿瘤医院、上海交通大学医学院附属瑞金医院肿瘤质子中心，指定疾病康复医疗保险金限指定康复医疗机构"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%）\n质子重离子、院外恶性肿瘤特定药基因检测费用不适用"
}
$$, '90天（重症监护住院津贴和少儿门急诊30天，意外无等待期）', $$
{
    "text": "1、保证续保20年\n2、前一年无理赔，下一年免赔额递减1千，最低减至免赔额5千\n3、家庭共享10000元免赔额"
}
$$, NULL, $$
{
    "plan": "性别"
}
$$, $$
{
    "familyDiscounts": [
        {
            "members": 2,
            "discount": 9.5
        },
        {
            "members": 3,
            "discount": 0.9
        }
    ]
}
$$, $$
{
    "text": "国内保险老三家\n世界五百强前五十"
}
$$, NULL, NULL, NULL, 82.6);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('安盛天平', '卓越馨选医疗保险（2024版）普通计划C', '中端医疗险', $$
{
    "一般住院医疗保险金": "200万",
    "重大疾病住院医疗保险金": "300万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、重大疾病住院津贴：150元/天，年累计60天为限",
    "_": [
        "2、耐用医疗设备：限额2万元",
        "3、临时关怀费用：限30日",
        "4、住院外购药品费：60%赔付，一般疾病住院限额5千元，重大疾病住院限额5万元",
        "可选1、重大疾病保险金；1万（给付）",
        "可选2、恶性肿瘤院外特定药品费用保险（77种）：150万/100%赔付/等待期30天",
        "可选3、院外特定药品费用保险金（105种+3种car-t）：150万，100%赔付，等待期30天",
        "可选3、海南博鳌乐城特定药品费用保险金；20种，150万/100%赔付",
        "可选4、特定医疗器械费用保险金；100万/100%赔付",
        "可选5、门急诊医疗保险金：计划一：1万保额/计划二1.5万保额，年免赔额0/200/500/1300"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "59周岁"
}
$$, 0, $$
{
    "amount": 911,
    "unit": "元"
}
$$, $$
{
    "amount": 2000000,
    "unit": "元"
}
$$, '1年', '1-4类职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及以上公立医院普通部及本公司指定的民营医疗机构"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '30天', $$
{
    "text": "癌症特药77种和院外特药105种只可选择其一；\n\n选择癌症特药77种不可附加海南博鳌乐城特药和特定器械责任/选择院外特药105种，必须附加海南博鳌乐城特药责任，可附加特定医疗器械；\n\n未成年人仅可单独投保所有计划，但不可附加门诊"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, $$
{
    "familyDiscounts": [
        {
            "members": 2,
            "discount": 9.5
        },
        {
            "members": 3,
            "discount": 0.9
        }
    ]
}
$$, $$
{
    "text": "安盛全资控股子公司\n全球顶尖健康险供应商\n2019年位列全球五百强第46位\n安盛（全球最大保险集团）\n全球顶尖健康险供应商\n全球统一续保政策\n风险评级：BB（2023年1季度）"
}
$$, NULL, NULL, NULL, 78.6);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('复星联合健康', '乐健中端医疗保险（2023版）计划四', '中端医疗险', $$
{
    "一般住院医疗保险金": "150万",
    "重大疾病住院医疗保险金": "150万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、重大疾病住院津贴：150元/天，年累计60天为限",
    "_": [
        "2、床位费/膳食费/护理费：日限额400元（不承担单人病房，特需病房，VIP病房等）",
        "3、耐用医疗设备：限15000元",
        "4、临终关怀费用：累计限30日",
        "5、精神和心理障碍治疗费：累计限额5万元",
        "6、手术机器人费用含耗材：无单项限额",
        "7、癌症基因检测费用：限额2000元",
        "可选1、癌症海外医疗：",
        "①计划一：新加坡/日本：80%赔付，100万",
        "②计划二：美国：80%赔付，100万",
        "可选2、门诊责任：",
        "①计划一：1万保额",
        "②计划二：1.5万保额",
        "年免赔额：0/200元/500元/1300元"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "64周岁"
}
$$, 0, $$
{
    "amount": 780,
    "unit": "元"
}
$$, $$
{
    "amount": 1500000,
    "unit": "元"
}
$$, '1年', '1-4类', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "本机构指定或认可的医疗机构的普通部"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "text": "1"
}
$$, '30天', NULL, NULL, $$
{
    "plan": "有无社保"
}
$$, NULL, $$
{
    "text": "2017年成立\n健康险保险公司之一\n上海复星集团旗下"
}
$$, NULL, NULL, NULL, 77.6);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('君龙人寿', '臻爱无忧（6年保证续保）普通计划二', '中端医疗险', $$
{
    "一般住院医疗保险金": "200万",
    "重大疾病住院医疗保险金": "200万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前7天后30天",
    "其他责任": "1、重疾住院津贴：150元/天，限60天",
    "_": [
        "可选1、重大疾病保险金：1万/2万（给付）",
        "可选2、恶性肿瘤—重度特定药品保险金：150万124种，100%赔付",
        "可选3、门急诊医疗"
    ],
    "保障责任": "产品责任图解"
}
$$, NULL, NULL, NULL, $$
{
    "min": "28天",
    "max": "65周岁"
}
$$, 0, $$
{
    "amount": 751,
    "unit": "元"
}
$$, $$
{
    "amount": 2000000,
    "unit": "元"
}
$$, '1年', '1-4类', '6年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上公立医院普通部/认可民营机构118家"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '30天', $$
{
    "text": "6年保证续保；\n\n3年无理赔，可拓展既往症；\n\n涵盖指定私立医院；\n\n未成年人可单独投保"
}
$$, NULL, $$
{
    "plan": "必选"
}
$$, NULL, $$
{
    "text": "2022最具竞争力保险产品\n股东厦门国资建发集团\n股东台湾首家保司台湾人寿\n风险评级：BB（2023年1季度）"
}
$$, NULL, NULL, NULL, 78.2);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('众安保险', '尊享e生中高端医疗保险2024版普通计划二', '中端医疗险', $$
{
    "一般住院医疗保险金": "300万",
    "重大疾病住院医疗保险金": "300万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、床位费：1500元/天，不高于双人病房标准",
    "_": [
        "2、陪床费：600元/天，不高于双人病房标准",
        "3、医疗器械费：无单项限额",
        "重建手术费：每次手术限10万",
        "出院后特别关怀费用：无单项限额",
        "精神和心理障碍治疗费：限额10万",
        "耐用医疗设备：限额10万",
        "4、恶性肿瘤先进疗法医疗保险金：300万（恶性肿瘤质子重离子/硼中子俘获疗法/光免疫疗法），床位费限1500元/天",
        "5、外购药品及外购医疗器械费用医疗保险金：100万，社保目录内药品，社保报销后扣除年免赔额100%报销，社保罚则60%/社保目录外药品，扣除年免赔额后100%报销",
        "6、特定药品费用医疗保险金：300万，135种癌症特药，30种罕见病特药，15种海南博鳌特药，癌症基因检测费用2万",
        "7、重大疾病异地转诊公共交通费用及住宿费用保险金：1万",
        "可选1：重疾关爱金：5万/10万/15万/20万",
        "可选2：门急诊医疗保险金：2万，0免赔，100%赔付，有社保但未以社保身份结算60%赔付/互联网药品费用：1200元，100%赔付，次限额100元，次免赔0元"
    ],
    "保障责任": "=DISPIMG(\"ID_ED87CC9335614B258A905A1A7016D684\",1)"
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "65周岁"
}
$$, 0, $$
{
    "amount": 1953,
    "unit": "元"
}
$$, $$
{
    "amount": 3000000,
    "unit": "元"
}
$$, '1年', '除特殊职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上的公立医院普通部/质子重离子医院"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '30天', $$
{
    "text": "重疾关爱金区分性别费率\n\n支持未成年人单独投保所有计划\n\n智能核保宽松，结节1-3级未手术可尝试，除外承保；结节术后良性可尝试标体承保；高血压二级以内，原发性可尝试除外承保"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, $$
{
    "familyDiscounts": [
        {
            "members": 2,
            "discount": 9.5
        },
        {
            "members": 3,
            "discount": 0.9
        },
        {
            "members": 4,
            "discount": 8.5
        }
    ]
}
$$, $$
{
    "text": "全国首家互联网保险公司\n由蚂蚁金服、中国平安和腾讯于2013年联合发起设立"
}
$$, NULL, $$
{
    "scope": "https://kdocs.cn/l/cbBCymQbbYem"
}
$$, NULL, 77.0);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('美亚财险', '“臻馨优选”个人中端医疗保障计划计划二', '中端医疗险', $$
{
    "一般住院医疗保险金": "300万",
    "重大疾病住院医疗保险金": "300万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前7天后30天",
    "其他责任": "1、耐用医疗设备：限额2万",
    "_": [
        "2、临终关怀费用：累计赔偿天数30天",
        "3、重疾住院津贴：300元/天，限60天",
        "4、重症疾病异地治疗交通住宿保障：1万",
        "5、特定疾病药械费用补偿保障：300万，院外特定药品/临床急需进口药品/特定医疗器械费用",
        "可选1：意外门急诊医疗保障：3万",
        "可选2：门急诊医疗保障：1万，0免赔",
        "两个可选仅限选择一项"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "66周岁"
}
$$, 0, $$
{
    "amount": 859,
    "unit": "元"
}
$$, $$
{
    "amount": 3000000,
    "unit": "元"
}
$$, '1年', '除特殊职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上的公立医院普通部/质子重离子医院"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '30天', NULL, NULL, $$
{
    "plan": "有无社保"
}
$$, NULL, $$
{
    "text": "中国最大外资财产保险公司\n最早外资财产保险机构牌照\n风险评级：AA（2023年1季度）"
}
$$, NULL, NULL, NULL, 81.0);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('京东安联', '臻爱无限医疗保险（互联网pro版）京选计划', '中端医疗险', $$
{
    "一般住院医疗保险金": "300万",
    "重大疾病住院医疗保险金": "300万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前7天后30天",
    "其他责任": "1、一般医疗保险金：300万，扩展特定既往症限：甲状腺结节、乳腺结节、肺结节、胆结石、胆囊息肉，与一般医疗保险金共享保额和1万免赔额，赔付比例最高不超过50%，具体以保单约定为准。",
    "_": [
        "2、恶性肿瘤—重度院外特定药品费用保险金：与重大疾病医疗保险金共享300万额度（Car-t用药仅限赔付一次）",
        "3、恶性肿瘤—重度特定药品基因检测费用保险金：1万",
        "4、指定疾病急需药品费用保险金：3万",
        "5、临床急需进口特定药品费用保险金：与重大疾病医疗保险金共享300万额度",
        "6、特定医疗器械费用保险金：与重大疾病医疗保险金共享300万额度",
        "7、重症监护病房住院津贴医疗保险金：500元/天，每次免赔5天，单次最高给付日数10天，年累计给付日数30天",
        "8、重疾异地转诊公共交通费用保险金：1万",
        "9、100种特定疾病住院津贴：100元/天，每次事故免赔3天，单次最高给付日数30天，年累计给付日数90天",
        "10、恶性肿瘤—重度先进疗法医疗保险金：与重大疾病医疗保险金共享300万额度",
        "可选1：意外身故伤残：30万",
        "可选2：重症/中症/轻症给付：5万/2万/1万",
        "可选3：一万免赔额补偿（0免赔）：100%赔付（不承担女性生殖系统疾病、结节、息肉、囊肿、增生、结石赔付）"
    ],
    "保障责任": "=DISPIMG(\"ID_A335D2C2921444E2974AE52C663298EE\",1)"
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "65周岁"
}
$$, 10000, $$
{
    "amount": 294,
    "unit": "元"
}
$$, $$
{
    "amount": 3000000,
    "unit": "元"
}
$$, '1年', '1-4类', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上的公立医院普通部/质子重离子医院"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '疾病等待期为30天，轻/中/重疾病给付等待期90天。意外医疗及连续投保没有等待期。在线问诊及药品服务中，在线问诊无等待期，药品服务等待期为30天。', $$
{
    "text": "=DISPIMG(\"ID_DD341D9829FE42B686649EB7D910AF26\",1)"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, $$
{
    "familyDiscounts": [
        {
            "members": 2,
            "discount": 9.5
        },
        {
            "members": 3,
            "discount": 0.9
        }
    ]
}
$$, NULL, NULL, NULL, NULL, 82.4);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('京东安联', '臻爱无限医疗保险（互联网pro版）臻选计划', '中端医疗险', $$
{
    "一般住院医疗保险金": "600万",
    "重大疾病住院医疗保险金": "600万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前7天后30天",
    "其他责任": "1、一般医疗保险金：600万，扩展特定既往症限：甲状腺结节、乳腺结节、肺结节、胆结石、胆囊息肉，与一般医疗保险金共享保额和1万免赔额，赔付比例最高不超过50%，具体以保单约定为准。",
    "_": [
        "2、恶性肿瘤—重度院外特定药品费用保险金：与重大疾病医疗保险金共享但限额300万额度（Car-t用药仅限赔付一次）",
        "3、恶性肿瘤—重度特定药品基因检测费用保险金：1万",
        "4、指定疾病急需药品费用保险金：3万",
        "5、临床急需进口特定药品费用保险金：与重大疾病医疗保险金共享600万额度",
        "6、特定医疗器械费用保险金：与重大疾病医疗保险金共享但限额300万额度",
        "7、重症监护病房住院津贴医疗保险金：500元/天，每次免赔5天，单次最高给付日数10天，年累计给付日数30天",
        "8、重疾异地转诊公共交通费用保险金：1万",
        "9、100种特定疾病住院津贴：100元/天，每次事故免赔3天，单次最高给付日数30天，年累计给付日数90天",
        "10、恶性肿瘤—重度先进疗法医疗保险金：与重大疾病医疗保险金共享但限额300万额度",
        "11、重大疾病特需费用保险金：与主线共享600万",
        "12、在线问诊及药品服务：2万，80%赔付，不限次",
        "可选1：意外身故伤残：60万",
        "可选2：重症/中症/轻症给付：5万/2万/1万",
        "可选3：一万免赔额补偿（0免赔）：100%赔付（不承担女性生殖系统疾病、结节、息肉、囊肿、增生、结石赔付）",
        "可选4：一万免赔额补偿+特需医疗：拓展0免赔+拓展至二级及以上公立医院特需部"
    ],
    "保障责任": "=DISPIMG(\"ID_A335D2C2921444E2974AE52C663298EE\",1)"
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "65周岁"
}
$$, 0, $$
{
    "amount": 752,
    "unit": "元"
}
$$, $$
{
    "amount": 6000000,
    "unit": "元"
}
$$, '1年', '1-4类', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上的公立医院普通部（可拓展至公立医院特需部）/质子重离子医院"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '疾病等待期为30天，轻/中/重疾病给付等待期90天。意外医疗及连续投保没有等待期。在线问诊及药品服务中，在线问诊无等待期，药品服务等待期为30天。', $$
{
    "text": "=DISPIMG(\"ID_DD341D9829FE42B686649EB7D910AF26\",1)"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, $$
{
    "familyDiscounts": [
        {
            "members": 2,
            "discount": 9.5
        },
        {
            "members": 3,
            "discount": 0.9
        }
    ]
}
$$, NULL, NULL, NULL, NULL, 82.4);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('平安财险', '平安人生中高端医疗险（互联网版）计划A', '中端医疗险', $$
{
    "一般住院医疗保险金": "400万",
    "重大疾病住院医疗保险金": "400万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、床位费：2500元/天，限标准单人病房，重大疾病拓展公立医院特需/国际部",
    "_": [
        "2、陪床费：800元/晚",
        "3、住院护工费：300元/天，每年最长15天",
        "4、医疗运送和异地就医：救护车费用全额，重大疾病异地就医交通，住宿费1万",
        "5、恶性肿瘤—重度临床急需进口药品费用：全额（大湾区、天津自贸区、海南博鳌肿瘤特药）",
        "6、未成年人先天性疾病医疗：1万（首次投保尚未发现先天性疾病）",
        "可选1：门急诊医疗：2.5万，前6次100%赔付，后续60%",
        "可选2：重大疾病境外医疗：800万，100%赔付，全球范围",
        "可选3：大湾区门诊医疗费用责任：10万/50万，大湾区指定医疗机构（仅限内地，不含港澳）",
        "限额10万：前10次100%赔付，10次后自费",
        "限额50万：不限次，100%赔付"
    ],
    "保障责任": "=DISPIMG(\"ID_FFB9163BDEA94A98B2C04C75D5B5D96B\",1)"
}
$$, NULL, NULL, $$
{
    "text": "https://kdocs.cn/l/copgGZcM4lqu"
}
$$, $$
{
    "min": "30天",
    "max": "70周岁"
}
$$, 0, $$
{
    "amount": 885,
    "unit": "元"
}
$$, $$
{
    "amount": 4000000,
    "unit": "元"
}
$$, '1年', '除高危职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及以上公立医院普通部、指定医保定点非公医院；重大疾病扩展：公立医院特需部/国际部"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '住院（必选）30天，门急诊医疗（可选）30天，重疾境外医疗（可选）90天，大湾区门诊医疗（可选）无等待期', $$
{
    "text": "增值服务：\n1.住院、门诊就医绿通\n2.MDT多学科会诊/海外专家二诊\n3.全程陪同免现金住院服务\n4.恶性肿瘤基因检测\n5.特定药品院外药房直付\n6.临床急需进口院外特药\n7.术后居家康护服务\n8.临床前沿治疗直通车\n9.肿瘤患者家族早癌筛查\n10.专属病案管家\n11.互联网超低购药折扣\n12.重大疾病海外就医服务（若选购海外医疗）\n13.肿瘤类器官（PDO）药敏试验服务\n14.肿瘤营养管理\n\n7周岁以上长期开放单独投保，可优于主被保险人"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, $$
{
    "text": "支持家庭单，2人以上95折"
}
$$, $$
{
    "text": "国内保险老三家\n世界五百强前五十"
}
$$, NULL, NULL, NULL, 82.0);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('泰康在线', '臻安心·高疗无忧医疗险轻享版', '中端医疗险', $$
{
    "一般住院医疗保险金": "600万",
    "重大疾病住院医疗保险金": "600万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前7天后30天",
    "其他责任": "1、床位费：限双人病房，日限额1500元",
    "_": [
        "2、加床费：涵盖",
        "3、院内药品费：涵盖",
        "4、外购药品费：涵盖，无清单限制",
        "5、特定疾病门急诊及一般疾病特殊门急诊医疗保险金：400万",
        "①普通部赔付比例100%",
        "②药品费和医疗器械费（含约定的《特种药品清单》和《特定医疗起诶清单》中适用的药品和器械费用）：涵盖",
        "③确诊前门急诊费用：前30天"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天（含）",
    "max": "70周岁（含）"
}
$$, 0, $$
{
    "amount": 916,
    "unit": "元"
}
$$, $$
{
    "text": "年最高800万"
}
$$, '1年', NULL, '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及以上公立医院普通部、指定或认可的医院、指定或认可的医疗机构"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 0.8,
    "note": "以社保身份参保未使用社保结算48%），自担金额累计到5千元时，赔付比例100%（未经社保60%）\n特定疾病住院医疗普通部100%赔付，未经社保60%赔付"
}
$$, '30天', $$
{
    "text": "1、自担金额=责任范围内的医疗费用。其他途径补偿(包括但不限于社保报销、大病保险、医疗救助和商保赔付(含本产品赔付)等任何第三方获得的医疗费用补偿)\n2、院外恶性肿瘤特药扩展符合临床医学的先进治疗用药标准，且可由医生多点执业的私立医疗机构开具特药处方"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, $$
{
    "familyDiscounts": [
        {
            "members": 2,
            "discount": 9.5
        },
        {
            "members": 3,
            "discount": 0.9
        }
    ]
}
$$, NULL, NULL, NULL, 'https://kdocs.cn/l/cndLGUkkHkQc', 82.6);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('泰康在线', '臻安心·高疗无忧医疗险轻享Pro版', '中端医疗险', $$
{
    "一般住院医疗保险金": "600万",
    "重大疾病住院医疗保险金": "600万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前7天后30天",
    "其他责任": "1、床位费：限双人病房，日限额1500元",
    "_": [
        "2、加床费：涵盖",
        "3、院内药品费：涵盖",
        "4、外购药品费：涵盖，无清单限制",
        "5、特定疾病门急诊及一般疾病特殊门急诊医疗保险金：400万",
        "①普通部赔付比例100%/特需部，国际部，普通部医保身份购买但自费身份就医，经过预授权后100%赔付",
        "②药品费和医疗器械费（含约定的《特种药品清单》和《特定医疗起诶清单》中适用的药品和器械费用）：涵盖",
        "③确诊前门急诊费用：前30天",
        "6、一般疾病门急诊医疗保险金：2万",
        "①赔付比例：30%，当自担金额到5千元时，赔付比例100%",
        "②免赔额：0元"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天（含）",
    "max": "70周岁（含）"
}
$$, 0, $$
{
    "amount": 1966,
    "unit": "元"
}
$$, $$
{
    "text": "年最高800万（一般疾病门急诊不超过3万）"
}
$$, '1年', NULL, '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及以上公立医院普通部、我们扩展承保的私立医院的普通部、指定或认可的医院、指定或认可的医疗机构；特定疾病拓展以上医院的特需部，国际部"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 0.8,
    "note": "以社保身份参保未使用社保结算48%），自担金额累计到5千元时，赔付比例100%（未经社保60%）\n特定疾病住院医疗普通部100%赔付，未经社保60%赔付/特需部，国际部以医保身份购买但自费身份就医经预授权后100%赔付"
}
$$, '30天', $$
{
    "text": "1、自担金额=责任范围内的医疗费用。其他途径补偿(包括但不限于社保报销、大病保险、医疗救助和商保赔付(含本产品赔付)等任何第三方获得的医疗费用补偿)\n2、院外恶性肿瘤特药扩展符合临床医学的先进治疗用药标准，且可由医生多点执业的私立医疗机构开具特药处方\n3、Pro版含门诊责任"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, $$
{
    "familyDiscounts": [
        {
            "members": 2,
            "discount": 9.5
        },
        {
            "members": 3,
            "discount": 0.9
        }
    ]
}
$$, NULL, NULL, NULL, NULL, 81.6);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('人保健康', '人人保中端医疗险计划一（健康人群）', '中端医疗险', $$
{
    "一般住院医疗保险金": "400万",
    "重大疾病住院医疗保险金": "400万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、床位费：800元/天，先标准单人病房",
    "_": [
        "2、陪床费：限800元/天",
        "3、院内基因检测费用：限额3万元",
        "4、首次投保时尚未发现的先天性疾病（仅适用于0-18岁的被保险人）：1万",
        "5、重大疾病异地转诊交通、住宿费：限额1万",
        "计划一经过智能核保可附加重大疾病拓展特需加油包：",
        "拓展公立医院的特需部，国际部，VIP部",
        "28种重大疾病相关治疗费用（床位费，膳食费和护理费用之和最高2500元/天）"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天（含）",
    "max": "70周岁（含）"
}
$$, 0, $$
{
    "amount": 785,
    "unit": "元"
}
$$, $$
{
    "amount": 4000000,
    "unit": "元"
}
$$, '1年', '除特殊职业', '1年（保证续保5年）', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级以上（含二级）公立医院普通部、保险公司指定的民营医院普通部、上海市质子重离子医院"
}
$$, $$
{
    "scope": "社保内外医疗费用，含院外购药"
}
$$, $$
{
    "ratio": 1.0,
    "note": "以社保身份参保未经社保报销60%"
}
$$, '30天，意外无等待期', $$
{
    "text": "门诊就医绿通；\n住院就医绿通；\nMDT多学科会诊/第二诊疗意见；\n免现金住院服务；\n门诊陪诊服务；\n恶性肿瘤靶向药物基因检测（院外）；\n特定药品院外药房直付；\n术后上门护理；\n临床前沿治疗直通车；\n专属病案管家；\n互联网超低购药折扣；\n慈善赠药"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, NULL, NULL, NULL, NULL, 'https://kdocs.cn/l/ccfV0g0NWmWG', 83.0);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('人保健康', '人人保中端医疗险计划二（次标体人群）', '中端医疗险', $$
{
    "一般住院医疗保险金": "400万",
    "重大疾病住院医疗保险金": "400万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、床位费：800元/天，先标准单人病房",
    "_": [
        "2、陪床费：限800元/天",
        "3、院内基因检测费用：限额3万元",
        "4、重大疾病异地转诊交通、住宿费：限额1万"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天（含）",
    "max": "70周岁（含）"
}
$$, 10000, $$
{
    "amount": 692,
    "unit": "元"
}
$$, $$
{
    "amount": 4000000,
    "unit": "元"
}
$$, '1年', '除特殊职业', '1年（保证续保5年）', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级以上（含二级）公立医院普通部、保险公司指定的民营医院普通部、上海市质子重离子医院"
}
$$, $$
{
    "scope": "社保内外医疗费用"
}
$$, $$
{
    "ratio": 1.0,
    "note": "以社保身份参保未经社保报销60%"
}
$$, '30天，意外无等待期', $$
{
    "text": "门诊就医绿通；\n住院就医绿通；\nMDT多学科会诊/第二诊疗意见；\n免现金住院服务；\n门诊陪诊服务；\n恶性肿瘤靶向药物基因检测（院外）；\n特定药品院外药房直付；\n术后上门护理；\n临床前沿治疗直通车；\n专属病案管家；\n互联网超低购药折扣；\n慈善赠药"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, NULL, NULL, NULL, NULL, 'https://kdocs.cn/l/ccfV0g0NWmWG', 82.6);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('人保健康', '人人保中端医疗险计划三（重大既往症人群）', '中端医疗险', $$
{
    "一般住院医疗保险金": "5万",
    "重大疾病住院医疗保险金": "5万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、床位费：800元/天，先标准单人病房",
    "_": [
        "2、陪床费：限800元/天",
        "3、院内基因检测费用：限额3万元",
        "4、重大疾病异地转诊交通、住宿费：限额1万"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天（含）",
    "max": "70周岁（含）"
}
$$, 20000, $$
{
    "amount": 4245,
    "unit": "元"
}
$$, $$
{
    "amount": 50000,
    "unit": "元"
}
$$, '1年', '除特殊职业', '1年（保证续保5年）', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级以上（含二级）公立医院普通部、保险公司指定的民营医院普通部、上海市质子重离子医院"
}
$$, $$
{
    "scope": "仅保障社保目录内医疗费用"
}
$$, $$
{
    "ratio": 0.4,
    "note": "有社保身份加入但未用社保赔付，赔付比例是24%"
}
$$, '30天，意外无等待期', $$
{
    "text": "门诊就医绿通；\n住院就医绿通；\nMDT多学科会诊/第二诊疗意见；\n免现金住院服务；\n门诊陪诊服务；\n恶性肿瘤靶向药物基因检测（院外）；\n特定药品院外药房直付；\n术后上门护理；\n临床前沿治疗直通车；\n专属病案管家；\n互联网超低购药折扣；\n慈善赠药"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, NULL, NULL, NULL, NULL, 'https://kdocs.cn/l/ccfV0g0NWmWG', 77.6);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('平安财险', '平安人生中高端医疗险（互联网版）计划B', '中端医疗险', $$
{
    "一般住院医疗保险金": "600万",
    "重大疾病住院医疗保险金": "600万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、床位费：2500元/天，限标准单人病房，重大疾病拓展公立医院特需/国际部",
    "_": [
        "2、陪床费：800元/晚",
        "3、住院护工费：300元/天，每年最长15天",
        "4、医疗运送和异地就医：救护车费用全额，重大疾病异地就医交通，住宿费1万",
        "5、恶性肿瘤—重度临床急需进口药品费用：全额（大湾区、天津自贸区、海南博鳌肿瘤特药）",
        "6、未成年人先天性疾病医疗：2万（首次投保尚未发现先天性疾病）",
        "可选1：门急诊医疗：5万",
        "可选2：重大疾病境外医疗：800万，100%赔付，全球范围",
        "可选3：大湾区门诊医疗费用责任：10万/50万，大湾区指定医疗机构（仅限内地，不含港澳）",
        "限额10万：前10次100%赔付，10次后自费",
        "限额50万：不限次，100%赔付"
    ],
    "保障责任": "=DISPIMG(\"ID_FFB9163BDEA94A98B2C04C75D5B5D96B\",1)"
}
$$, NULL, NULL, $$
{
    "text": "https://kdocs.cn/l/copgGZcM4lqu"
}
$$, $$
{
    "min": "30天",
    "max": "70周岁"
}
$$, 0, $$
{
    "amount": 2318,
    "unit": "元"
}
$$, $$
{
    "amount": 6000000,
    "unit": "元"
}
$$, '1年', '除高危职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及以上公立医院普通部/特需部/国际部、指定医保定点非公医院"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '住院（必选）30天，门急诊医疗（可选）30天，重疾境外医疗（可选）90天，大湾区门诊医疗（可选）无等待期', $$
{
    "text": "增值服务：\n1.住院、门诊就医绿通\n2.MDT多学科会诊/海外专家二诊\n3.全程陪同免现金住院服务\n4.恶性肿瘤基因检测\n5.特定药品院外药房直付\n6.临床急需进口院外特药\n7.术后居家康护服务\n8.临床前沿治疗直通车\n9.肿瘤患者家族早癌筛查\n10.专属病案管家\n11.互联网超低购药折扣\n12.重大疾病海外就医服务（若选购海外医疗）\n13.肿瘤类器官（PDO）药敏试验服务\n14.肿瘤营养管理\n\n7周岁以上长期开放单独投保，可优于主被保险人"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, $$
{
    "text": "支持家庭单，2人以上95折"
}
$$, $$
{
    "text": "国内保险老三家\n世界五百强前五十"
}
$$, NULL, NULL, NULL, 81.0);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('平安健康', '平安健康明爱安馨长期医疗险', '中端医疗险', $$
{
    "一般住院医疗保险金": "400万",
    "重大疾病住院医疗保险金": "400万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、耐用医疗设备：10万",
    "_": [
        "2、精神和心理障碍治疗：10万",
        "3、外购药品及外购医疗器械费用保险金：普通计划1万/特需计划100万",
        "4、出院后特别关怀（康复治疗费用/临终关怀费）：400万",
        "5、质子重离子：400万，床位费1500元/天，0免赔，100%赔付",
        "6、123种院外恶性肿瘤特定药品费用：400万，0免赔，100%赔付",
        "7、院外恶性肿瘤特定药品基因检测：2万，0免赔，100%赔付",
        "8、重大疾病就医补贴保险金：特需计划2万/普通计划无",
        "可选：附加特需，门诊，可选特需部免赔额"
    ],
    "保障责任": "产品责任图解"
}
$$, NULL, NULL, NULL, $$
{
    "min": "28天",
    "max": "70周岁"
}
$$, 0, $$
{
    "amount": 760,
    "unit": "元"
}
$$, $$
{
    "text": "400万/保证续保期间总800万"
}
$$, '1年', '除特殊职业', '10年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及以上公立医院普通部、特需部、VIP部、国际部及本公司指定拓展医院、河北一洲肿瘤医院/上海质子重离子"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '90天', $$
{
    "text": "1、免赔额：普通部部分，扣除社保后，10000元以下80%-100%赔付，无赔付逐年递增，最高100%，首年80%赔付，10000元以上100%赔付，触发理赔后，10000元以内降为80%赔付并不再边变化\n\n2、普通部计划保证续保10年，附加特需不保证续保\n\n3、未成年人支持单独投保"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, $$
{
    "familyDiscounts": [
        {
            "members": 2,
            "discount": 9.5
        },
        {
            "members": 3,
            "discount": 0.9
        }
    ]
}
$$, $$
{
    "text": "国内保险老三家\n世界五百强前五十"
}
$$, NULL, NULL, NULL, 82.2);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('安盛天平', '卓越馨选医疗保险（2024版）特需计划C', '中端医疗险', $$
{
    "一般住院医疗保险金": "200万",
    "重大疾病住院医疗保险金": "400万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30后30天",
    "其他责任": "1、重大疾病住院津贴：300元/天，年累计60天为限",
    "_": [
        "2、耐用医疗设备：无限额",
        "3、临终关怀：限30日",
        "4、住院外购要药品费：100%赔付，一般疾病住院限额1万元，重大疾病住院限额10万元",
        "可选1、重大疾病保险金；2万（给付）",
        "可选2、恶性肿瘤院外特定药品费用保险（77种）：150万/100%赔付/等待期30天",
        "可选3、院外特定药品费用保险金（105种）：150万/100%赔付/等待期30天",
        "可选3、海南博鳌乐城特定药品费用保险金；150万/100%赔付",
        "可选4、特定医疗器械费用保险金；100万/100%赔付",
        "可选5、门急诊医疗保险金"
    ],
    "保障责任": "产品责任图解"
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "59周岁"
}
$$, 0, $$
{
    "amount": 2029,
    "unit": "元"
}
$$, $$
{
    "amount": 2000000,
    "unit": "元"
}
$$, '1年', '1-4类职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及以上公立医院普通部、特需部、VIP部、国际部及本公司指定的民营医疗机构"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '30天', $$
{
    "text": "癌症特药77种和院外特药105种只可选择其一；\n\n选择癌症特药77种不可附加海南博鳌乐城特药和特定器械责任/选择院外特药105种，必须附加海南博鳌乐城特药责任，可附加特定医疗器械；\n\n未成年人仅可单独投保所有计划，但不可附加门诊"
}
$$, NULL, $$
{
    "plan": "有社保"
}
$$, $$
{
    "text": "支持家庭单，2人起，保障计划和免赔额须一致"
}
$$, $$
{
    "text": "安盛全资控股子公司\n全球顶尖健康险供应商\n2019年位列全球五百强第46位\n安盛（全球最大保险集团）\n全球顶尖健康险供应商\n全球统一续保政策\n风险评级：BB（2023年1季度）"
}
$$, NULL, NULL, NULL, 76.6);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('安盛天平', '卓越环球个人医疗保障智选住院计划', '中端医疗险', $$
{
    "一般住院医疗保险金": "800万",
    "重大疾病住院医疗保险金": "800万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30后30天",
    "其他责任": "1、公立医院普通部住院补偿500元/晚，每次住院限7天，每年限30天",
    "_": [
        "2、质子重离子住院床位限额1500元/晚",
        "3、近亲属陪床费：限90天，800元/晚",
        "4、耐用医疗设备年度限额3000元",
        "5、康复治疗费用：年最高限28天",
        "6、特定互联网医院在线问诊及药品费用：不限次数，药品80%赔付，每年累计限额2000元，每月限2次，每次限额200元",
        "7、24小时紧急救援服务费用补偿金：全额",
        "①紧急医疗护理，电话医疗建议，评估和推荐预约",
        "②紧急医疗运送",
        "③治疗后医疗护送",
        "④遗体或骨灰的送返"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "15天",
    "max": "65周岁"
}
$$, 0, $$
{
    "amount": 2677,
    "unit": "元"
}
$$, $$
{
    "amount": 8000000,
    "unit": "元"
}
$$, '1年', '除特殊职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上公立医院普通部/特需部/VIP部/国际部/认可的拓展承保医院"
}
$$, $$
{
    "scope": "不限社保目录"
}
$$, $$
{
    "text": "1"
}
$$, '30天', $$
{
    "text": "·"
}
$$, NULL, $$
{
    "plan": "非北京地区"
}
$$, $$
{
    "text": "支持家庭单，3人以上95折"
}
$$, $$
{
    "text": "安盛全资控股子公司\n全球顶尖健康险供应商\n2019年位列全球五百强第46位\n安盛（全球最大保险集团）\n全球顶尖健康险供应商\n全球统一续保政策\n风险评级：BB（2023年1季度）"
}
$$, NULL, NULL, NULL, 80.0);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('众安保险', '尊享e生中高端医疗保险2024版特需计划二', '中端医疗险', $$
{
    "一般住院医疗保险金": "300万",
    "重大疾病住院医疗保险金": "300万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、床位费：2500元/天，不高于标准单人病房标准",
    "_": [
        "2、陪床费：1000元/天，不高于标准单人病房标准",
        "3、医疗器械费：无单项限额",
        "重建手术费：每次手术限10万",
        "精神和心理障碍治疗费：限额10万",
        "出院后特别关怀费用：无单项限额",
        "耐用医疗设备：限额10万",
        "4、恶性肿瘤先进疗法医疗保险金：300万（恶性肿瘤质子重离子/硼中子俘获疗法/光免疫疗法），床位费限2500元/天",
        "5、外购药品及外购医疗器械费用医疗保险金：100万，社保目录内药品，社保报销后扣除年免赔额100%报销，社保罚则60%/社保目录外药品，扣除年免赔额后100%报销",
        "6、特定药品费用医疗保险金：300万，135种癌症特药，30种罕见病特药，15种海南博鳌特药，癌症基因检测费用2万",
        "7、重大疾病异地转诊公共交通费用及住宿费用保险金：2万",
        "可选1：重疾关爱金：5万/10万/15万/20万",
        "可选2：门急诊医疗保险金：5万，0免赔，100%赔付，有社保但未以社保身份结算60%赔付/互联网药品费用：1200元，100%赔付，次限额100元，次免赔0元"
    ],
    "保障责任": "=DISPIMG(\"ID_ED87CC9335614B258A905A1A7016D684\",1)"
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "65周岁"
}
$$, 0, $$
{
    "amount": 3022,
    "unit": "元"
}
$$, $$
{
    "amount": 3000000,
    "unit": "元"
}
$$, '1年', '除特殊职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上的公立医院普通部/特需部/国际部/VIP部/质子重离子医院"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '30天', $$
{
    "text": "重疾关爱金区分性别费率\n\n支持未成年人单独投保所有计划"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, $$
{
    "familyDiscounts": [
        {
            "members": 2,
            "discount": 9.5
        }
    ]
}
$$, $$
{
    "text": "全国首家互联网保险公司\n由蚂蚁金服、中国平安和腾讯于2013年联合发起设立"
}
$$, NULL, $$
{
    "scope": "https://kdocs.cn/l/cbBCymQbbYem"
}
$$, NULL, 77.0);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('招商信诺', '智惠人生Ⅱ医疗保险基础计划', '中端医疗险', $$
{
    "一般住院医疗保险金": "150万",
    "重大疾病住院医疗保险金": "150万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、床位费：2000元/天",
    "_": [
        "2、陪床费：800元/天",
        "3、外置修复体：无限额",
        "4、重大疾病住院津贴：350元/天，免赔天数5天，限30天",
        "可选1、牙科门诊：80%赔付，保额2000元",
        "可选2、体检保障：1000元/2000元，年限一次",
        "可选3：院外特定药品费用：300万"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "65周岁"
}
$$, 15000, $$
{
    "amount": 2040,
    "unit": "元"
}
$$, $$
{
    "amount": 1500000,
    "unit": "元"
}
$$, '1年', '除特殊职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "公立医院普通部/特需部/VIP部/国际部"
}
$$, $$
{
    "scope": "不限社保目录"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '30天', $$
{
    "text": "1、0-6周岁儿童不可单独投保\n2、风险等级：低风险95%，中风险100%，高风险105%\n3、医院不限制二级及二级以上，公立医院即可"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, $$
{
    "text": "支持家庭单，2人起95折"
}
$$, $$
{
    "text": "股东招商银行\n股东美国四大健康保司Cigna\n2019最佳健康保障创新公司\n高端医疗第一梯队\n风险评级：BBB（2023年1季度）"
}
$$, NULL, NULL, NULL, 77.0);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('MSH', '欣享人生2023版计划A', '中端医疗险', $$
{
    "一般住院医疗保险金": "100万",
    "重大疾病住院医疗保险金": "100万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前7天后7天",
    "其他责任": "1、床位费：1500元/天",
    "_": [
        "2、陪床费：300元/天",
        "3、护工津贴：200元/天，每年限10天",
        "4、耐用医疗设备：2000元"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "40周岁"
}
$$, 15000, $$
{
    "amount": 678,
    "unit": "元"
}
$$, $$
{
    "amount": 1000000,
    "unit": "元"
}
$$, '1年', '1-2类（3-4类具体情况具体分析）', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上公立医院普通部/特需部/国际部/指定私立医疗机构"
}
$$, $$
{
    "scope": "不限社保目录"
}
$$, $$
{
    "text": "1"
}
$$, '90天', $$
{
    "text": "0-17周岁未成年人不可单独投保；\n\n计划A限0-40周岁投保"
}
$$, NULL, $$
{
    "plan": "18-40周岁保费和0-17周岁连带投保保费"
}
$$, $$
{
    "familyDiscounts": [
        {
            "members": 2,
            "discount": 9.5
        }
    ]
}
$$, $$
{
    "text": "国际知名健康险服务商\n中国高端健康险市场份额第一\n深耕20年直付网络全国最广"
}
$$, NULL, NULL, NULL, 74.3);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('MSH', '欣享人生2023版计划B', '中端医疗险', $$
{
    "一般住院医疗保险金": "600万",
    "重大疾病住院医疗保险金": "600万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、床位费：2000元/天",
    "_": [
        "2、陪床费：800元/天",
        "3、特定疾病住院津贴：300元/天，限30天每年",
        "4、护工津贴：300元/天，每年限15天",
        "5、耐用医疗设备：无限额",
        "6、特定疾病异地公共交通费：8000元限"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "64周岁"
}
$$, 15000, $$
{
    "amount": 1656,
    "unit": "元"
}
$$, $$
{
    "amount": 6000000,
    "unit": "元"
}
$$, '1年', '1-4类', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上公立医院普通部/特需部/国际部/指定私立医疗机构"
}
$$, $$
{
    "scope": "不限社保目录"
}
$$, $$
{
    "text": "1"
}
$$, '30天', $$
{
    "text": "0-17周岁未成年人不可单独投保；"
}
$$, NULL, $$
{
    "plan": "18-64周岁保费和0-17周岁连带投保保费"
}
$$, $$
{
    "familyDiscounts": [
        {
            "members": 2,
            "discount": 9.5
        }
    ]
}
$$, $$
{
    "text": "国际知名健康险服务商\n中国高端健康险市场份额第一\n深耕20年直付网络全国最广"
}
$$, NULL, NULL, NULL, 80.6);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('平安财险', '北极星中端医疗险（2023互联网）特需版计划', '中端医疗险', $$
{
    "一般住院医疗保险金": "200万",
    "重大疾病住院医疗保险金": "200万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前7天后30天",
    "其他责任": "1、床位费：1500元/天",
    "_": [
        "2、耐用医疗设备：无限额",
        "3、陪床费：600元/天",
        "4、重大疾病住院津贴：300元/天，年累计限60天",
        "5、重大疾病保险金：2万，给付",
        "可选1、院外特种药品医疗保险金：150万，限药品清单",
        "可选2、门急诊费用医疗保险金：35000元"
    ],
    "保障责任": "https://kdocs.cn/l/clJ7s0vtACd1"
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "60周岁"
}
$$, 0, $$
{
    "amount": 5677.36,
    "unit": "元"
}
$$, $$
{
    "amount": 2000000,
    "unit": "元"
}
$$, '1年', '除特殊职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上公立医院普通部/特需部/国际部及其他认可的医疗机构"
}
$$, $$
{
    "scope": "不限社保目录"
}
$$, $$
{
    "text": "1"
}
$$, '30天', $$
{
    "text": "1、未成年人可单独投保特需计划，不可附加门诊\n2、续保无宽限期，需要及时注意提醒续保"
}
$$, NULL, $$
{
    "plan": "院外特种药医疗保险金"
}
$$, NULL, $$
{
    "text": "国内保险老三家\n世界五百强前五十"
}
$$, NULL, NULL, NULL, 74.7);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('利宝', '柏世利享医疗保险', '中端医疗险', $$
{
    "一般住院医疗保险金": "600万",
    "重大疾病住院医疗保险金": "600万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、未成年人住院陪床费：800元/天（30天-6周岁限90天）",
    "_": [
        "2、重大疾病住院津贴：500元/天，累计限30天",
        "3、耐用医疗设备：无限额",
        "4、恶性肿瘤基因检费用：3万",
        "5、重大疾病异地就诊交通费（含一位陪同人员）：1万",
        "6、特定互联网医院在线问诊及药品费用保险金：在线问诊不限次数，药品费80%赔付，限200元/次，每年限5次",
        "可选1、门急诊医疗费用保险金：5万",
        "可选2、住院医疗费用拓展医院保障责任：拓展私立医院"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "65周岁"
}
$$, 15000, $$
{
    "amount": 1246.36332857143,
    "unit": "元"
}
$$, $$
{
    "amount": 6000000,
    "unit": "元"
}
$$, '1年', '除特殊职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "中国大陆公立医院（普通部、特需部、国际部）及指定私立医院，重大疾病扩展至中国香港地区指定医院"
}
$$, $$
{
    "scope": "不限社保目录"
}
$$, $$
{
    "text": "1"
}
$$, '30天', $$
{
    "text": "截止至2023年12月31日，未成年人单独投保规则：仅限15000元/30000元免赔额的住院方案（不含门急诊）"
}
$$, NULL, $$
{
    "plan": "门急诊医疗保险金"
}
$$, NULL, $$
{
    "text": "世界500强企业美国利宝互助保险集团在中国的全资子公司\n多险种财产险和意外险公司\n成立于1912年，中国总部重庆"
}
$$, NULL, NULL, NULL, 81.0);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('利宝', '智享安康中端医疗保险2024版（特需计划二）', '中端医疗险', $$
{
    "一般住院医疗保险金": "200万",
    "重大疾病住院医疗保险金": "200万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、床位费，膳食费：1500元/天",
    "_": [
        "2、陪床费：800元/天，针对非普通部",
        "3、护理费，重症监护室非，检查检验费，手术室费（含麻醉，材料费），医生费治疗费，手术植入器材费，药品费，西式理疗费：包含",
        "4、耐用医疗设备：包含（假肢，义肢，义乳需要机构的发票）",
        "5、耐用医疗设备费：限额2000元/年(限公立普通部、特需部、vip部、国际部100%报销，如院外购买80%报销）",
        "(仅包括外詈胰岛素泵、脚托、臂托、颈背托或束芾、轮椅(非电动轮椅)、助听器、外置心脏起博器、便式零化器)1.院内的需医疗机构的发票、2.院外购买需提供医疗机构的治疗单，医疗建议等",
        "6、临终关怀费：限30日",
        "(需主治医师提供住院部无对应药品或治疗仪器，确需在住院期间前往门诊部开具对应药品或治疗的说明。)",
        "8、重大疾病住院医疗保险金：200万",
        "9、重大疾病住院津贴：200元/天，限二级及以上公立医院普通部，限60日",
        "10、恶性肿瘤院外特药：150万，国内上市特药和博鳌乐城进口药"
    ],
    "7、住院期间、同一病因的门诊药品、治疗费": "限额1000元/年(限公立医院普通部、特需部、VIP部、国际部)",
    "保障责任": "https://kdocs.cn/l/cuxzZ7M4iM3W"
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "65周岁"
}
$$, 10000, $$
{
    "amount": 1103.6,
    "unit": "元"
}
$$, $$
{
    "amount": 2000000,
    "unit": "元"
}
$$, '1年', '除特殊职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上公立医院的普通部、特需部、VP部、国际部，上海质子重离子医院及本公司指定的民营医疗机构(详见《特\n需版医院列表》)"
}
$$, $$
{
    "scope": "不限社保目录"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '30天', $$
{
    "text": "可选1、门急诊医疗保险金：30000元，0免赔，100%赔付（以社保身份参保未使用社保结算60%），不限次数，次限额500元\n\n住院免赔额为0时才可以附加门急诊医疗\n\n可选2：重大疾病定额给付保险金：20000元，等待期60天"
}
$$, NULL, $$
{
    "plan": "计划"
}
$$, NULL, $$
{
    "text": "世界500强企业美国利宝互助保险集团在中国的全资子公司\n多险种财产险和意外险公司\n成立于1912年，中国总部重庆"
}
$$, NULL, NULL, NULL, 78.0);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('复星联合健康', '乐健中端医疗保险（2023版）计划八', '中端医疗险', $$
{
    "一般住院医疗保险金": "200万",
    "重大疾病住院医疗保险金": "200万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、床位费：2000元/天",
    "_": [
        "2、陪床费：600元/日",
        "3、耐用医疗设备：无限额",
        "4、临终关怀费用：累计限30日",
        "5、精神和心理障碍治疗费用：累计限额10万元",
        "6、恶性肿瘤基因检测费用：3000元",
        "7、重大疾病住院津贴：300元/天，年累计限60天",
        "8、恶性肿瘤特定药品费用医疗保险金：200万",
        "可选1、门急诊医疗保险金（1万/1.5万/2万/3.5万）",
        "可选2、恶性肿瘤海外特定地区海外医疗（计划一：新加坡/日本；计划二：美国）：100万，80%赔付"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "64周岁"
}
$$, 15000, $$
{
    "amount": 1382.8,
    "unit": "元"
}
$$, $$
{
    "amount": 2000000,
    "unit": "元"
}
$$, '1年', '1-4类', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "本机构指定或认可的医疗机构的普通部/特需门诊或病房"
}
$$, $$
{
    "scope": "不限社保目录"
}
$$, $$
{
    "text": "1"
}
$$, '30天', $$
{
    "text": "1、55周岁以上不可单独投保\n2、未成年人需要满足条件才可以单独投保\n   ①住院免赔额为0时，仅可单独投保住院计划1-4，不可附加门诊\n   ②免赔额≥5000时，可投保住院计划1-8，不可附加门诊"
}
$$, NULL, $$
{
    "plan": "门急诊医疗保险金"
}
$$, NULL, $$
{
    "text": "2017年成立\n健康险保险公司之一\n上海复星集团旗下"
}
$$, NULL, NULL, NULL, 76.6);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('中间带', '都邦财险中间带臻合意2024医疗保险计划一（含直付）', '中端医疗险', $$
{
    "一般住院医疗保险金": "800万",
    "重大疾病住院医疗保险金": "800万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、住院津贴（公立医院普通部）：500元/日，限60日",
    "_": [
        "2、住院精神和心理障碍治疗费：限10万元",
        "3、临终关怀：限30日",
        "4、恶性肿瘤靶向治疗费用基因检测费：限3万元",
        "5、恶性肿瘤院外特定药品费用：全额赔付",
        "6、在线问诊费用：指定互联网医院健康咨询，在线问诊费用全额理赔/图文问诊药品费用全额赔付（限2次）",
        "7、急救和转运费用：①急救费：全额赔付",
        "②异地就诊交通费 （含一位陪同人员）",
        "限额1万",
        "③紧急医疗转运费，遗体送返或就地安葬",
        "费：全额赔付"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "15天",
    "max": "65周岁"
}
$$, 15000, $$
{
    "amount": 1359,
    "unit": "元"
}
$$, $$
{
    "amount": 8000000,
    "unit": "元"
}
$$, '1年', '除特殊职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "公立医院，指定私立医院（质子重离子医院，医保定点非营利性民营机构普通部）"
}
$$, $$
{
    "scope": "不限社保目录"
}
$$, $$
{
    "text": "1"
}
$$, '30天', $$
{
    "text": "1、6岁以上未成年人可单独投保"
}
$$, NULL, NULL, $$
{
    "text": "2人投保95折，3人及以上9折"
}
$$, $$
{
    "text": "MediLink-Global中间带——全生命周期健康管理服务平台\n\n中间带始创于2005年，是中国规模和影响力显著的全生命周期健康管理服务平台。"
}
$$, NULL, NULL, NULL, 82.0);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('美亚财险', '“臻馨优选”个人中端医疗保障计划计划三', '中端医疗险', $$
{
    "一般住院医疗保险金": "300万",
    "重大疾病住院医疗保险金": "300万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、耐用医疗设备：限额2万",
    "_": [
        "2、临终关怀费用：累计赔偿天数30天",
        "3、重疾住院津贴：300元/天，限60天",
        "4、重症疾病异地治疗交通住宿保障：1万",
        "5、特定疾病药械费用补偿保障：300万，院外特定药品/临床急需进口药品/特定医疗器械费用",
        "可选1：意外门急诊医疗保障：3万",
        "可选2：门急诊医疗保障：1万，0免赔",
        "两个可选仅限选择一项"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "66周岁"
}
$$, 10000, $$
{
    "amount": 1107,
    "unit": "元"
}
$$, $$
{
    "amount": 3000000,
    "unit": "元"
}
$$, '1年', '除特殊职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上的公立医院普通部/特需部/国际版/质子重离子医院"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '30天', NULL, NULL, $$
{
    "plan": "有无社保"
}
$$, NULL, $$
{
    "text": "中国最大外资财产保险公司\n最早外资财产保险机构牌照\n风险评级：AA（2023年1季度）"
}
$$, NULL, NULL, NULL, 80.0);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('京东安联', '优享人生中端医疗保险计划二', '中端医疗险', $$
{
    "一般住院医疗保险金": "600万",
    "重大疾病住院医疗保险金": "600万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、限标准单人病房，限2500元/天，陪床费800元/天",
    "_": [
        "2、重大疾病住院津贴：5天免赔，300元/天，限30天/年",
        "3、住院护工费：300元/天，20天/年",
        "4、耐用医疗设备，手术机器人使用费，质子重离子治疗：全额赔付",
        "5、医疗运送和异地就医：救护车费用全额，重大疾病异地就诊交通，住宿费2万",
        "6、未成年人先天性疾病医疗费用：2万，首次投保时尚未发现的先天性疾病仅适用于0-18岁被保险人",
        "7、恶性肿瘤-重度特许医疗费用：",
        "临床急需进口药品费（大湾区、天津自贸区、海南博鳌肿瘤特药）：全额赔付",
        "恶性肿瘤（重度）精准治疗检测费：全额赔付",
        "其中：恶性肿瘤基因检测3万",
        "其中：肿瘤类器官药敏检测，限肿瘤二线或三线治疗，限一次，限食管癌、胃癌、肠癌、肝癌、肺癌、肾癌、乳腺癌、胆管癌",
        "可选1、特定重大疾病海外就医：800万，等待期90天，全球，特定重大疾病",
        "可选2、门急诊医疗费用：5万，等待期30天"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "70周岁"
}
$$, 10000, $$
{
    "amount": 1174,
    "unit": "元"
}
$$, $$
{
    "amount": 6000000,
    "unit": "元"
}
$$, '1年', '1-4类', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及以上公立医院普通部/特需部/国际部、指定私立医疗机构"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "text": "1"
}
$$, '30天', NULL, NULL, $$
{
    "plan": "有无社保"
}
$$, NULL, NULL, NULL, NULL, NULL, 82.6);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('泰康在线', '臻如意中端医疗保险计划（互联网专属）计划一', '中端医疗险', $$
{
    "一般住院医疗保险金": "150万",
    "重大疾病住院医疗保险金": "300万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前7天后30天",
    "其他责任": "1、特定既往症一般住院医疗保险金：150万",
    "_": [
        "因特定既往症发生的一般住院费用：免赔额2万，赔付比例：限公立医院：50%，罚则30%",
        "2、特定既往症重疾住院医疗保险金：300万",
        "因特定既往症发生的重疾住院费用：免赔额0元，赔付比例：公立医院：100%，罚则60%/民营：80%，罚则60%",
        "3、先进药械责任：",
        "①特药基因检测（因治疗癌症而必要进行的基因检测，限定于恶性肿瘤—重度）2万元，限1次，赔付比例100%，指定基因检测机构",
        "②特定疗法（2种CART+质子重离子等癌症创新疗法）",
        "150万，赔付比例100%，指定药房、指定医院",
        "③境内特定药械（198种内地肿瘤特药+14种内地特定器械）与海南进口药械共享保额300万，赔付比例100%，罚则60%，指定药房、指定医院",
        "④海南进口药械（54种临床急需进口药品+9种临床急需进口器械）与境内特定药械共享保额300万，赔付比例100%，海南指定医院",
        "4、重疾住院津贴：200元/天，年限60天",
        "可选1：门急诊医疗：2万；可选0/1000免赔；赔付比例-公立100%，罚则60%；",
        "因特定既往症导致的必需的门诊检查费用：限额300元，限1次，赔付比例-公立100%，罚则60%"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天",
    "max": "65周岁"
}
$$, 0, $$
{
    "amount": 1913,
    "unit": "元"
}
$$, $$
{
    "amount": 1500000,
    "unit": "元"
}
$$, '1年', '除特殊职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及以上公立医院普通部、特需部/国际部+指定民营医院普通部"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "罚则60%/民营：80%，罚则60%"
}
$$, '30天', NULL, NULL, $$
{
    "plan": "有无社保"
}
$$, NULL, NULL, $$
{
    "scope": "https://kdocs.cn/l/ci8ZPddfCl8C"
}
$$, $$
{
    "scope": "https://kdocs.cn/l/cdwzjx1cdtvJ"
}
$$, 'https://kdocs.cn/l/ccux26B97nya', 77.0);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('泰康在线', '臻安心·高疗无忧医疗险臻享版', '中端医疗险', $$
{
    "一般住院医疗保险金": "600万",
    "重大疾病住院医疗保险金": "600万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前7天后30天",
    "其他责任": "1、床位费：限双人病房，日限额1500元",
    "_": [
        "2、加床费：涵盖",
        "3、院内药品费：涵盖",
        "4、外购药品费：涵盖，无清单限制",
        "5、特定疾病门急诊及一般疾病特殊门急诊医疗保险金：400万",
        "①普通部：100%；特需部/国际部/普通部医保身份购买但自费身份就医：预授权后100%",
        "②药品费和医疗器械费（含约定的《特种药品清单》和《特定医疗起诶清单》中适用的药品和器械费用）：涵盖",
        "③确诊前门急诊费用：前30天"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天（含）",
    "max": "70周岁（含）"
}
$$, 0, $$
{
    "amount": 1504,
    "unit": "元"
}
$$, $$
{
    "text": "年最高800万"
}
$$, '1年', NULL, '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及以上公立医院普通部、我们扩展承保的私立医院的普通部、指定或认可的医院、指定或认可的医疗机构及扩展承保的二级及以上公立医院的特需医疗部与国际部、拓展承保的私立医院的特需医疗部与国际部"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "；普通部医保身份购买但自费身份就医：预授权后80%；特需部/国际部： 预授权后80%；当自担金额累计到25000元时，赔付比例100%\n特定疾病住院医疗普通部：100%；特需部/国际部/普通部医保身份购买但自费身份就医：预授权后100%"
}
$$, '30天', $$
{
    "text": "1、自担金额=责任范围内的医疗费用。其他途径补偿(包括但不限于社保报销、大病保险、医疗救助和商保赔付(含本产品赔付)等任何第三方获得的医疗费用补偿)\n2、院外恶性肿瘤特药扩展符合临床医学的先进治疗用药标准，且可由医生多点执业的私立医疗机构开具特药处方"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, $$
{
    "familyDiscounts": [
        {
            "members": 2,
            "discount": 9.5
        },
        {
            "members": 3,
            "discount": 0.9
        }
    ]
}
$$, NULL, NULL, NULL, 'https://kdocs.cn/l/cndLGUkkHkQc', 82.6);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('泰康在线', '臻安心·高疗无忧医疗险臻享Pro版', '中端医疗险', $$
{
    "一般住院医疗保险金": "600万",
    "重大疾病住院医疗保险金": "600万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前7天后30天",
    "其他责任": "1、床位费：限双人病房，日限额1500元",
    "_": [
        "2、加床费：涵盖",
        "3、院内药品费：涵盖",
        "4、外购药品费：涵盖，无清单限制",
        "5、特定疾病门急诊及一般疾病特殊门急诊医疗保险金：400万",
        "①普通部：100%；特需部/国际部/普通部医保身份购买但自费身份就医：预授权后100%",
        "②药品费和医疗器械费（含约定的《特种药品清单》和《特定医疗起诶清单》中适用的药品和器械费用）：涵盖",
        "③确诊前门急诊费用：前30天",
        "6、一般疾病门急诊医疗保险金：5万",
        "①赔付比例：普通部：30%；特需部/国际部：预授权后30%；当自担金额累计到25000元时，赔付比例100%",
        "②免赔额：0元"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "30天（含）",
    "max": "70周岁（含）"
}
$$, 0, $$
{
    "amount": 4484,
    "unit": "元"
}
$$, $$
{
    "text": "年最高800万（一般疾病门急诊不超过5万）"
}
$$, '1年', NULL, '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及以上公立医院普通部、我们扩展承保的私立医院的普通部、指定或认可的医院、指定或认可的医疗机构及扩展承保的二级及以上公立医院的特需医疗部与国际部、拓展承保的私立医院的特需医疗部与国际部"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "；普通部医保身份购买但自费身份就医：预授权后80%；特需部/国际部： 预授权后80%；当自担金额累计到25000元时，赔付比例100%\n特定疾病住院医疗普通部100%赔付，未经社保60%赔付/特需部，国际部以医保身份购买但自费身份就医经预授权后100%赔付"
}
$$, '30天', $$
{
    "text": "1、自担金额=责任范围内的医疗费用。其他途径补偿(包括但不限于社保报销、大病保险、医疗救助和商保赔付(含本产品赔付)等任何第三方获得的医疗费用补偿)\n2、院外恶性肿瘤特药扩展符合临床医学的先进治疗用药标准，且可由医生多点执业的私立医疗机构开具特药处方\n3、Pro版含门诊责任"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, $$
{
    "familyDiscounts": [
        {
            "members": 2,
            "discount": 9.5
        },
        {
            "members": 3,
            "discount": 0.9
        }
    ]
}
$$, NULL, NULL, NULL, NULL, 78.6);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('平安健康', '平安e生保互联网特需医疗保险', '中端医疗险', $$
{
    "一般住院医疗保险金": "200万",
    "重大疾病住院医疗保险金": "400万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "=DISPIMG(\"ID_2505AA3433A94BF2B5C159A625D6E51E\",1)"
}
$$, NULL, NULL, NULL, $$
{
    "min": "28天",
    "max": "65岁"
}
$$, 0, $$
{
    "amount": 1901,
    "unit": "元"
}
$$, $$
{
    "amount": 4000000,
    "unit": "元"
}
$$, '1年', '除特殊职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "国内二级及二级以上公立医院普通部，特需部，国际部，VIP部和约定的私立医院"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0
}
$$, '30天，意外无等待期', NULL, NULL, $$
{
    "plan": "有无社保"
}
$$, NULL, $$
{
    "text": "国内保险老三家\n世界五百强前五十"
}
$$, NULL, NULL, NULL, 79.4);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('平安健康', '平安e生安心·中端医疗险（易保版）', '中端医疗险', $$
{
    "一般住院医疗保险金": "400万",
    "重大疾病住院医疗保险金": "400万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、特需医疗保险金：400万",
    "_": [
        "2、质子重离子医疗保险金：400万，床位费1500元/天",
        "3、院外恶性肿瘤特定用药基因检测费用保险金：2万",
        "4、院外恶性肿瘤特定药品费用保险金：400万，212种"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "18",
    "max": "65岁"
}
$$, 30, $$
{
    "amount": 1480,
    "unit": "元"
}
$$, $$
{
    "amount": 4000000,
    "unit": "元"
}
$$, '1年', '除特殊职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "一般医疗保险金的医院范围为二级以上（含二级）公立医院的普通部（不包含公立医院的特需部、vip部、国际部或国际医疗中心）以及本主险合同约定的其他医院，特需医疗保险金的医院范围为二级以上（含二级）公立医院的特需部、vip部、国际部或国际医疗中心以及本主险合同约定的其他医院"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "赔付，社保罚则60%\n特需医疗保险金：100%赔付\n院外恶性肿瘤特定药品费用医疗保险金：100%赔付，社保罚则60%"
}
$$, '30天', $$
{
    "text": "1、过去6个月内：\n存在以下检查异常或被医生建议进行以下检查: 核磁共振（MRI）、钼靶、病理检查或活检、穿刺、内窥镜、血液涂片细胞学的检查、肿瘤标志物、CT。\n\n2、过去2年内：\n是否接受过住院手术或单次住院超过5天？是否曾被医生建议住院或手术但未行住院或手术？\n\n3、过去5年内是否患有或被告知患有以下任一疾病或情况：\n①肿瘤类疾病：恶性肿瘤、原位癌、颅内占位、肺部结节大小≥6mm（已手术切除除外）、甲状腺结节4级及以上（已手术切除除外）、乳腺结节4级及以上（已手术切除除外）；\n②循环、呼吸系统疾病：慢性阻塞性肺疾病（COPD）、动脉瘤、冠心病、主动脉夹层、心肌梗死、心绞痛、心脏瓣膜疾病、心肌病、心功能不全II级及以上、脑卒中、脑血管畸形、高血压且存在清单中疾病（请点击链接查看疾病清单）、糖尿病且存在清单中疾病（请点击链接查看疾病清单）；\n③其他疾病：椎管狭窄、自身免疫性肝病、慢性病毒性肝炎且伴有肝功能异常、肝纤维化或肝硬化、慢性肾病、再生障碍性贫血、帕金森病、瘫痪、系统性红斑狼疮、股骨头/颈坏死、重度骨性关节炎、接受过重大器官（心、肝、肾、肺）移植或造血干细胞移植。\n\n4、职业情况\n被保险人目前专职或兼职从事属于《特殊职业类别表》中所列种类的职业。（注：这些职业之外的其他职业可以投保；曾经从事这些职业，但已转行为其他职业的，可以投保）"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, NULL, $$
{
    "text": "国内保险老三家\n世界五百强前五十"
}
$$, NULL, NULL, NULL, 80.0);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('众安保险', '众民保·中高端医疗险', '中端医疗险', $$
{
    "一般住院医疗保险金": "300万",
    "重大疾病住院医疗保险金": "300万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、重大疾病医疗及外购药械费用：床位费限2500元/天，陪床费1500元/天",
    "_": [
        "2、质子重离子医疗：300万，0免赔，100%赔付，床位费2500元/天",
        "3、重大疾病异地转诊公共交通费用及住宿费用保险金：1万，0免赔，100%赔付",
        "4、特定药品费用医疗保险金：156种院外特药（含3款CAR-T,钇90）；300万，0免赔，100%赔付",
        "5、救护车费用保险金：1000元，0免赔，100%赔付",
        "6、重大疾病康复医疗费用：1万元，0免赔，100%赔付"
    ],
    "保障责任": "https://kdocs.cn/l/ceqd8RVlL8Vq"
}
$$, NULL, NULL, NULL, $$
{
    "min": "18",
    "max": "80周岁"
}
$$, 0, $$
{
    "amount": 1306,
    "unit": "元"
}
$$, $$
{
    "amount": 6010000,
    "unit": "元"
}
$$, '1年', '除特殊职业', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上公立医院普通部及指定102家民营医疗机构的普通部\n其中，重大疾病可扩展至上述医院的特需部、国际部、VIP部"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 0.5,
    "note": "赔付，20000元以上100%赔付\n重大疾病医疗100%赔付"
}
$$, '30天，意外无等待期', $$
{
    "text": "不承担初次投保前、等待期内或非连续重新投保前已罹患的以下5类疾病，及因该疾病或并发症导致的医疗费用；\n不承担初次投保前或非连续重新投保前已发生意外事故导致的相关医疗费用：\n（1）肿瘤类：恶性肿瘤*、颅内肿瘤或占位、脊髓肿瘤或占位、肝占位；\n（2）肝肾疾病类：慢性肾病（CKD4期及以上）、肝衰竭、肝硬化；\n（3）心脑血管及糖脂代谢疾病类：冠心病、心肌梗死、心功能不全(心功能Ⅲ级及以上)、主动脉夹层、心肌病、\n房颤/房扑、肺动脉高压、脑梗死、脑出血、心瓣膜病、高血压伴并发症、糖尿病伴并发症；\n（4）肺部疾病类：慢性阻塞性肺病、呼吸衰竭、间质性肺病；\n（5）其他：帕金森病，动脉瘤，系统性红斑狼疮，再生障碍性贫血、骨髓增生异常综合征，嗜（噬）血细胞综合\n征，胰腺炎，溃疡性结肠炎，克罗恩病，骨坏死，脊椎/脊柱/胸廓疾病*，癫痫，瘫痪；\n（6）意外：初次投保前或非连续重新投保前已发生的意外事故"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, $$
{
    "familyDiscounts": [
        {
            "members": 2,
            "discount": 9.5
        },
        {
            "members": 3,
            "discount": 0.9
        },
        {
            "members": 4,
            "discount": 8.5
        }
    ]
}
$$, $$
{
    "text": "全国首家互联网保险公司\n由蚂蚁金服、中国平安和腾讯于2013年联合发起设立"
}
$$, NULL, NULL, NULL, 81.4);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('友邦人寿', '长保康惠长期医疗保险（20年保证续保）', '长期百万医疗险', $$
{
    "一般住院医疗保险金": "200万",
    "重大疾病住院医疗保险金": "400万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天",
    "其他责任": "1、质子重离子医疗保险：200万",
    "_": [
        "2、恶性肿瘤cart疗法院外购药补偿金：400万，无免赔额",
        "3、重症监护病房住院津贴：500元/天"
    ]
}
$$, NULL, NULL, NULL, $$
{
    "min": "出生满7天",
    "max": "65周岁"
}
$$, 5000, $$
{
    "amount": 475,
    "unit": "元"
}
$$, $$
{
    "text": "年200万/保证续保期间总限额1000万"
}
$$, '1年', '1-4类', '20年（费率可调）', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上公立医院普通部/上海质子重离子医院"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算60%"
}
$$, '30天', $$
{
    "text": "保证续保20年"
}
$$, NULL, $$
{
    "plan": "有无社保"
}
$$, NULL, NULL, NULL, NULL, NULL, 79.2);
INSERT INTO insurance_products_medical (company_name, product_name, insurance_type, coverage_content, exclusion_clause, renewable, underwriting_rules, entry_age, deductible, premium, coverage_amount, coverage_period, occupation, payment_period, coverage_region, hospital_scope, reimbursement_scope, reimbursement_ratio, waiting_period, remarks, age, plan_choice, discount, company_intro, drug_list, service_manual, clause_link, total_score) VALUES ('友邦人寿', '尊享康惠计划2', '百万医疗险', $$
{
    "一般住院医疗保险金": "10万",
    "重大疾病住院医疗保险金": "10万",
    "特殊门诊": "含",
    "门诊手术": "含",
    "住院前后门急诊": "前30天后30天"
}
$$, NULL, NULL, NULL, $$
{
    "min": "出生满7天",
    "max": "65周岁"
}
$$, 0, $$
{
    "amount": 1036,
    "unit": "元"
}
$$, $$
{
    "amount": 100000,
    "unit": "元"
}
$$, '1年', '1-4类', '1年', $$
{
    "region": "中国大陆（不含港澳台）"
}
$$, $$
{
    "scope": "二级及二级以上公立医院（入住特需，VIP，国际部不承担床位费膳食费）/上海质子重离子医院"
}
$$, $$
{
    "scope": "不限社保"
}
$$, $$
{
    "ratio": 1.0,
    "note": "若以社保身份参保未使用社保结算80%"
}
$$, '30天', $$
{
    "text": "二级及二级以上公立医院普通部全额报销，特需部，VIP部，国际部和其他高端病房和门急诊不承担床位费和膳食费，其他费用承担"
}
$$, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 75.2);
