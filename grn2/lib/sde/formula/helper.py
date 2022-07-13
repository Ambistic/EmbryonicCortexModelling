def formula_from_dict(dictf):
    from lib.sde.formula import And, Or, Var, One, Zero
    op = dictf["op"]

    if op == "And":
        return And(*[formula_from_dict(child)
                     for child in dictf["children"]])

    if op == "Or":
        return Or(*[formula_from_dict(child)
                    for child in dictf["children"]])

    if op == "Var":
        return Var(dictf["val"], sign=dictf["sign"], weight=dictf["weight"])

    if op == "One":
        return One(weight=dictf["weight"])

    if op == "Zero":
        return Zero(weight=dictf["weight"])
