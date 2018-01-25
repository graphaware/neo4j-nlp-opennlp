package com.graphaware.nlp.processor.opennlp;

import com.graphaware.nlp.NLPIntegrationTest;
import com.graphaware.nlp.dsl.AbstractDSL;
import org.neo4j.kernel.impl.proc.Procedures;
import org.reflections.Reflections;

import java.util.Set;

public class OpenNLPIntegrationTest extends NLPIntegrationTest {

    @Override
    protected void registerProceduresAndFunctions(Procedures procedures) throws Exception {
        super.registerProceduresAndFunctions(procedures);
        Reflections reflections = new Reflections("com.graphaware.nlp.dsl");
        Set<Class<? extends AbstractDSL>> cls = reflections.getSubTypesOf(AbstractDSL.class);
        for (Class c : cls) {
            procedures.registerProcedure(c);
            procedures.registerFunction(c);
        }
    }
}
