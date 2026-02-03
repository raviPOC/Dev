public with sharing class FormBuilderController {
    @AuraEnabled(cacheable=true)
    public static List<Form__c> listForms() {
        return [
            SELECT Id, Name, Target_Object__c, Active__c, LastModifiedDate
            FROM Form__c
            ORDER BY LastModifiedDate DESC
            LIMIT 200
        ];
    }

    @AuraEnabled(cacheable=true)
    public static Form__c getForm(Id formId) {
        if (formId == null) return null;
        return [
            SELECT Id, Name, Target_Object__c, Active__c, Definition__c
            FROM Form__c
            WHERE Id = :formId
            LIMIT 1
        ];
    }

    @AuraEnabled
    public static Id saveForm(
        Id formId,
        String name,
        String targetObjectApiName,
        Boolean isActive,
        String definitionJson
    ) {
        if (String.isBlank(name)) {
            throw new AuraHandledException('Form name is required.');
        }
        if (String.isBlank(targetObjectApiName)) {
            throw new AuraHandledException('Target object API name is required.');
        }
        if (String.isBlank(definitionJson)) {
            throw new AuraHandledException('Form definition is empty.');
        }

        Form__c f = (formId == null) ? new Form__c() : new Form__c(Id = formId);
        f.Name = name.trim();
        f.Target_Object__c = targetObjectApiName.trim();
        f.Active__c = (isActive == null) ? false : isActive;
        f.Definition__c = definitionJson;
        upsert f;
        return f.Id;
    }
}

