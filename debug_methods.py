import re

# The actual controller code from the debug output
controller_code = """package com.iemr.common.notification.agent;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;
import com.iemr.common.notification.agent.DTO.AlertAndNotificationChangeStatusDTO;
import com.iemr.common.notification.agent.DTO.AlertAndNotificationCountDTO;
import com.iemr.common.notification.agent.DTO.AlertAndNotificationSetDeleteDTO;
import com.iemr.common.notification.agent.DTO.SuccessObjectDTO;
import com.iemr.common.notification.agent.DTO.UserNotificationDisplayMaxDTO;
import com.iemr.common.notification.agent.DTO.UserNotificationDisplayMinDTO;
import com.iemr.common.notification.util.InputMapper;
import com.iemr.common.utils.response.OutputResponse;
import io.lettuce.core.dynamic.annotation.Param;
@CrossOrigin()
@RequestMapping(value = "/notification")
@RestController
public class UserNotificationMappingController
{
private final Logger logger = LoggerFactory.getLogger(this.getClass().getName());
private OutputResponse response = new OutputResponse();
@Autowired
UserNotificationMappingService notificationService;
@CrossOrigin()
@RequestMapping(value = "/getAlertsAndNotificationCount", method = RequestMethod.POST,
produces = MediaType.APPLICATION_JSON_VALUE, consumes = MediaType.APPLICATION_JSON_VALUE,
headers = "Authorization")
public @ResponseBody String
getAlertsAndNotificationCount(@Param(value = "{" + "\"userID\":\"Integer\"\n"
+ "\"roleID\":\"Integer\"\n" + "\"providerServiceMapID\":\"Integer\"\n"
+ "\"workingLocationID\":\"Integer\"\n" + "}") @RequestBody String getNotificationCountFilter)
{
logger.info("UserNotificationMappingController.getAlertsAndNotificationCount - start");
try
{
JsonElement json = new JsonParser().parse(getNotificationCountFilter);
logger.info("UserNotificationMappingController.getAlertsAndNotificationCount : json");
UserNotificationDisplayMinDTO inputData =
InputMapper.getInstance().gson().fromJson(json, UserNotificationDisplayMinDTO.class);
AlertAndNotificationCountDTO dtoOut = notificationService.getAlertAndNotificationCount(inputData);
response.setResponse(InputMapper.getInstance().gson().toJson(dtoOut));
logger.info("UserNotificationMappingController.getAlertsAndNotificationCount: success - finish");
return response.toString();
} catch (Exception e)
{
response.setError(e);
logger.error("UserNotificationMappingController.getAlertsAndNotificationCount: failure - finish",e.getMessage());
return response.toString();
}
}
@CrossOrigin()
@RequestMapping(value = "/getAlertsAndNotificationDetail", method = RequestMethod.POST,
produces = MediaType.APPLICATION_JSON_VALUE, consumes = MediaType.APPLICATION_JSON_VALUE,
headers = "Authorization")
public @ResponseBody String
getAlertsAndNotificationDetail(@Param(value = "{" + "\"userID\":\"Integer\"\n"
+ "\"roleID\":\"Integer\"\n" + "\"notificationTypeID\":\"Integer\"\n"
+ "\"providerServiceMapID\":\"Integer\"\n" + "\"workingLocationID\":\"Integer\"\n"
+ "}") @RequestBody String getNotificationDetailFilter)
{
OutputResponse output = new OutputResponse();
logger.info("UserNotificationMappingController.getAlertsAndNotificationDetail - start");
try
{
JsonElement json = new JsonParser().parse(getNotificationDetailFilter);
UserNotificationDisplayMaxDTO inputData =
InputMapper.getInstance().gson().fromJson(json, UserNotificationDisplayMaxDTO.class);
List<UserNotificationMapping> list = notificationService.getAlertAndNotificationDetail(inputData);
System.out.println("hello");
output.setResponse(list.toString());
logger.info("UserNotificationMappingController.getAlertsAndNotificationDetail: success - finish");
} catch (Exception e)
{
output.setError(e);
}
return output.toString();
}
@CrossOrigin()
@RequestMapping(value = "/changeNotificationStatus", method = RequestMethod.POST,
produces = MediaType.APPLICATION_JSON_VALUE, consumes = MediaType.APPLICATION_JSON_VALUE,
headers = "Authorization")
public @ResponseBody String changeNotificationStatus(@Param(value = "{" + "\"notficationStatus\":\"String\"\n"
+ "\"notificationMapIDList\":\"List<Integer>\"\n" + "}") @RequestBody String changeNotificationStatusFilter)
{
logger.info("UserNotificationMappingController.changeNotificationStatus - start");
try
{
JsonElement json = new JsonParser().parse(changeNotificationStatusFilter);
AlertAndNotificationChangeStatusDTO dto =
InputMapper.getInstance().gson().fromJson(json, AlertAndNotificationChangeStatusDTO.class);
if (dto.getUserNotificationMapIDList().size() == 1)
{
notificationService.markNotificationSingle(dto.getNotficationStatus(),
dto.getUserNotificationMapIDList().get(0));
} else if (dto.getUserNotificationMapIDList().size() > 1)
{
notificationService.markNotificationList(dto.getNotficationStatus(),
dto.getUserNotificationMapIDList());
} else
{
response.setError(new Throwable("Missing mandatory Parameter - at least 1 NotificationMapId needed."));
logger.info("UserNotificationMappingController.changeNotificationStatus: failure - finish");
return response.toString();
}
SuccessObjectDTO obj = new SuccessObjectDTO();
obj.setOperation(dto.getNotficationStatus());
obj.setStatus("success");
response.setResponse(InputMapper.getInstance().gson().toJson(obj));
logger.info("UserNotificationMappingController.changeNotificationStatus: success - finish");
return response.toString();
} catch (Exception e)
{
response.setError(e);
logger.error("UserNotificationMappingController.changeNotificationStatus: failure - finish");
return response.toString();
}
}
@CrossOrigin()
@RequestMapping(value = "/markDelete", method = RequestMethod.POST, produces = MediaType.APPLICATION_JSON_VALUE,
consumes = MediaType.APPLICATION_JSON_VALUE, headers = "Authorization")
public @ResponseBody String markDelete(@Param(value = "{" + "\"isDeleted\":\"Boolean\"\n"
+ "\"userNotificationMapIDList\":\"List<Integer>\"\n" + "}") @RequestBody String markDeleteFilter)
{
logger.info("UserNotificationMappingController.markDelete - start");
try
{
JsonElement json = new JsonParser().parse(markDeleteFilter);
AlertAndNotificationSetDeleteDTO dto =
InputMapper.getInstance().gson().fromJson(json, AlertAndNotificationSetDeleteDTO.class);
if (dto.getUserNotificationMapIDList().size() == 1)
{
notificationService.deleteNotificationSingle(dto.getIsDeleted(),
dto.getUserNotificationMapIDList().get(0));
} else if (dto.getUserNotificationMapIDList().size() > 1)
{
notificationService.deleteNotificationList(dto.getIsDeleted(), dto.getUserNotificationMapIDList());
} else
{
response.setError(new Throwable("Missing mandatory Parameter - at least 1 NotificationMapId needed."));
logger.info("UserNotificationMappingController.markDelete: failure - finish");
return response.toString();
}
SuccessObjectDTO obj = new SuccessObjectDTO();
obj.setOperation("isDeleted = " + dto.getIsDeleted().toString());
obj.setStatus("success");
response.setResponse(InputMapper.getInstance().gson().toJson(obj));
logger.info("UserNotificationMappingController.markDelete: success - finish");
return response.toString();
} catch (Exception e)
{
response.setError(e);
logger.info("UserNotificationMappingController.markDelete: failure - finish");
return response.toString();
}
}
}"""

def extract_public_methods(class_code: str) -> set:
    """
    Extracts all public method names from the given Java class code using regex.
    Includes constructors and static methods. Handles annotations, generics, and line breaks.
    """
    methods = set()
    
    print("=== Testing extract_public_methods ===")
    
    # Match public constructors (same name as class)
    constructor_pattern = re.compile(r'public\s+(\w+)\s*\([^)]*\)\s*\{')
    for match in constructor_pattern.finditer(class_code):
        methods.add(match.group(1))
        print(f"Found constructor: {match.group(1)}")
    
    # Current method pattern
    method_pattern = re.compile(
        r'(?:@[\w.]+\s*)*'  # Annotations (optional, possibly multiline)
        r'public\s+'         # public modifier
        r'(?:[\w<>\[\],\s]+\s+)*'  # Modifiers, generics, return type (optional, multiline)
        r'(\w+)\s*\(',     # Method name
        re.MULTILINE
    )
    
    print("\n=== Testing current method pattern ===")
    for match in method_pattern.finditer(class_code):
        methods.add(match.group(1))
        print(f"Found method: {match.group(1)}")
    
    # NEW: Better pattern for Spring controller methods
    print("\n=== Testing improved pattern for Spring controllers ===")
    improved_pattern = re.compile(
        r'public\s+'  # public modifier
        r'(?:@[\w.]+\s*)*'  # Annotations after public (like @ResponseBody)
        r'(?:[\w<>\[\],\s]+\s+)*'  # Return type
        r'(\w+)\s*\(',  # Method name
        re.MULTILINE
    )
    
    for match in improved_pattern.finditer(class_code):
        methods.add(match.group(1))
        print(f"Found method (improved): {match.group(1)}")
    
    # NEW: Even more flexible pattern
    print("\n=== Testing very flexible pattern ===")
    flexible_pattern = re.compile(
        r'public\s+'  # public modifier
        r'(?:[^{]*?)'  # Any characters until method name (non-greedy)
        r'(\w+)\s*\([^)]*\)\s*\{',  # Method name followed by parameters and opening brace
        re.MULTILINE | re.DOTALL
    )
    
    for match in flexible_pattern.finditer(class_code):
        methods.add(match.group(1))
        print(f"Found method (flexible): {match.group(1)}")
    
    # Match public static methods that might have different patterns
    static_method_pattern = re.compile(r'public\s+static\s+(?:final\s+)?(?:<[^>]+>\s+)?[\w<>,\[\]]+\s+(\w+)\s*\(')
    for match in static_method_pattern.finditer(class_code):
        methods.add(match.group(1))
        print(f"Found static method: {match.group(1)}")
    
    print(f"\n=== Final result: {methods} ===")
    return methods

# Test the function
extract_public_methods(controller_code) 