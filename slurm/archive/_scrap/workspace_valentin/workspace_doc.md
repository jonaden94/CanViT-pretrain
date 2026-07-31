Workspaces allow users to request temporary storage with an expiration date on the high performance filesystems. This increases the efficiency and overall performance of the HPC system, as files that are no longer required for compute jobs do not accumulate and fill up these filesystems.

The typical workflow is that the workspaces commands are used to request and create a directory on the appropriate filesystem before a new chain of compute jobs is launched. After the job chain has finished, the results should be copied to a more permanent location in a data store shared with other project members.

Each project and project user can have multiple workspaces per data store, each with their own expiration dates. After the expiration date, the expired workspace and the data within it are archived in an inaccessible location and eventually deleted.

The concept of workspaces has become quite popular and is applied in a large number of HPC centers around the world. We use the well known HPC Workspace tooling from Holger Berger to manage workspaces.

 Info
All Slurm jobs get their own temporary storage directories on the nodes themselves and the fastest shared filesystems available to the particular nodes, which are cleaned up when the job is finished. If you only need temporary storage for the lifetime of the job, those directories are better suited than workspaces. See Temporary Storage for more information.

Workspaces are meant for active data and are configured for high aggregate bandwidth (across all files and nodes at once) at the expense of robustness. The bandwidth for single files varies based on the underlying filesystem and how they are stored, please check the filesystem pages for more details.

Common workflows are to use workspaces for temporary data, or store data in a compact form in a Project or COLD data store and then copy and/or decompress it to a higher performance workspace to be used for a short period of time. The workspaces have a generous quota for the whole project (the finite lifetimes of workspaces help protect against running out of space as well).

The data stores for workspaces available to each kind of project are:

Kind of Project	Name	Filesystem	Purpose
SCC, NHR, REACT	Ceph HDD WS (ceph-hdd)	CephFS	Medium-term storage for NHR Test Accounts and for projects that have a temporary demand for more storage space
SCC, NHR, REACT	Ceph SSD WS (ceph-ssd)	CephFS	Significantly faster than Ceph HDD, useful for jobs that need storage available from all Cluster Islands
NHR	Lustre RZG WS (lustre-rzg)	Lustre	High performance WORK filesystem for users of Grete and Emmy P3
NHR	Lustre MDC WS (lustre-mdc)	Lustre	High performance WORK filesystem for users of Emmy P2
CIDBN	BeeGFS CIDBN HDD WS (beegfs-cidbn-hdd)	BeeGFS	Medium performance HDD filesystem for users of CIDBN
CIDBN	BeeGFS CIDBN SSD WS (beegfs-cidbn-ssd)	BeeGFS	High performance SSD filesystem for users of CIDBN
Workspace Basics
The six basic commands to handle workspaces and manage their lifecycles are:

Command	Description
ws_allocate	Create a workspace
ws_extend	Extend a workspace’s expiration date
ws_list	List all workspaces or available data stores for them
ws_release	Release a workspace (files will be deleted after a grace period)
ws_restore	Restore a previously released workspace (if in grace period)
ws_register	Manage symbolic links to workspaces
All six commands have help messages accessible by COMMAND -h and man pages accessible by man COMMAND.

 Note
None of the workspace commands except for ws_list and ws_register work inside user namespaces such as created by running containers of any sort (e.g. Apptainer) or manually with unshare. This includes the JupyterHPC service and HPC desktops.

Workspaces are created with the requested expiration time (each data store has a maximum allowed value). The default expiration time if none is requested is 1 day. Workspaces can have their expiration extended a limited number of times. After a workspace expires, it is released. Released workspaces can be restored for a limited grace period, after which the data is permanently deleted. Note that released workspaces still count against your filesystem quota during the grace period. All workspace tools use the -F <name> option to control which data store to operate on, where the default depends on the kind of project. The various limits, default data store for each kind of project, as well as which cluster islands each data store is meant to be used from and their purpose/specialty are:

Name	Default	Islands	Purpose/Specialty	Time Limit	Extensions	Grace Period
ceph-ssd	NHR
SCC
REACT	all	all-rounder	30 days	2 (90 days max lifetime)	30 days
ceph-hdd		all	large size	60 days	5 (360 days max lifetime)	30 days
lustre-mdc		Emmy P2
Emmy P1	Max performance	30 days	2 (90 days max lifetime)	30 days
lustre-rzg		Grete
Emmy P3	Max performance	30 days	2 (90 days max lifetime)	30 days
beegfs-cidbn-hdd		CIDBN	all-rounder	30 days	2 (90 days max lifetime)	30 days
beegfs-cidbn-ssd		CIDBN	Max performance	30 days	2 (90 days max lifetime)	30 days
 Note
Only workspaces on data stores mounted on a particular node are visible and can be managed (allocate, release, list, etc.). If the data store that is the default for your kind of project is not available on a particular node, the special DONT_USE data store will be the default that doesn’t support allocation (you must then specify -F <name> in all cases). See Cluster Storage Map for more information on which filesystems are available where.

Managing Workspaces
Allocating
Workspaces are created via

ws_allocate [OPTIONS] WORKSPACE_NAME DURATION
The duration is given in days and workspace names are limited to ASCII letters, numbers, dashes, dots, and underscores. The most important options (run ws_allocate -h to see the full list) are:

  -F [ --filesystem ] arg   filesystem
  -r [ --reminder ] arg     reminder to be sent n days before expiration
  -m [ --mailaddress ] arg  mailaddress to send reminder to
  -g [ --group ]            group workspace
  -G [ --groupname ] arg    groupname
  -c [ --comment ] arg      comment
Use --reminder <days> --mailaddress <email> to be emailed a reminder the specified number of days before the workspace expires. Use --group --groupname <group> to make the workspace readable and writable by the members of the specified group, however this only works for members of the group that are also members of the same project. Members of other projects (than the username you used to create the workspace) cannot access it, even if you have a common POSIX group and use the group option. Thus, usually the only value that makes sense is the group HPC_<project>, which can be conveniently generated via "HPC_${PROJECT_NAME}" in the shell. If you run ws_allocate for a workspace that already exists, it just prints its path to stdout, which can be used if you forgot the path (you can also use ws_list).

 Note
Workspace names and their paths are not private. Any user on the cluster can see which workspaces exist and who created them. However, other usernames cannot access workspaces unless the workspace was created with --group --groupname <group> and they are both a member of the same project and of that group.

To create a workspace named MayData on ceph-ssd with a lifetime of 6 days which emails a reminder 2 days before expiration, you could run:

~ $ ws_allocate -F ceph-ssd -r 2 -m myemail@example.com MayData 6
Info: creating workspace.
/mnt/ceph-ssd/workspaces/ws/nhr_internal_ws_test/u17588-MayData
remaining extensions  : 2
remaining time in days: 6
The path to the workspace is printed to stdout while additional information is printed to stderr. This makes it easy to get the path and save it as an environment variable:

~ $ WS_PATH=$(ws_allocate -F ceph-ssd -r 2 -m myemail@example.com MayData 6)
Info: creating workspace.
remaining extensions  : 2
remaining time in days: 6
~ $ echo $WS_PATH
/mnt/ceph-ssd/workspaces/ws/nhr_internal_ws_test/u17588-MayData
You can set defaults for the duration as well as the --reminder, --mailaddress, and --groupname options by creating a YAML config file at $HOME/.ws_user.conf formatted like this:

duration: 15
groupname: HPC_foo
reminder: 3
mail: myemail@example.com
Listing
Use the ws_list command to list your workspaces, the workspaces made available to you by other users in your project, and the available datastores. Use the -l option to see the available data stores for your username on the particular node you are currently using:

~ $ ws_list -l
available filesystems:
ceph-ssd
lustre-rzg
DONT_USE (default)
Note that the special unusable location DONT_USE is always listed as the default even if the default for the kind of project your username is in is available.

Running ws_list by itself lists your workspaces that can be accessed from the node (not all data stores are available on all nodes). Add the -g option to additionally list the ones made available to you by other users. If you run ws_list -g after creating the workspace in the previous example, you would get:

~ $ ws_list -g
id: MayData
     workspace directory  : /mnt/ceph-ssd/workspaces/ws/nhr_internal_ws_test/u17588-MayData
     remaining time       : 5 days 23 hours
     creation time        : Thu Jun  5 15:01:14 2025
     expiration date      : Wed Jun 25 15:01:14 2025
     filesystem name      : ceph-ssd
     available extensions : 2
Extending
The expiration time of a workspace can be extended with ws_extend up to the maximum time allowed for an allocation on the chosen data store. It is even possible to reduce the expiration time by requesting a value lower than the remaining duration. The number of times a workspace on a particular data store can be extended is also limited, to two times on our cluster. Workspaces are extended by running:

ws_extend -F DATA_STORE WORKSPACE_NAME DURATION
Don’t forget to specify the data store with -F <data-store>. For example, to extend the workspace allocated in the previous example to 20 days, run:

~ $ ws_extend -F ceph-ssd MayData 20
Info: extending workspace.
Info: reused mail address example@example.com
/mnt/ceph-ssd/workspaces/ws/nhr_internal_ws_test/u17588-MayData
remaining extensions  : 1
remaining time in days: 20
Releasing
A workspace can be released before its expiration time by running:

ws_release -F DATA_STORE [OPTIONS] WORKSPACE_NAME
The most important option here is --delete-data which causes the workspace’s data to be deleted immediately (remember, the data stores for workspaces have NEITHER backups NOR snapshots, so the data is lost forever). Otherwise, the workspace will be set aside and remain restorable for the duration of the grace period of the respective data store.

 Note
Workspaces released without the --delete-data option still count against your project’s quota until the grace period is over and they are automatically cleaned up.

Restoring
A released or expired workspace can be restored within the grace period using the ws_restore command. Use ws_restore -l to list your restorable workspaces and to get their full IDs. If the previously created example workspace was released, you would get:

~ $ ws_restore -l
ceph-ssd:
u17588-MayData-1749129222
        unavailable since Thu Jun  5 15:13:42 2025
lustre-rzg:
DONT_USE:
Note that the full ID of a restorable workspace includes your username, the workspace name, and the unix timestamp from when it was released. In order to restore a workspace, you must first have another workspace available on the same data store to restore it into. Then, you would call the command like this:

ws_restore -F DATA_STORE WORKSPACE_ID_TO_RESTORE DESTINATION_WORKSPACE
and it will ask you to type back a set of randomly generated characters before restoring (restoration is interactive and is not meant to be scripted). The workspace being restored is placed as a subdirectory in the destination workspace with its ID. Using the previous example, one could create a new workspace MayDataRestored and restore the workspace to it via:

~ $ WS_DIR=$(ws_allocate -F ceph-ssd MayDataRestored 30)
Info: creating workspace.
remaining extensions  : 2
remaining time in days: 30
~ $ ws_restore -F ceph-ssd u17588-MayData-1749129222 MayDataRestored
to verify that you are human, please type 'tafutewisu': tafutewisu
you are human
Info: restore successful, database entry removed.
~ $ ls $WS_DIR
u17588-MayData-1749129222
Managing Links
Keeping track of the paths to each workspace can be difficult sometimes. You can use the ws_register DIR command to setup symbolic links (symlinks) to all of your workspaces in the directory DIR. After doing that, each of your workspaces has a symlink <dir>/<datastore>/<username>-<workspacename>.

​
~ $ mkdir ws-links
~ $ ws_register ws-links
keeping link  ws-links/ceph-ssd/u17588-MayDataRestored
~ $ ls -lh ws-links/
total 0
drwxr-x--- 2 u17588 GWDG 4.0K Jun  5 15:47 DONT_USE
drwxr-x--- 2 u17588 GWDG 4.0K Jun  5 15:47 ceph-ssd
drwxr-x--- 2 u17588 GWDG 4.0K Jun  5 15:47 lustre-rzg
~ $ ls -lh ws-links/*
ws-links/DONT_USE:
total 0

ws-links/ceph-ssd:
total 0
lrwxrwxrwx 1 u17588 GWDG 71 Jun  5 15:47 u17588-MayDataRestored -> /mnt/ceph-ssd/workspaces/ws/nhr_internal_ws_test/u17588-MayDataRestored

ws-links/lustre-rzg:
total 0